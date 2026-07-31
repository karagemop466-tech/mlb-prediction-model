"""Collect starting lineups and build point-in-time batter quality features.

TWO HARD CONSTRAINTS, both verified rather than assumed:

1. AVAILABILITY. Lineups post roughly 3-4 hours before first pitch. Measured on
   the 2026-07-31 slate at 18:52 UTC: 3 of 15 games had lineups, and those were
   the games starting within ~4 hours. The daily workflow runs at 11:00 UTC,
   before almost every lineup is posted. So lineup features MUST degrade
   gracefully: the model has to work without them and simply improve when they
   exist. Anything else would break live prediction.

2. LEAKAGE. A batter's season line includes the game being predicted. Using it
   would leak the outcome directly. Player stats are therefore accumulated from
   per-game logs and shifted, so a game on date D only ever sees games before D.

Historical lineups come from the schedule endpoint with `hydrate=lineups`, which
returns all games for a date in one request (~1,900 requests for 2016-2026
rather than ~24,000 individual boxscore calls).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
RAW = ROOT / "data" / "raw" / "lineups"
RAW_STATS = ROOT / "data" / "raw" / "player_stats"

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"


# --------------------------------------------------------------- collection
def fetch_date(day: date, retries: int = 3) -> list[dict] | None:
    """All lineups for one date. Cached as JSON."""
    path = RAW / f"{day.isoformat()}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            path.unlink(missing_ok=True)

    url = (f"{SCHEDULE}?sportId=1&date={day.isoformat()}&gameType=R"
           "&hydrate=lineups,probablePitcher,team")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=90).read())
            rows = []
            for blk in d.get("dates", []):
                for g in blk.get("games", []):
                    lu = g.get("lineups") or {}
                    rows.append({
                        "game_pk": g["gamePk"],
                        "date": g["officialDate"],
                        "away": g["teams"]["away"]["team"].get("abbreviation"),
                        "home": g["teams"]["home"]["team"].get("abbreviation"),
                        "away_lineup": [p["id"] for p in lu.get("awayPlayers", [])],
                        "home_lineup": [p["id"] for p in lu.get("homePlayers", [])],
                        "away_sp": (g["teams"]["away"].get("probablePitcher") or {}).get("id"),
                        "home_sp": (g["teams"]["home"].get("probablePitcher") or {}).get("id"),
                    })
            RAW.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows))
            return rows
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(3 * (attempt + 1))
    return None


def season_dates(year: int):
    d, end = date(year, 3, 15), date(year, 11, 10)
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_lineups(years) -> None:
    for yr in years:
        got = games = with_lu = 0
        for d in season_dates(yr):
            rows = fetch_date(d)
            if rows is None:
                continue
            got += 1
            games += len(rows)
            with_lu += sum(1 for r in rows if len(r["home_lineup"]) >= 9)
            time.sleep(0.12)
        print(f"  [lineups] {yr}: {got} dates, {games} games, "
              f"{with_lu} with full lineups ({100*with_lu/max(games,1):.1f}%)")


# ------------------------------------------------------- player game logs
def fetch_player_gamelog(pid: int, season: int, retries: int = 2) -> pd.DataFrame | None:
    """Per-game hitting log for one player-season. Cached."""
    path = RAW_STATS / f"p{pid}_{season}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)

    url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
           f"?stats=gameLog&season={season}&group=hitting&gameType=R")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=90).read())
            stats = d.get("stats") or []
            if not stats:
                RAW_STATS.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_parquet(path)
                return pd.DataFrame()
            rows = []
            for s in stats[0].get("splits", []):
                st = s.get("stat", {})
                rows.append({
                    "date": s.get("date"),
                    "pa": st.get("plateAppearances", 0) or 0,
                    "ab": st.get("atBats", 0) or 0,
                    "h": st.get("hits", 0) or 0,
                    "hr": st.get("homeRuns", 0) or 0,
                    "bb": st.get("baseOnBalls", 0) or 0,
                    "so": st.get("strikeOuts", 0) or 0,
                    "rbi": st.get("rbi", 0) or 0,
                    "tb": st.get("totalBases", 0) or 0,
                    "hbp": st.get("hitByPitch", 0) or 0,
                    "sf": st.get("sacFlies", 0) or 0,
                })
            df = pd.DataFrame(rows)
            RAW_STATS.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            return df
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def build_player_history(seasons) -> pd.DataFrame:
    """Cumulative PRIOR-TO-DATE batting lines for every player who started.

    For each player-season, per-game logs are accumulated and shifted by one
    game, so the value attached to a game on date D reflects only games before D.
    """
    needed: dict[int, set[int]] = {}
    for f in sorted(RAW.glob("*.json")):
        yr = int(f.stem[:4])
        if yr not in seasons:
            continue
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        for r in rows:
            for side in ("away_lineup", "home_lineup"):
                for pid in r[side]:
                    needed.setdefault(yr, set()).add(pid)

    frames = []
    for yr, pids in sorted(needed.items()):
        print(f"  [player-stats] {yr}: {len(pids)} distinct starters")
        for i, pid in enumerate(sorted(pids), 1):
            df = fetch_player_gamelog(pid, yr)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["player_id"] = pid
            df["season"] = yr
            frames.append(df)
            if i % 200 == 0:
                print(f"      {i}/{len(pids)}")
            time.sleep(0.05)

    if not frames:
        return pd.DataFrame()

    allp = pd.concat(frames, ignore_index=True)
    allp["date"] = pd.to_datetime(allp["date"])
    allp = allp.sort_values(["player_id", "season", "date"])

    # Cumulative, shifted -> strictly prior games only.
    g = allp.groupby(["player_id", "season"], sort=False)
    for c in ("pa", "ab", "h", "hr", "bb", "so", "rbi", "tb", "hbp", "sf"):
        allp[f"c_{c}"] = g[c].transform(lambda s: s.shift(1).cumsum())

    ab = allp["c_ab"].replace(0, np.nan)
    pa = allp["c_pa"].replace(0, np.nan)
    obp_den = (allp["c_ab"] + allp["c_bb"] + allp["c_hbp"] + allp["c_sf"]).replace(0, np.nan)
    allp["p_avg"] = allp["c_h"] / ab
    allp["p_obp"] = (allp["c_h"] + allp["c_bb"] + allp["c_hbp"]) / obp_den
    allp["p_slg"] = allp["c_tb"] / ab
    allp["p_ops"] = allp["p_obp"] + allp["p_slg"]
    allp["p_iso"] = allp["p_slg"] - allp["p_avg"]
    allp["p_bb_rate"] = allp["c_bb"] / pa
    allp["p_k_rate"] = allp["c_so"] / pa
    allp["p_hr_rate"] = allp["c_hr"] / pa
    allp["p_pa"] = allp["c_pa"]

    keep = ["player_id", "season", "date", "p_avg", "p_obp", "p_slg", "p_ops",
            "p_iso", "p_bb_rate", "p_k_rate", "p_hr_rate", "p_pa"]
    out = allp[keep]
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "player_batting_history.parquet", index=False)
    print(f"  [player-stats] {len(out):,} player-game rows")
    return out


def main() -> None:
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    years = [int(a) for a in args] or list(range(2016, 2027))
    if "--lineups" in sys.argv:
        download_lineups(years)
    elif "--stats" in sys.argv:
        build_player_history(set(years))
    else:
        download_lineups(years)
        build_player_history(set(years))


if __name__ == "__main__":
    main()
