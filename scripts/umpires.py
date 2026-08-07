"""Home-plate umpire assignments and point-in-time umpire tendencies.

WHY THIS IS THE NEXT CANDIDATE
------------------------------
Eight feature families have been tested. The pattern is consistent:

    features describing WHO IS PLAYING  -> redundant with team rolling form
    features describing STRUCTURE       -> not redundant (weather, game phase)

The home-plate umpire is structural. His strike zone is a property of the game
environment, not of either team, and it cannot be absorbed by team run
production the way lineup quality is. A wide zone suppresses offense for BOTH
sides simultaneously, which is exactly the signature weather had.

Umpire crews are also assigned on a rotation that is effectively independent of
team quality, so this should not proxy for "good team plays bad team".

LEAKAGE CONTROL
---------------
An umpire's tendency must be computed from his PRIOR games only. Season-long
umpire stats would include the game being predicted. All tendencies here are
cumulative-then-shifted, and additionally shrunk toward the league mean because
an umpire with 4 prior games has a meaningless average.

Data: one schedule request per date with hydrate=officials returns the full
crew for every game that day (~1,900 requests for 2016-2026).
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
RAW = ROOT / "data" / "raw" / "umpires"
PROC = ROOT / "data" / "proc"

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"

UMP_COLS = ["ump_runs_tend", "ump_k_tend", "ump_bb_tend", "ump_hwin_tend",
            "ump_games"]


def fetch_date(day: date, retries: int = 3) -> list[dict] | None:
    path = RAW / f"{day.isoformat()}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            path.unlink(missing_ok=True)

    url = (f"{SCHEDULE}?sportId=1&date={day.isoformat()}&gameType=R"
           "&hydrate=officials,team")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=90).read())
            rows = []
            for blk in d.get("dates", []):
                for g in blk.get("games", []):
                    hp = None
                    for o in g.get("officials", []) or []:
                        if o.get("officialType") == "Home Plate":
                            hp = o["official"]
                            break
                    rows.append({
                        "date": g["officialDate"],
                        "game_pk": g["gamePk"],
                        "away": g["teams"]["away"]["team"].get("abbreviation"),
                        "home": g["teams"]["home"]["team"].get("abbreviation"),
                        "hp_id": hp["id"] if hp else None,
                        "hp_name": hp["fullName"] if hp else None,
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


def download(years) -> None:
    for yr in years:
        got = games = with_hp = 0
        for d in season_dates(yr):
            rows = fetch_date(d)
            if rows is None:
                continue
            got += 1
            games += len(rows)
            with_hp += sum(1 for r in rows if r["hp_id"])
            time.sleep(0.12)
        print(f"  [umpires] {yr}: {got} dates, {games} games, "
              f"{with_hp} with HP umpire ({100*with_hp/max(games,1):.1f}%)")


# ------------------------------------------------------------- tendencies
SHRINK_K = 40.0     # games at which an umpire's own average is trusted 50/50


def build_tendencies() -> pd.DataFrame:
    """Prior-games-only umpire tendencies, empirical-Bayes shrunk.

    Four tendencies, each expressed as a deviation from the league average of
    the same quantity:
        runs   total runs in his games
        k      strikeouts per game
        bb     walks per game
        hwin   home win rate in his games
    """
    rows = []
    for f in sorted(RAW.glob("*.json")):
        try:
            rows.extend(json.loads(f.read_text()))
        except Exception:
            continue
    if not rows:
        print("  [umpires] no cached assignments")
        return pd.DataFrame()

    ump = pd.DataFrame(rows)
    ump = ump[ump.hp_id.notna()].copy()
    ump["date"] = pd.to_datetime(ump["date"])
    ump["hp_id"] = ump["hp_id"].astype(int)

    games = pd.read_parquet(PROC / "games.parquet")
    games["date"] = pd.to_datetime(games["date"])
    from build_dataset import TEAM_MAP
    ump["home_r"] = ump["home"].map(lambda x: TEAM_MAP.get(x, x))
    ump["away_r"] = ump["away"].map(lambda x: TEAM_MAP.get(x, x))
    ump = ump.drop_duplicates(["date", "home_r", "away_r"])

    g = games[["date", "home", "away", "home_score", "away_score", "home_win",
               "home_k", "away_k", "home_bb", "away_bb"]].copy()
    m = ump.merge(g, left_on=["date", "home_r", "away_r"],
                  right_on=["date", "home", "away"], how="inner",
                  suffixes=("", "_g"))
    if m.empty:
        print("  [umpires] no overlap with games table")
        return pd.DataFrame()

    m["tot_runs"] = m.home_score + m.away_score
    m["tot_k"] = m.home_k.fillna(np.nan) + m.away_k.fillna(np.nan)
    m["tot_bb"] = m.home_bb.fillna(np.nan) + m.away_bb.fillna(np.nan)
    m = m.sort_values(["hp_id", "date"]).reset_index(drop=True)

    grp = m.groupby("hp_id", sort=False)
    for src, dst in (("tot_runs", "runs"), ("tot_k", "k"),
                     ("tot_bb", "bb"), ("home_win", "hwin")):
        m[f"c_{dst}"] = grp[src].transform(lambda s: s.shift(1).cumsum())
    m["c_n"] = grp.cumcount()

    # League baselines from prior games only, per season.
    m["season"] = m["date"].dt.year
    lg = {}
    for col in ("tot_runs", "tot_k", "tot_bb", "home_win"):
        lg[col] = m.groupby("season")[col].transform(
            lambda s: s.shift(1).expanding(min_periods=50).mean())

    n = m["c_n"].astype(float)
    for src, dst in (("tot_runs", "runs"), ("tot_k", "k"),
                     ("tot_bb", "bb"), ("home_win", "hwin")):
        own = m[f"c_{dst}"] / n.replace(0, np.nan)
        base = lg[src]
        shrunk = (n * own.fillna(base) + SHRINK_K * base) / (n + SHRINK_K)
        m[f"ump_{dst}_tend"] = shrunk - base      # deviation from league

    m["ump_games"] = n

    out = m[["date", "home_r", "away_r", "hp_id", "hp_name"] +
            [f"ump_{d}_tend" for d in ("runs", "k", "bb", "hwin")] +
            ["ump_games"]].rename(columns={"home_r": "home", "away_r": "away"})

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "umpire_features.parquet", index=False)
    cov = out["ump_runs_tend"].notna().mean()
    print(f"  [umpires] {len(out):,} games, {cov:.1%} with tendency, "
          f"{out.hp_id.nunique()} distinct umpires")
    return out


def main() -> None:
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    years = [int(a) for a in args] or list(range(2016, 2027))
    if "--build" in sys.argv:
        build_tendencies()
    elif "--download" in sys.argv:
        download(years)
    else:
        download(years)
        build_tendencies()


if __name__ == "__main__":
    main()
