"""Parse Retrosheet game logs (2016-2025) + MLB StatsAPI (2026) into one clean game table.

Retrosheet game logs are the canonical public record of every MLB game.
Field layout: https://www.retrosheet.org/gamelogs/glfields.txt

Output: data/proc/games.parquet (one row per game, chronological)
"""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "gamelogs"
PROC = ROOT / "data" / "proc"

# Retrosheet game log column indices (0-based) for the fields we use.
IDX = {
    "date": 0, "dblhdr": 1, "dow": 2,
    "away": 3, "away_lg": 4, "away_gnum": 5,
    "home": 6, "home_lg": 7, "home_gnum": 8,
    "away_score": 9, "home_score": 10,
    "outs": 11, "daynight": 12,
    "park": 16, "attendance": 17, "duration": 18,
    # away offense
    "away_ab": 21, "away_h": 22, "away_2b": 23, "away_3b": 24, "away_hr": 25,
    "away_rbi": 26, "away_sh": 27, "away_sf": 28, "away_hbp": 29, "away_bb": 30,
    "away_ibb": 31, "away_k": 32, "away_sb": 33, "away_cs": 34, "away_gdp": 35,
    "away_lob": 42,
    # away pitching / defense
    "away_p_used": 43, "away_er": 45, "away_e": 50,
    # home offense
    "home_ab": 49 + 4, "home_h": 54, "home_2b": 55, "home_3b": 56, "home_hr": 57,
    "home_rbi": 58, "home_sh": 59, "home_sf": 60, "home_hbp": 61, "home_bb": 62,
    "home_ibb": 63, "home_k": 64, "home_sb": 65, "home_cs": 66, "home_gdp": 67,
    "home_lob": 74,
    "home_p_used": 75, "home_er": 77, "home_e": 82,
    # starting pitchers
    "away_sp_id": 101, "away_sp_name": 102,
    "home_sp_id": 103, "home_sp_name": 104,
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_retrosheet(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="latin-1") as fh:
        for rec in csv.reader(fh):
            if len(rec) < 105:
                continue
            g = lambda k: rec[IDX[k]]  # noqa: E731
            date = g("date")
            rows.append(
                {
                    "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                    "season": int(date[:4]),
                    "dblhdr": g("dblhdr"),
                    "away": g("away"),
                    "home": g("home"),
                    "away_score": int(g("away_score")),
                    "home_score": int(g("home_score")),
                    "park": g("park"),
                    "daynight": g("daynight"),
                    "attendance": _num(g("attendance")),
                    "away_hits": _num(g("away_h")),
                    "home_hits": _num(g("home_h")),
                    "away_hr": _num(g("away_hr")),
                    "home_hr": _num(g("home_hr")),
                    "away_bb": _num(g("away_bb")),
                    "home_bb": _num(g("home_bb")),
                    "away_k": _num(g("away_k")),
                    "home_k": _num(g("home_k")),
                    "away_e": _num(g("away_e")),
                    "home_e": _num(g("home_e")),
                    "away_sp": g("away_sp_id"),
                    "home_sp": g("home_sp_id"),
                    "away_sp_name": g("away_sp_name"),
                    "home_sp_name": g("home_sp_name"),
                    "source": "retrosheet",
                }
            )
    return rows


def fetch_statsapi_season(year: int) -> list[dict]:
    """2026 (or any season Retrosheet hasn't published) from the MLB Stats API."""
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R"
        f"&startDate={year}-03-01&endDate={year}-11-15"
        f"&hydrate=linescore,probablePitcher,team"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=120).read())
    out = []
    for block in data.get("dates", []):
        for gm in block.get("games", []):
            if (gm.get("status", {}) or {}).get("abstractGameState") != "Final":
                continue
            a, h = gm["teams"]["away"], gm["teams"]["home"]
            ls = gm.get("linescore", {}) or {}
            lt = ls.get("teams", {}) or {}
            out.append(
                {
                    "date": gm["officialDate"],
                    "season": year,
                    "dblhdr": str(gm.get("gameNumber", 0)),
                    "away": a["team"]["abbreviation"],
                    "home": h["team"]["abbreviation"],
                    "away_score": a.get("score"),
                    "home_score": h.get("score"),
                    "park": (gm.get("venue") or {}).get("name"),
                    "daynight": gm.get("dayNight", "")[:1].upper(),
                    "attendance": None,
                    "away_hits": (lt.get("away") or {}).get("hits"),
                    "home_hits": (lt.get("home") or {}).get("hits"),
                    "away_hr": None, "home_hr": None,
                    "away_bb": None, "home_bb": None,
                    "away_k": None, "home_k": None,
                    "away_e": (lt.get("away") or {}).get("errors"),
                    "home_e": (lt.get("home") or {}).get("errors"),
                    "away_sp": str((a.get("probablePitcher") or {}).get("id", "")),
                    "home_sp": str((h.get("probablePitcher") or {}).get("id", "")),
                    "away_sp_name": (a.get("probablePitcher") or {}).get("fullName", ""),
                    "home_sp_name": (h.get("probablePitcher") or {}).get("fullName", ""),
                    "source": "statsapi",
                }
            )
    return out


# Retrosheet uses different abbreviations than StatsAPI; normalize to Retrosheet.
TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHN",
    "CWS": "CHA", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KC": "KCA", "LAA": "ANA", "LAD": "LAN", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYN", "NYY": "NYA", "OAK": "OAK",
    "ATH": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SDN", "SF": "SFN",
    "SEA": "SEA", "STL": "SLN", "TB": "TBA", "TEX": "TEX", "TOR": "TOR",
    "WSH": "WAS",
}


def main() -> None:
    rows: list[dict] = []
    for path in sorted(RAW.glob("gl*.txt")):
        got = parse_retrosheet(path)
        rows.extend(got)
        print(f"[dataset] {path.name}: {len(got)} games")

    try:
        cur = fetch_statsapi_season(2026)
        for r in cur:
            r["away"] = TEAM_MAP.get(r["away"], r["away"])
            r["home"] = TEAM_MAP.get(r["home"], r["home"])
        rows.extend(cur)
        print(f"[dataset] statsapi 2026: {len(cur)} games")
    except Exception as err:
        print(f"[dataset] 2026 fetch failed ({err}); continuing without it")

    df = pd.DataFrame(rows)
    df = df[df["away_score"].notna() & df["home_score"].notna()]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "home", "dblhdr"]).reset_index(drop=True)
    df = df.drop_duplicates(
        subset=["date", "away", "home", "dblhdr", "away_score", "home_score"],
        keep="first",
    ).reset_index(drop=True)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_runs"] = df["home_score"] + df["away_score"]
    df["game_id"] = (
        df["date"].dt.strftime("%Y%m%d") + "_" + df["away"] + "_" + df["home"]
        + "_" + df["dblhdr"].astype(str)
    )

    dupes = df["game_id"].duplicated().sum()
    if dupes:
        print(f"[dataset] WARNING: {dupes} duplicate game_ids remain; disambiguating")
        df["game_id"] = df["game_id"] + "_" + df.groupby("game_id").cumcount().astype(str)

    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "games.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[dataset] {len(df):,} games  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"[dataset] home win rate: {df['home_win'].mean():.4f}")
    print(f"[dataset] -> {out}")


if __name__ == "__main__":
    main()
