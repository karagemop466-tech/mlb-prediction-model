"""Team-level Statcast features from Baseball Savant pitch-level data.

Savant's leaderboard CSV endpoint ignores type=team (always returns players), so
we aggregate the authoritative pitch-level `statcast_search` feed to game-team
quality-of-contact metrics, then build ROLLING (prior-games-only) team features.

Metrics per team-game:
  xwoba_off / xwoba_def : estimated wOBA on batted balls, offense and defense
  ev_off / ev_def       : average exit velocity
  barrel_off / barrel_def : barrel rate (launch_speed_angle == 6)
  hardhit_off / hardhit_def : share of batted balls >= 95 mph

Downloads are chunked by date range and cached to data/raw/statcast/.
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "statcast"
PROC = ROOT / "data" / "proc"

SAVANT_TO_RETRO = {
    "AZ": "ARI", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CWS": "CHA", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KC": "KCA", "LAA": "ANA", "LAD": "LAN",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYN", "NYY": "NYA",
    "OAK": "OAK", "ATH": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SDN",
    "SF": "SFN", "SEA": "SEA", "STL": "SLN", "TB": "TBA", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WAS", "WAS": "WAS",
}

USE_COLS = [
    "game_date", "game_pk", "home_team", "away_team", "inning_topbot",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle", "woba_value", "events",
]


def fetch_chunk(d0: date, d1: date, retries: int = 3) -> pd.DataFrame | None:
    path = RAW / f"sc_{d0.isoformat()}_{d1.isoformat()}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)

    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C"
        f"&game_date_gt={d0.isoformat()}&game_date_lt={d1.isoformat()}&type=details"
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
            raw = urllib.request.urlopen(req, timeout=300).read()
            if len(raw) < 2000:
                return None
            df = pd.read_csv(io.BytesIO(raw), low_memory=False,
                             usecols=lambda c: c in USE_COLS)
            df = df[df["launch_speed"].notna() | df["events"].notna()]
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            return df
        except Exception as err:
            if attempt == retries - 1:
                print(f"    {d0}..{d1} failed: {type(err).__name__}")
                return None
            time.sleep(4 * (attempt + 1))
    return None


def season_windows(year: int, step_days: int = 10):
    start, end = date(year, 3, 15), date(year, 11, 10)
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=step_days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def download(years) -> None:
    for yr in years:
        wins = list(season_windows(yr))
        got = 0
        for d0, d1 in wins:
            df = fetch_chunk(d0, d1)
            if df is not None:
                got += len(df)
            time.sleep(0.6)
        print(f"  [statcast] {yr}: {got:,} pitch rows cached")


def aggregate() -> pd.DataFrame:
    files = sorted(RAW.glob("sc_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["launch_speed"].notna()]
    if df.empty:
        return pd.DataFrame()

    # Batting team = away when top of inning, home when bottom.
    df["bat_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])
    df["pit_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    df["barrel"] = (df["launch_speed_angle"] == 6).astype(float)
    df["hardhit"] = (df["launch_speed"] >= 95).astype(float)
    df["game_date"] = pd.to_datetime(df["game_date"])

    off = df.groupby(["game_date", "game_pk", "bat_team"]).agg(
        xwoba_off=("estimated_woba_using_speedangle", "mean"),
        ev_off=("launch_speed", "mean"),
        barrel_off=("barrel", "mean"),
        hardhit_off=("hardhit", "mean"),
        bip_off=("launch_speed", "size"),
    ).reset_index().rename(columns={"bat_team": "team"})

    dfn = df.groupby(["game_date", "game_pk", "pit_team"]).agg(
        xwoba_def=("estimated_woba_using_speedangle", "mean"),
        ev_def=("launch_speed", "mean"),
        barrel_def=("barrel", "mean"),
        hardhit_def=("hardhit", "mean"),
    ).reset_index().rename(columns={"pit_team": "team"})

    tg = off.merge(dfn, on=["game_date", "game_pk", "team"], how="outer")
    tg["team"] = tg["team"].map(lambda t: SAVANT_TO_RETRO.get(str(t).strip(), None))
    tg = tg[tg["team"].notna()].sort_values(["team", "game_date"])

    # Rolling, shifted -> strictly prior games only.
    metrics = ["xwoba_off", "ev_off", "barrel_off", "hardhit_off",
               "xwoba_def", "ev_def", "barrel_def", "hardhit_def"]
    parts = []
    for team, g in tg.groupby("team", sort=False):
        g = g.sort_values("game_date").copy()
        for m in metrics:
            g[f"sc_{m}_30"] = g[m].shift(1).rolling(30, min_periods=8).mean()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    keep = ["game_date", "team"] + [f"sc_{m}_30" for m in metrics]
    out = out[keep].rename(columns={"game_date": "date"})

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "statcast_team.parquet", index=False)
    print(f"  [statcast] {len(out):,} team-game rows -> data/proc/statcast_team.parquet")
    return out


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or list(range(2016, 2027))
    download(yrs)
    aggregate()
