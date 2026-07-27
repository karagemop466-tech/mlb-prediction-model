"""Individual starting-pitcher Statcast features.

Team-level Statcast failed to help (see FINDINGS.md) because quality-of-contact
is already embedded in team run differential. Pitcher-level is different: the
starter is the single largest identifiable factor in one game, and his recent
contact-quality profile is not captured by team form at all.

IMPORTANT — Savant's CSV endpoint hard-caps at 25,000 rows per request and
truncates silently. A full slate is ~3,400 pitches/day, so windows must stay
small (default 5 days). Every chunk is checked against the cap and split if hit.

Per start we compute, for the pitcher only:
    xwoba_against   estimated wOBA allowed on batted balls
    ev_against      average exit velocity allowed
    barrel_rate     barrels allowed per batted ball
    hardhit_rate    share of batted balls >= 95 mph
    whiff_rate      swinging strikes per swing
    k_rate / bb_rate per plate appearance
    batters_faced   workload

Those are then rolled over the pitcher's PRIOR starts (shift(1) before rolling),
so a start never sees its own outcome. Retrosheet player IDs are mapped to MLBAM
IDs via the Chadwick Bureau register.
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
RAW = ROOT / "data" / "raw" / "pitcher_sc"
REF = ROOT / "data" / "raw" / "ref"
PROC = ROOT / "data" / "proc"

SAVANT_ROW_CAP = 25000

USE_COLS = [
    "game_date", "game_pk", "pitcher", "batter", "events", "description",
    "stand", "p_throws", "inning_topbot", "home_team", "away_team",
    "launch_speed", "launch_speed_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom",
]

SWING_DESC = {
    "hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip"}


def build_crosswalk() -> pd.DataFrame:
    """Chadwick Bureau register: Retrosheet ID <-> MLBAM ID."""
    path = REF / "crosswalk.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for i in range(16):
        url = ("https://raw.githubusercontent.com/chadwickbureau/register/"
               f"master/data/people-{i:x}.csv")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            raw = urllib.request.urlopen(req, timeout=180).read()
            frames.append(pd.read_csv(
                io.BytesIO(raw),
                usecols=["key_mlbam", "key_retro", "name_last", "name_first"],
                low_memory=False))
        except Exception as err:
            print(f"  crosswalk part {i:x} failed: {type(err).__name__}")
    x = pd.concat(frames, ignore_index=True)
    x = x[x.key_mlbam.notna() & x.key_retro.notna()].copy()
    x["key_mlbam"] = x["key_mlbam"].astype(int)
    REF.mkdir(parents=True, exist_ok=True)
    x.to_parquet(path, index=False)
    print(f"  [crosswalk] {len(x):,} players")
    return x


def fetch_chunk(d0: date, d1: date, depth: int = 0) -> pd.DataFrame | None:
    """Download one date window, splitting recursively if the row cap is hit."""
    tag = f"{d0.isoformat()}_{d1.isoformat()}"
    path = RAW / f"p_{tag}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)

    url = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C"
           f"&game_date_gt={d0.isoformat()}&game_date_lt={d1.isoformat()}&type=details")
    df = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
            raw = urllib.request.urlopen(req, timeout=420).read()
            if len(raw) < 2000:
                return None
            df = pd.read_csv(io.BytesIO(raw), low_memory=False,
                             usecols=lambda c: c in USE_COLS)
            break
        except Exception as err:
            if attempt == 2:
                print(f"    {tag} failed: {type(err).__name__}")
                return None
            time.sleep(5 * (attempt + 1))

    if df is None:
        return None

    # Truncation guard: split the window and recurse.
    if len(df) >= SAVANT_ROW_CAP and d0 < d1 and depth < 4:
        mid = d0 + (d1 - d0) / 2
        print(f"    {tag} hit row cap; splitting")
        a = fetch_chunk(d0, mid, depth + 1)
        b = fetch_chunk(mid + timedelta(days=1), d1, depth + 1)
        parts = [x for x in (a, b) if x is not None]
        return pd.concat(parts, ignore_index=True) if parts else None

    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def windows(year: int, step: int = 5):
    start, end = date(year, 3, 15), date(year, 11, 10)
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=step - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def download(years) -> None:
    for yr in years:
        got = capped = 0
        for d0, d1 in windows(yr):
            df = fetch_chunk(d0, d1)
            if df is not None:
                got += len(df)
                if len(df) >= SAVANT_ROW_CAP:
                    capped += 1
            time.sleep(0.5)
        print(f"  [pitcher-sc] {yr}: {got:,} pitches" +
              (f"  ({capped} windows still capped)" if capped else ""))


def _agg_one(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a single chunk to pitcher-game rows (memory-bounded)."""
    df = df[df["pitcher"].notna()]
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher"] = df["pitcher"].astype(int)

    desc = df["description"].fillna("")
    df["is_swing"] = desc.isin(SWING_DESC).astype("float32")
    df["is_whiff"] = desc.isin(WHIFF_DESC).astype("float32")
    ev = df["launch_speed"]
    df["is_bip"] = ev.notna().astype("float32")
    df["barrel"] = (df["launch_speed_angle"] == 6).astype("float32")
    df["hardhit"] = (ev >= 95).astype("float32")

    ev_ = df["events"].fillna("")
    df["is_pa"] = (df["woba_denom"].fillna(0) > 0).astype("float32")
    df["is_k"] = ev_.str.startswith("strikeout").astype("float32")
    df["is_bb"] = ev_.isin(["walk", "intent_walk"]).astype("float32")

    return df.groupby(["game_date", "game_pk", "pitcher"], sort=False).agg(
        pitches=("is_swing", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        bip=("is_bip", "sum"),
        barrels=("barrel", "sum"),
        hardhits=("hardhit", "sum"),
        ev_against=("launch_speed", "mean"),
        xwoba_against=("estimated_woba_using_speedangle", "mean"),
        pa=("is_pa", "sum"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
    ).reset_index()


def aggregate() -> pd.DataFrame:
    """Chunk-by-chunk aggregation. Never holds all pitches in memory at once.

    The full pitch feed is ~3.4M rows; concatenating it exhausts a 2GB worker.
    Each cached chunk is reduced to pitcher-game rows first, which is ~500x
    smaller, and only those are combined.
    """
    files = sorted(RAW.glob("p_*.parquet"))
    if not files:
        print("  [pitcher-sc] no cached chunks")
        return pd.DataFrame()

    partials = []
    for i, f in enumerate(files, 1):
        try:
            chunk = pd.read_parquet(f, columns=[c for c in USE_COLS])
        except Exception:
            try:
                chunk = pd.read_parquet(f)
            except Exception:
                continue
        a = _agg_one(chunk)
        if not a.empty:
            partials.append(a)
        del chunk
        if i % 40 == 0:
            print(f"    aggregated {i}/{len(files)} chunks")

    if not partials:
        return pd.DataFrame()

    agg = pd.concat(partials, ignore_index=True)
    del partials

    # Chunk boundaries can split a game; sum the pieces back together.
    agg = agg.groupby(["game_date", "game_pk", "pitcher"], as_index=False).agg(
        pitches=("pitches", "sum"), swings=("swings", "sum"),
        whiffs=("whiffs", "sum"), bip=("bip", "sum"),
        barrels=("barrels", "sum"), hardhits=("hardhits", "sum"),
        ev_against=("ev_against", "mean"), xwoba_against=("xwoba_against", "mean"),
        pa=("pa", "sum"), k=("k", "sum"), bb=("bb", "sum"),
    )

    # The starter is the highest-pitch-count arm per side (top 2 per game).
    agg = agg.sort_values(["game_pk", "pitches"], ascending=[True, False])
    agg["rank_in_game"] = agg.groupby("game_pk").cumcount()
    starters = agg[(agg.rank_in_game <= 1) & (agg.pitches >= 30)].copy()

    starters["whiff_rate"] = starters.whiffs / starters.swings.replace(0, np.nan)
    starters["barrel_rate"] = starters.barrels / starters.bip.replace(0, np.nan)
    starters["hardhit_rate"] = starters.hardhits / starters.bip.replace(0, np.nan)
    starters["k_rate"] = starters.k / starters.pa.replace(0, np.nan)
    starters["bb_rate"] = starters.bb / starters.pa.replace(0, np.nan)

    metrics = ["xwoba_against", "ev_against", "barrel_rate", "hardhit_rate",
               "whiff_rate", "k_rate", "bb_rate"]

    starters = starters.sort_values(["pitcher", "game_date"])
    out = starters.copy()
    grp = out.groupby("pitcher", sort=False)
    for m in metrics:
        out[f"sp_{m}_10"] = grp[m].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    out["sp_career_starts"] = grp.cumcount()

    keep = (["game_date", "game_pk", "pitcher", "sp_career_starts"]
            + [f"sp_{m}_10" for m in metrics])
    out = out[keep].rename(columns={"game_date": "date", "pitcher": "mlbam"})

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "pitcher_statcast.parquet", index=False)
    cov = int(out[f"sp_{metrics[0]}_10"].notna().sum())
    print(f"  [pitcher-sc] {len(out):,} starts, {cov:,} with rolling history "
          f"-> data/proc/pitcher_statcast.parquet")
    return out


if __name__ == "__main__":
    build_crosswalk()
    yrs = [int(a) for a in sys.argv[1:]] or list(range(2016, 2027))
    download(yrs)
    aggregate()
