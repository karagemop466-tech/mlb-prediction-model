"""Pitch-arsenal vs lineup matchup features.

See DESIGN_MATCHUP.md. Short version: batter-vs-specific-pitcher history is
unusable (~1.6 PA per pair), so the matchup is factored through pitch TYPES,
which have usable sample (~120-150 PA per batter-season).

    matchup = sum_p  arsenal_share[pitcher, p] * batter_quality[batter, p]

Two things make or break this:

1. SHRINKAGE. A batter with 12 PA vs sweepers who went 4-for-9 shows a .500
   wOBA that is pure noise. Every batter-vs-pitch-type value is empirical-Bayes
   shrunk toward the league mean for that pitch type. Without this the feature
   family is a noise generator.

2. LEAKAGE. Arsenal shares and batter splits are both computed from strictly
   prior games (.shift(1) before any rolling/cumulative operation).

Downloads pitch-level data WITH pitch_type, which the earlier pitcher_statcast
cache omitted.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "arsenal"
PROC = ROOT / "data" / "proc"

SAVANT_ROW_CAP = 25000

USE_COLS = [
    "game_date", "game_pk", "pitcher", "batter", "pitch_type",
    "stand", "p_throws", "events", "description",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
]

# Group rare variants into their parent families. Keeps sample per cell usable.
PITCH_FAMILY = {
    "FF": "FF", "FA": "FF",                    # four-seam
    "SI": "SI", "FT": "SI",                    # sinker / two-seam
    "FC": "FC",                                # cutter
    "SL": "SL", "ST": "SL", "SV": "SL",        # slider / sweeper / slurve
    "CU": "CU", "KC": "CU", "CS": "CU",        # curveball family
    "CH": "CH", "FS": "CH", "SC": "CH",        # change / split
    "KN": "KN", "EP": "KN", "FO": "CH",
}
FAMILIES = ["FF", "SI", "FC", "SL", "CU", "CH", "KN"]


def fetch_chunk(d0: date, d1: date, depth: int = 0) -> pd.DataFrame | None:
    tag = f"{d0.isoformat()}_{d1.isoformat()}"
    path = RAW / f"a_{tag}.parquet"
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
        except Exception:
            if attempt == 2:
                print(f"    {tag} failed")
                return None
            time.sleep(6 * (attempt + 1))
    if df is None:
        return None

    # Savant silently truncates at 25k rows; split and recurse.
    if len(df) >= SAVANT_ROW_CAP and d0 < d1 and depth < 4:
        mid = d0 + (d1 - d0) / 2
        a = fetch_chunk(d0, mid, depth + 1)
        b = fetch_chunk(mid + timedelta(days=1), d1, depth + 1)
        parts = [x for x in (a, b) if x is not None]
        return pd.concat(parts, ignore_index=True) if parts else None

    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def windows(year: int, step: int = 4):
    d, end = date(year, 3, 15), date(year, 11, 10)
    while d < end:
        nxt = min(d + timedelta(days=step - 1), end)
        yield d, nxt
        d = nxt + timedelta(days=1)


def download(years) -> None:
    for yr in years:
        got = 0
        for d0, d1 in windows(yr):
            df = fetch_chunk(d0, d1)
            if df is not None:
                got += len(df)
            time.sleep(0.4)
        print(f"  [arsenal] {yr}: {got:,} pitches")


# ------------------------------------------------------------------ build
def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["pitch_type"].notna() & df["pitcher"].notna()].copy()
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["fam"] = df["pitch_type"].map(PITCH_FAMILY)
    df = df[df["fam"].notna()]
    df["pitcher"] = df["pitcher"].astype(int)
    df["batter"] = df["batter"].astype(int)
    df["is_pa"] = (df["woba_denom"].fillna(0) > 0).astype("float32")
    return df


def build_arsenals() -> pd.DataFrame:
    """Rolling pitch-mix share per pitcher-game, prior games only."""
    files = sorted(RAW.glob("a_*.parquet"))
    if not files:
        print("  [arsenal] no cached chunks")
        return pd.DataFrame()

    parts = []
    for i, f in enumerate(files, 1):
        try:
            c = pd.read_parquet(f, columns=[x for x in USE_COLS])
        except Exception:
            continue
        c = _prep(c)
        if c.empty:
            continue
        agg = (c.groupby(["game_date", "game_pk", "pitcher", "fam"])
                 .size().reset_index(name="n"))
        parts.append(agg)
        del c
        if i % 60 == 0:
            print(f"    arsenals {i}/{len(files)}")
    if not parts:
        return pd.DataFrame()

    a = pd.concat(parts, ignore_index=True)
    a = a.groupby(["game_date", "game_pk", "pitcher", "fam"], as_index=False)["n"].sum()
    wide = a.pivot_table(index=["game_date", "game_pk", "pitcher"],
                         columns="fam", values="n", fill_value=0).reset_index()
    for fam in FAMILIES:
        if fam not in wide:
            wide[fam] = 0

    wide["total"] = wide[FAMILIES].sum(axis=1)
    wide = wide[wide["total"] >= 30]        # starters only
    wide = wide.sort_values(["pitcher", "game_date"])

    g = wide.groupby("pitcher", sort=False)
    out = wide[["game_date", "game_pk", "pitcher"]].copy()
    roll_tot = g["total"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).sum())
    for fam in FAMILIES:
        rs = g[fam].transform(lambda s: s.shift(1).rolling(10, min_periods=3).sum())
        out[f"arr_{fam}"] = rs / roll_tot.replace(0, np.nan)
    out["arr_n"] = roll_tot

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "pitcher_arsenal.parquet", index=False)
    print(f"  [arsenal] {len(out):,} pitcher-game arsenals")
    return out


def build_batter_splits() -> pd.DataFrame:
    """Cumulative batter xwOBA by pitch family and pitcher handedness.

    Empirical-Bayes shrunk toward the league mean for that (family, hand) cell.
    """
    files = sorted(RAW.glob("a_*.parquet"))
    if not files:
        return pd.DataFrame()

    parts = []
    for i, f in enumerate(files, 1):
        try:
            c = pd.read_parquet(f, columns=[x for x in USE_COLS])
        except Exception:
            continue
        c = _prep(c)
        if c.empty:
            continue
        c = c[c["is_pa"] > 0]
        if c.empty:
            continue
        agg = (c.groupby(["game_date", "batter", "fam", "p_throws"])
                 .agg(pa=("is_pa", "sum"),
                      xw=("estimated_woba_using_speedangle", "sum"),
                      xn=("estimated_woba_using_speedangle", "count"))
                 .reset_index())
        parts.append(agg)
        del c
        if i % 60 == 0:
            print(f"    splits {i}/{len(files)}")
    if not parts:
        return pd.DataFrame()

    b = pd.concat(parts, ignore_index=True)
    b["season"] = b["game_date"].dt.year
    b = b.sort_values(["batter", "fam", "p_throws", "game_date"])

    # Cumulative prior-to-date, within season.
    key = ["batter", "fam", "p_throws", "season"]
    g = b.groupby(key, sort=False)
    b["c_pa"] = g["pa"].transform(lambda s: s.shift(1).cumsum())
    b["c_xw"] = g["xw"].transform(lambda s: s.shift(1).cumsum())
    b["c_xn"] = g["xn"].transform(lambda s: s.shift(1).cumsum())

    b["raw_xwoba"] = b["c_xw"] / b["c_xn"].replace(0, np.nan)

    # League mean per (family, hand, season) from the same prior-only totals.
    lg = (b.groupby(["fam", "p_throws", "season"])
            .apply(lambda d: d["c_xw"].sum() / max(d["c_xn"].sum(), 1),
                   include_groups=False)
            .rename("lg_xwoba").reset_index())
    b = b.merge(lg, on=["fam", "p_throws", "season"], how="left")

    # Empirical-Bayes shrinkage. k ~ number of observations at which we trust
    # the batter's own value as much as the league prior. 60 batted balls is a
    # conservative, standard choice for xwOBA stabilisation.
    K = 60.0
    n = b["c_xn"].fillna(0)
    b["bx_xwoba"] = ((n * b["raw_xwoba"].fillna(b["lg_xwoba"]) + K * b["lg_xwoba"])
                     / (n + K))
    b["bx_n"] = n

    out = b[["game_date", "batter", "fam", "p_throws", "bx_xwoba", "bx_n",
             "lg_xwoba"]].rename(columns={"game_date": "date"})
    out.to_parquet(PROC / "batter_pitch_splits.parquet", index=False)
    print(f"  [arsenal] {len(out):,} batter-family-hand rows")
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    years = [int(a) for a in args] or list(range(2021, 2027))
    if "--download" in sys.argv:
        download(years)
    elif "--build" in sys.argv:
        build_arsenals()
        build_batter_splits()
    else:
        download(years)
        build_arsenals()
        build_batter_splits()


if __name__ == "__main__":
    main()
