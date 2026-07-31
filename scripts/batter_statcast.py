"""Player-level Statcast contact quality, aggregated to the posted lineup.

WHY THIS IS A DIFFERENT HYPOTHESIS FROM LINEUP OPS
--------------------------------------------------
Lineup OPS was rejected because it correlates 0.51 with team rolling runs
scored -- traditional batting lines are largely redundant with team run totals.

Contact quality is not the same thing. xwOBA and barrel rate measure what a
hitter DESERVED based on exit velocity and launch angle, stripping out the
sequencing luck and defensive positioning that inflate or deflate OPS over a
few weeks. Two lineups with identical OPS can have very different underlying
contact profiles, and the difference is exactly the part team run totals cannot
see.

Reuses the pitch-level chunks already cached by pitcher_statcast.py (209 chunks,
~3.4M pitches, batter column included), so no new downloads.

Same memory discipline as pitcher_statcast: each chunk is reduced to
batter-game rows before concatenation, because holding 3.4M rows exhausts the
2 GB worker (a failure that previously produced silently truncated output).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "pitcher_sc"
PROC = ROOT / "data" / "proc"

USE_COLS = ["game_date", "game_pk", "batter", "events", "description",
            "launch_speed", "launch_speed_angle",
            "estimated_woba_using_speedangle", "woba_value", "woba_denom"]

SWING_DESC = {"hit_into_play", "foul", "swinging_strike",
              "swinging_strike_blocked", "foul_tip", "foul_bunt",
              "missed_bunt", "bunt_foul_tip"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "missed_bunt",
              "foul_tip"}

BATTER_METRICS = ["bx_xwoba", "bx_ev", "bx_barrel", "bx_hardhit", "bx_whiff"]


def _agg_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["batter"].notna()]
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["batter"] = df["batter"].astype(int)

    desc = df["description"].fillna("")
    df["is_swing"] = desc.isin(SWING_DESC).astype("float32")
    df["is_whiff"] = desc.isin(WHIFF_DESC).astype("float32")
    ev = df["launch_speed"]
    df["is_bip"] = ev.notna().astype("float32")
    df["barrel"] = (df["launch_speed_angle"] == 6).astype("float32")
    df["hardhit"] = (ev >= 95).astype("float32")
    df["is_pa"] = (df["woba_denom"].fillna(0) > 0).astype("float32")

    return df.groupby(["game_date", "game_pk", "batter"], sort=False).agg(
        pitches=("is_swing", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        bip=("is_bip", "sum"),
        barrels=("barrel", "sum"),
        hardhits=("hardhit", "sum"),
        ev_sum=("launch_speed", "sum"),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_n=("estimated_woba_using_speedangle", "count"),
        pa=("is_pa", "sum"),
    ).reset_index()


def aggregate() -> pd.DataFrame:
    files = sorted(RAW.glob("p_*.parquet"))
    if not files:
        print("  [batter-sc] no cached pitch chunks")
        return pd.DataFrame()

    parts = []
    for i, f in enumerate(files, 1):
        try:
            c = pd.read_parquet(f, columns=[x for x in USE_COLS])
        except Exception:
            try:
                c = pd.read_parquet(f)
            except Exception:
                continue
        a = _agg_chunk(c)
        if not a.empty:
            parts.append(a)
        del c
        if i % 50 == 0:
            print(f"    {i}/{len(files)} chunks")

    if not parts:
        return pd.DataFrame()
    bg = pd.concat(parts, ignore_index=True)
    del parts

    # Chunk boundaries can split a game.
    bg = bg.groupby(["game_date", "game_pk", "batter"], as_index=False).sum()

    # Rolling over the batter's PRIOR games only.
    bg = bg.sort_values(["batter", "game_date"])
    g = bg.groupby("batter", sort=False)

    def roll(col, window=50, minp=15):
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=minp).sum())

    for c in ("bip", "barrels", "hardhits", "ev_sum", "xwoba_sum", "xwoba_n",
              "swings", "whiffs", "pa"):
        bg[f"r_{c}"] = roll(c)

    bip = bg["r_bip"].replace(0, np.nan)
    bg["bx_xwoba"] = bg["r_xwoba_sum"] / bg["r_xwoba_n"].replace(0, np.nan)
    bg["bx_ev"] = bg["r_ev_sum"] / bip
    bg["bx_barrel"] = bg["r_barrels"] / bip
    bg["bx_hardhit"] = bg["r_hardhits"] / bip
    bg["bx_whiff"] = bg["r_whiffs"] / bg["r_swings"].replace(0, np.nan)
    bg["bx_pa"] = bg["r_pa"]

    out = bg[["game_date", "batter"] + BATTER_METRICS + ["bx_pa"]].rename(
        columns={"game_date": "date", "batter": "player_id"})
    out = out.drop_duplicates(["date", "player_id"], keep="last")

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "batter_statcast.parquet", index=False)
    cov = int(out["bx_xwoba"].notna().sum())
    print(f"  [batter-sc] {len(out):,} batter-game rows, {cov:,} with history")
    return out


# ------------------------------------------------------ lineup aggregation
ORDER_WEIGHTS = np.array([1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91, 0.88])

LINEUP_SC_COLS = ["lsc_xwoba", "lsc_ev", "lsc_barrel", "lsc_hardhit",
                  "lsc_whiff", "lsc_n_known"]


def build_lineup_statcast() -> pd.DataFrame:
    """Aggregate batter Statcast to each posted lineup."""
    bpath = PROC / "batter_statcast.parquet"
    if not bpath.exists():
        print("[lineup-sc] run aggregate() first")
        return pd.DataFrame()
    bx = pd.read_parquet(bpath)
    bx["date"] = pd.to_datetime(bx["date"])
    lut = {(r.player_id, r.date): r for r in bx.itertuples(index=False)}
    print(f"[lineup-sc] batter-date records: {len(lut):,}")

    import glob as _g
    rows = []
    for f in sorted(_g.glob(str(ROOT / "data" / "raw" / "lineups" / "*.json"))):
        try:
            games = json.loads(Path(f).read_text())
        except Exception:
            continue
        for gm in games:
            day = pd.Timestamp(gm["date"])
            rec = {"game_pk": gm["game_pk"], "date": day,
                   "home_abbr": gm["home"], "away_abbr": gm["away"]}
            for side, key in (("h", "home_lineup"), ("a", "away_lineup")):
                pids = gm[key]
                vals, wts, known = [], [], 0
                for i, pid in enumerate(pids[:9]):
                    r = lut.get((pid, day))
                    if r is None or not np.isfinite(getattr(r, "bx_xwoba", np.nan)):
                        continue
                    known += 1
                    vals.append(r)
                    wts.append(ORDER_WEIGHTS[i] if i < 9 else 1.0)
                if known < 4:
                    for c in LINEUP_SC_COLS:
                        rec[f"{side}_{c}"] = np.nan
                    rec[f"{side}_lsc_n_known"] = known
                    continue
                w = np.array(wts); w = w / w.sum()
                for m in BATTER_METRICS:
                    a = np.array([getattr(v, m, np.nan) for v in vals], float)
                    ok = np.isfinite(a)
                    # bx_xwoba -> lsc_xwoba (write the final name directly;
                    # renaming afterwards collided and produced duplicates)
                    rec[f"{side}_lsc_{m[3:]}"] = (
                        float(np.average(a[ok], weights=w[ok])) if ok.any() else np.nan)
                rec[f"{side}_lsc_n_known"] = known
            rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.loc[:, ~out.columns.duplicated()]
    for c in LINEUP_SC_COLS:
        if f"h_{c}" in out and f"a_{c}" in out:
            out[f"d_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]

    out.to_parquet(PROC / "lineup_statcast.parquet", index=False)
    cov = out["h_lsc_xwoba"].notna().mean() if "h_lsc_xwoba" in out else 0
    print(f"[lineup-sc] {len(out):,} games, {cov:.1%} with lineup Statcast")
    return out


def main() -> None:
    import sys
    if "--lineup" in sys.argv:
        build_lineup_statcast()
    else:
        aggregate()
        build_lineup_statcast()


if __name__ == "__main__":
    main()
