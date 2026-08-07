"""Bullpen availability from actual reliever workload.

WHY RETRY THIS
--------------
`research_loop.py` tested a bullpen hypothesis earlier and rejected it
(-0.0023). But that version used a crude proxy: total runs allowed in the prior
3 days, which conflates starter and reliever performance and says nothing about
which arms are actually available.

The pitch-level data cached for the arsenal experiment allows a direct
measurement: for each team and date, how many pitches did its RELIEVERS throw in
the previous 1/2/3 days, and how many distinct relievers are likely unavailable.

This is structural in the same sense weather is -- it is a constraint on the
game environment, not a statement about team quality -- so it is worth one
properly-measured attempt.

DEFINITION
----------
A reliever is any pitcher in a game who was not one of the two highest-pitch-count
arms for his side (the same starter rule used in pitcher_statcast.py). Workload
is measured in pitches, which is the quantity managers actually track.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# pitcher_sc chunks retain inning_topbot/home_team/away_team, which the arsenal
# chunks do not (they were trimmed to pitch_type fields). Reading the wrong
# cache silently yields zero rows, so the source matters.
RAW = ROOT / "data" / "raw" / "pitcher_sc"
PROC = ROOT / "data" / "proc"

USE_COLS = ["game_date", "game_pk", "pitcher", "inning_topbot",
            "home_team", "away_team"]

BULLPEN_COLS = ["bp_pitches_1d", "bp_pitches_2d", "bp_pitches_3d",
                "bp_arms_used_3d", "bp_heavy_arms_2d"]


def build_workload() -> pd.DataFrame:
    """Pitches thrown by each team's relievers, per team-date."""
    files = sorted(RAW.glob("p_*.parquet"))
    if not files:
        print("  [bullpen] no cached pitch chunks")
        return pd.DataFrame()

    # Fail loudly on a schema mismatch instead of returning an empty frame.
    probe = pd.read_parquet(files[0])
    missing = [c for c in USE_COLS if c not in probe.columns]
    if missing:
        raise SystemExit(f"[bullpen] {files[0].name} lacks {missing}; "
                         "wrong cache directory?")

    parts = []
    for i, f in enumerate(files, 1):
        try:
            c = pd.read_parquet(f, columns=[x for x in USE_COLS])
        except Exception:
            continue
        c = c[c["pitcher"].notna()]
        if c.empty:
            continue
        c = c.copy()
        c["game_date"] = pd.to_datetime(c["game_date"])
        c["pitcher"] = c["pitcher"].astype(int)
        # Pitching team is home when the away side bats (top of inning).
        c["pit_team"] = np.where(c["inning_topbot"] == "Top",
                                 c["home_team"], c["away_team"])
        agg = (c.groupby(["game_date", "game_pk", "pit_team", "pitcher"])
                 .size().reset_index(name="pitches"))
        parts.append(agg)
        del c
        if i % 60 == 0:
            print(f"    {i}/{len(files)}")
    if not parts:
        return pd.DataFrame()

    p = pd.concat(parts, ignore_index=True)
    p = p.groupby(["game_date", "game_pk", "pit_team", "pitcher"],
                  as_index=False)["pitches"].sum()

    # Starter = highest pitch count for his team in that game.
    p = p.sort_values(["game_pk", "pit_team", "pitches"],
                      ascending=[True, True, False])
    p["rank"] = p.groupby(["game_pk", "pit_team"]).cumcount()
    rel = p[(p["rank"] > 0) & (p["pitches"] >= 3)]

    daily = (rel.groupby(["game_date", "pit_team"])
                .agg(rel_pitches=("pitches", "sum"),
                     rel_arms=("pitcher", "nunique"),
                     heavy_arms=("pitches", lambda s: int((s >= 25).sum())))
                .reset_index()
                .rename(columns={"game_date": "date", "pit_team": "team"}))

    PROC.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(PROC / "bullpen_daily.parquet", index=False)
    print(f"  [bullpen] {len(daily):,} team-days of reliever workload")
    return daily


def build_features() -> pd.DataFrame:
    """Rolling prior-days workload, joined to games."""
    dpath = PROC / "bullpen_daily.parquet"
    if not dpath.exists():
        print("  [bullpen] run build_workload() first")
        return pd.DataFrame()
    daily = pd.read_parquet(dpath)
    daily["date"] = pd.to_datetime(daily["date"])

    from build_dataset import TEAM_MAP
    daily["team"] = daily["team"].map(lambda x: TEAM_MAP.get(x, x))
    daily = daily.groupby(["date", "team"], as_index=False).sum(numeric_only=True)

    out = []
    for team, g in daily.groupby("team", sort=False):
        g = g.sort_values("date").set_index("date")
        # Prior-days only: shift before rolling on a time window.
        r = pd.DataFrame(index=g.index)
        r["bp_pitches_1d"] = g["rel_pitches"].shift(1).rolling("1D").sum()
        r["bp_pitches_2d"] = g["rel_pitches"].shift(1).rolling("2D").sum()
        r["bp_pitches_3d"] = g["rel_pitches"].shift(1).rolling("3D").sum()
        r["bp_arms_used_3d"] = g["rel_arms"].shift(1).rolling("3D").sum()
        r["bp_heavy_arms_2d"] = g["heavy_arms"].shift(1).rolling("2D").sum()
        r["team"] = team
        out.append(r.reset_index())
    bp = pd.concat(out, ignore_index=True)

    games = pd.read_parquet(PROC / "games.parquet")
    games["date"] = pd.to_datetime(games["date"])
    h = bp.rename(columns={"team": "home",
                           **{c: f"h_{c}" for c in BULLPEN_COLS}})
    a = bp.rename(columns={"team": "away",
                           **{c: f"a_{c}" for c in BULLPEN_COLS}})
    m = games[["game_id", "date", "home", "away"]].merge(
        h[["date", "home"] + [f"h_{c}" for c in BULLPEN_COLS]],
        on=["date", "home"], how="left")
    m = m.merge(a[["date", "away"] + [f"a_{c}" for c in BULLPEN_COLS]],
                on=["date", "away"], how="left")
    for c in BULLPEN_COLS:
        m[f"d_{c}"] = m[f"h_{c}"] - m[f"a_{c}"]

    m.to_parquet(PROC / "bullpen_features.parquet", index=False)
    cov = m["h_bp_pitches_3d"].notna().mean()
    print(f"  [bullpen] {len(m):,} games, {cov:.1%} with workload history")
    return m


def main() -> None:
    import sys
    if "--features" in sys.argv:
        build_features()
    else:
        build_workload()
        build_features()


if __name__ == "__main__":
    main()
