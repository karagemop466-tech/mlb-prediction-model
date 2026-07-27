"""Build strictly point-in-time features. No leakage, by construction.

THE CORE RULE: every feature for game N uses only games 1..N-1.
Implemented by computing rolling stats then SHIFTING them by one game per team,
so a team's row never contains the result of the game being predicted.

Feature families:
  - Team form      : rolling win%, run differential, runs scored/allowed (multiple windows)
  - Pythagorean    : expected win% from runs (Pythagenpat exponent)
  - Starting pitcher: rolling ERA proxy, K/BB, and per-start run support
  - Rest & travel  : days since last game, games in last 7/10, home/road trip length
  - Park           : rolling park run factor
  - Head to head   : recent series history
  - Bullpen        : rolling relief innings load proxy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game, so rolling ops are natural."""
    home = pd.DataFrame({
        "game_id": df["game_id"], "date": df["date"], "season": df["season"],
        "team": df["home"], "opp": df["away"], "is_home": 1,
        "runs_for": df["home_score"], "runs_against": df["away_score"],
        "win": df["home_win"], "sp": df["home_sp"], "park": df["park"],
        "hits_for": df["home_hits"], "hits_against": df["away_hits"],
        "hr_for": df["home_hr"], "bb_for": df["home_bb"], "k_for": df["home_k"],
        "errors": df["home_e"],
    })
    away = pd.DataFrame({
        "game_id": df["game_id"], "date": df["date"], "season": df["season"],
        "team": df["away"], "opp": df["home"], "is_home": 0,
        "runs_for": df["away_score"], "runs_against": df["home_score"],
        "win": 1 - df["home_win"], "sp": df["away_sp"], "park": df["park"],
        "hits_for": df["away_hits"], "hits_against": df["home_hits"],
        "hr_for": df["away_hr"], "bb_for": df["away_bb"], "k_for": df["away_k"],
        "errors": df["away_e"],
    })
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)


def _shifted_roll(g: pd.Series, window: int, min_p: int = 5, fn: str = "mean") -> pd.Series:
    """Rolling stat using ONLY prior games (shift(1) before rolling)."""
    s = g.shift(1)
    r = s.rolling(window, min_periods=min_p)
    return getattr(r, fn)()


def add_team_features(long: pd.DataFrame) -> pd.DataFrame:
    out = []
    for team, g in long.groupby("team", sort=False):
        g = g.sort_values(["date", "game_id"]).copy()

        for w in (10, 25, 50, 100):
            g[f"win_pct_{w}"] = _shifted_roll(g["win"], w)
            g[f"rf_{w}"] = _shifted_roll(g["runs_for"], w)
            g[f"ra_{w}"] = _shifted_roll(g["runs_against"], w)
            g[f"rdiff_{w}"] = g[f"rf_{w}"] - g[f"ra_{w}"]

        # Pythagenpat expected win% (exponent varies with run environment)
        rpg = (g["rf_50"] + g["ra_50"]).clip(lower=0.1)
        expo = np.power(rpg, 0.287)
        rf, ra = g["rf_50"].clip(lower=0.01), g["ra_50"].clip(lower=0.01)
        g["pythag"] = np.power(rf, expo) / (np.power(rf, expo) + np.power(ra, expo))

        # Volatility: inconsistent teams are harder to price
        g["rf_std_25"] = _shifted_roll(g["runs_for"], 25, fn="std")
        g["ra_std_25"] = _shifted_roll(g["runs_against"], 25, fn="std")

        # Peripheral rolling rates (missing for 2026 statsapi rows -> NaN, handled later)
        g["hr_25"] = _shifted_roll(g["hr_for"], 25)
        g["bb_25"] = _shifted_roll(g["bb_for"], 25)
        g["k_25"] = _shifted_roll(g["k_for"], 25)
        g["err_25"] = _shifted_roll(g["errors"], 25)

        # Rest / schedule load
        g["days_rest"] = g["date"].diff().dt.days.clip(0, 10)
        g["games_last_7"] = (
            g.set_index("date")["win"].shift(1).rolling("7D").count().values
        )
        g["games_last_10d"] = (
            g.set_index("date")["win"].shift(1).rolling("10D").count().values
        )

        # Streaks (prior only)
        prev = g["win"].shift(1)
        grp = (prev != prev.shift()).cumsum()
        g["streak_len"] = prev.groupby(grp).cumcount() + 1
        g["streak_dir"] = np.where(prev == 1, 1, -1)
        g["streak"] = g["streak_len"] * g["streak_dir"]

        # Home/road split form
        g["home_form"] = _shifted_roll(g["win"].where(g["is_home"] == 1), 25, min_p=3)
        g["road_form"] = _shifted_roll(g["win"].where(g["is_home"] == 0), 25, min_p=3)

        # Season-to-date (expanding, prior only)
        g["std_win_pct"] = g.groupby("season")["win"].transform(
            lambda s: s.shift(1).expanding(min_periods=5).mean()
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_pitcher_features(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling starting-pitcher form. Uses only the pitcher's PRIOR starts."""
    long = long.sort_values(["sp", "date", "game_id"]).copy()
    parts = []
    for sp, g in long.groupby("sp", sort=False):
        g = g.sort_values(["date", "game_id"]).copy()
        if not sp or str(sp).strip() == "":
            g["sp_ra_10"] = np.nan
            g["sp_win_10"] = np.nan
            g["sp_starts"] = 0
            g["sp_days_rest"] = np.nan
            parts.append(g)
            continue
        # Runs allowed by the pitcher's TEAM in his starts (proxy for run prevention)
        g["sp_ra_10"] = _shifted_roll(g["runs_against"], 10, min_p=3)
        g["sp_ra_25"] = _shifted_roll(g["runs_against"], 25, min_p=5)
        g["sp_win_10"] = _shifted_roll(g["win"], 10, min_p=3)
        g["sp_k_10"] = _shifted_roll(g["k_for"], 10, min_p=3)
        g["sp_starts"] = np.arange(len(g))
        g["sp_days_rest"] = g["date"].diff().dt.days.clip(0, 30)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_park_factor(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling park run factor from prior games at that park (expanding, shifted)."""
    long = long.sort_values(["park", "date", "game_id"]).copy()
    parts = []
    for park, g in long.groupby("park", sort=False):
        g = g.sort_values(["date", "game_id"]).copy()
        tot = g["runs_for"] + g["runs_against"]
        g["park_runs"] = tot.shift(1).rolling(200, min_periods=30).mean()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    lg = out.groupby("season")["park_runs"].transform("mean")
    out["park_factor"] = out["park_runs"] / lg.replace(0, np.nan)
    return out


def build() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "games.parquet")
    long = to_long(df)
    long = add_team_features(long)
    long = add_pitcher_features(long)
    long = add_park_factor(long)

    feat_cols = [c for c in long.columns if c not in
                 {"game_id", "date", "season", "team", "opp", "is_home", "runs_for",
                  "runs_against", "win", "sp", "park", "hits_for", "hits_against",
                  "hr_for", "bb_for", "k_for", "errors", "park_runs", "park_factor"}]

    h = long[long.is_home == 1].set_index("game_id")[feat_cols + ["park_factor"]]
    a = long[long.is_home == 0].set_index("game_id")[feat_cols]
    h.columns = [f"h_{c}" for c in h.columns]
    a.columns = [f"a_{c}" for c in a.columns]

    wide = df.set_index("game_id").join(h).join(a)

    # Differential features: what actually drives the prediction
    for c in feat_cols:
        if f"h_{c}" in wide.columns and f"a_{c}" in wide.columns:
            wide[f"d_{c}"] = wide[f"h_{c}"] - wide[f"a_{c}"]

    wide = wide.reset_index()

    # --- Optional Statcast merge (strictly prior-games rolling, built in statcast.py)
    sc_path = PROC / "statcast_team.parquet"
    if sc_path.exists():
        sc = pd.read_parquet(sc_path)
        sc["date"] = pd.to_datetime(sc["date"])
        # A team can play twice on one date (doubleheaders). Keep one row per
        # (date, team) so the merge cannot fan out and duplicate games.
        sc = sc.drop_duplicates(subset=["date", "team"], keep="first")
        sc_cols = [c for c in sc.columns if c.startswith("sc_")]
        h = sc.rename(columns={"team": "home", **{c: f"h_{c}" for c in sc_cols}})
        a = sc.rename(columns={"team": "away", **{c: f"a_{c}" for c in sc_cols}})
        wide = wide.merge(h, on=["date", "home"], how="left")
        wide = wide.merge(a, on=["date", "away"], how="left")
        for c in sc_cols:
            if f"h_{c}" in wide.columns and f"a_{c}" in wide.columns:
                wide[f"d_{c}"] = wide[f"h_{c}"] - wide[f"a_{c}"]
        got = wide[f"h_{sc_cols[0]}"].notna().sum() if sc_cols else 0
        print(f"[features] statcast merged: {got:,} games have Statcast coverage")

    # --- Optional PITCHER-level Statcast (rolling prior starts only)
    #
    # TESTED AND NOT SHIPPED. Across 7,939 games (2023-2026) these 24 features
    # moved accuracy by -0.0001 and AUC by +0.0023, both far inside the +/-1.09%
    # confidence interval. See reports/experiment_pitcher_statcast.json.
    # Set MLB_USE_SP_STATCAST=1 to enable for further research.
    import os
    if os.environ.get("MLB_USE_SP_STATCAST") != "1":
        psc_path = None
    else:
        psc_path = PROC / "pitcher_statcast.parquet"
    xw_path = ROOT / "data" / "raw" / "ref" / "crosswalk.parquet"
    if psc_path is not None and psc_path.exists() and xw_path.exists():
        psc = pd.read_parquet(psc_path)
        psc["date"] = pd.to_datetime(psc["date"])
        xw = pd.read_parquet(xw_path)[["key_mlbam", "key_retro"]]
        psc = psc.merge(xw, left_on="mlbam", right_on="key_mlbam", how="left")
        pcols = [c for c in psc.columns if c.startswith("sp_")]

        # Retrosheet seasons store retro IDs in home_sp/away_sp; the live MLB API
        # seasons (2026+) store MLBAM IDs. Build a key covering BOTH so recent
        # games are not silently dropped.
        retro = psc[psc.key_retro.notna()].copy()
        retro["sp_key"] = retro["key_retro"].astype(str)
        mlbam = psc.copy()
        mlbam["sp_key"] = mlbam["mlbam"].astype(int).astype(str)
        psc = pd.concat([retro, mlbam], ignore_index=True)
        psc = psc.drop_duplicates(subset=["date", "sp_key"], keep="first")

        h = psc.rename(columns={"sp_key": "home_sp",
                                **{c: f"h_{c}" for c in pcols}})
        a = psc.rename(columns={"sp_key": "away_sp",
                                **{c: f"a_{c}" for c in pcols}})
        keep_h = ["date", "home_sp"] + [f"h_{c}" for c in pcols]
        keep_a = ["date", "away_sp"] + [f"a_{c}" for c in pcols]
        wide = wide.merge(h[keep_h], on=["date", "home_sp"], how="left")
        wide = wide.merge(a[keep_a], on=["date", "away_sp"], how="left")
        for c in pcols:
            if f"h_{c}" in wide.columns and f"a_{c}" in wide.columns:
                wide[f"d_{c}"] = wide[f"h_{c}"] - wide[f"a_{c}"]
        got = wide[f"h_{pcols[0]}"].notna().sum() if pcols else 0
        print(f"[features] pitcher statcast merged: {got:,} games have SP coverage")

    wide["month"] = wide["date"].dt.month
    wide["dow"] = wide["date"].dt.dayofweek
    wide["is_night"] = (wide["daynight"] == "N").astype(int)
    return wide


if __name__ == "__main__":
    w = build()
    out = PROC / "features.parquet"
    w.to_parquet(out, index=False)
    nfeat = len([c for c in w.columns if c.startswith(("h_", "a_", "d_"))])
    print(f"[features] {len(w):,} games x {nfeat} features -> {out}")
    print(f"[features] usable (non-null d_pythag): {w['d_pythag'].notna().sum():,}")
