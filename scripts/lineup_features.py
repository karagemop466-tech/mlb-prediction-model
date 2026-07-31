"""Turn starting lineups into team-level batting quality features.

DESIGN: GRACEFUL DEGRADATION
----------------------------
Lineups post ~3-4 hours before first pitch, but the daily workflow runs at 11:00
UTC, well before that. So these features are NULL for most live predictions.
Every consumer must therefore treat them as optional. The models already impute
missing values with the training median, so a game without a posted lineup falls
back to team-level form exactly as before.

This is why the features are built as DEVIATIONS from each team's rolling
average rather than as absolute levels. An absolute OPS of .740 means nothing
without knowing the team's baseline; "this lineup is .028 OPS below what this
team usually fields" is directly interpretable and is what actually carries
information -- it detects rest days, injuries and September callups.

Features per side:
    lu_ops / lu_obp / lu_slg / lu_iso     PA-weighted lineup quality
    lu_top5_ops                            top of the order carries more PAs
    lu_k_rate / lu_bb_rate                 plate discipline
    lu_pa_experience                       mean prior PAs (callup detector)
    lu_n_known                             how many of the 9 had prior stats
    lu_ops_vs_team                         DEVIATION from team's season norm
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
RAW = ROOT / "data" / "raw" / "lineups"

# Batting-order weights. Leadoff hitters get ~4.6 PA/game, the 9-hole ~3.9.
# Source: relative PA share, normalised to sum to 9.
ORDER_WEIGHTS = np.array([1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91, 0.88])

LINEUP_COLS = [
    "lu_ops", "lu_obp", "lu_slg", "lu_iso", "lu_top5_ops",
    "lu_k_rate", "lu_bb_rate", "lu_pa_experience", "lu_n_known",
]


def load_lineups() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*.json")):
        try:
            rows.extend(json.loads(f.read_text()))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def lineup_quality(pids: list[int], day: pd.Timestamp,
                   hist: dict[tuple[int, pd.Timestamp], dict]) -> dict:
    """PA-weighted quality of one posted lineup, using prior-to-date stats."""
    out = {c: np.nan for c in LINEUP_COLS}
    if not pids:
        return out

    vals, weights, known = [], [], 0
    top5 = []
    for i, pid in enumerate(pids[:9]):
        rec = hist.get((pid, day))
        w = ORDER_WEIGHTS[i] if i < 9 else 1.0
        if rec is None or not np.isfinite(rec.get("p_ops", np.nan)):
            continue
        known += 1
        vals.append(rec)
        weights.append(w)
        if i < 5:
            top5.append(rec["p_ops"])

    out["lu_n_known"] = known
    if known < 4:          # too thin to be meaningful
        return out

    w = np.array(weights)
    w = w / w.sum()

    def wavg(key):
        a = np.array([v.get(key, np.nan) for v in vals], dtype=float)
        m = np.isfinite(a)
        return float(np.average(a[m], weights=w[m])) if m.any() else np.nan

    out["lu_ops"] = wavg("p_ops")
    out["lu_obp"] = wavg("p_obp")
    out["lu_slg"] = wavg("p_slg")
    out["lu_iso"] = wavg("p_iso")
    out["lu_k_rate"] = wavg("p_k_rate")
    out["lu_bb_rate"] = wavg("p_bb_rate")
    out["lu_pa_experience"] = wavg("p_pa")
    out["lu_top5_ops"] = float(np.nanmean(top5)) if top5 else np.nan
    return out


def build() -> pd.DataFrame:
    lu = load_lineups()
    if lu.empty:
        print("[lineup-features] no lineup data")
        return pd.DataFrame()

    hpath = PROC / "player_batting_history.parquet"
    if not hpath.exists():
        print("[lineup-features] no player history; run lineups.py --stats")
        return pd.DataFrame()

    hist_df = pd.read_parquet(hpath)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist: dict[tuple[int, pd.Timestamp], dict] = {}
    for r in hist_df.itertuples(index=False):
        hist[(r.player_id, r.date)] = {
            "p_ops": r.p_ops, "p_obp": r.p_obp, "p_slg": r.p_slg,
            "p_iso": r.p_iso, "p_k_rate": r.p_k_rate,
            "p_bb_rate": r.p_bb_rate, "p_pa": r.p_pa,
        }
    print(f"[lineup-features] player-date records: {len(hist):,}")

    rows = []
    for r in lu.itertuples(index=False):
        h = lineup_quality(list(r.home_lineup), r.date, hist)
        a = lineup_quality(list(r.away_lineup), r.date, hist)
        rec = {"game_pk": r.game_pk, "date": r.date,
               "home_abbr": r.home, "away_abbr": r.away}
        rec.update({f"h_{k}": v for k, v in h.items()})
        rec.update({f"a_{k}": v for k, v in a.items()})
        rows.append(rec)

    out = pd.DataFrame(rows)
    for c in LINEUP_COLS:
        out[f"d_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "lineup_features.parquet", index=False)
    cov = out["h_lu_ops"].notna().mean()
    print(f"[lineup-features] {len(out):,} games, "
          f"{cov:.1%} with usable home lineup quality")
    return out


def attach(df: pd.DataFrame) -> pd.DataFrame:
    """Join lineup features to a games/features frame, adding team deviations.

    The deviation columns are the point: they express how the posted lineup
    compares with what that team normally fields, which is what detects rest
    days, injuries and callups.
    """
    path = PROC / "lineup_features.parquet"
    if not path.exists():
        return df
    lf = pd.read_parquet(path)

    from build_dataset import TEAM_MAP
    lf["home_r"] = lf["home_abbr"].map(lambda x: TEAM_MAP.get(x, x))
    lf["away_r"] = lf["away_abbr"].map(lambda x: TEAM_MAP.get(x, x))
    lf = lf.drop_duplicates(["date", "home_r", "away_r"])

    cols = ([f"h_{c}" for c in LINEUP_COLS] + [f"a_{c}" for c in LINEUP_COLS]
            + [f"d_{c}" for c in LINEUP_COLS])
    merged = df.merge(
        lf[["date", "home_r", "away_r"] + cols],
        left_on=["date", "home", "away"], right_on=["date", "home_r", "away_r"],
        how="left").drop(columns=["home_r", "away_r"], errors="ignore")

    # Deviation from each team's own rolling lineup quality (prior games only).
    for side, teamcol in (("h", "home"), ("a", "away")):
        col = f"{side}_lu_ops"
        if col not in merged:
            continue
        merged = merged.sort_values("date")
        base = (merged.groupby(teamcol)[col]
                .transform(lambda s: s.shift(1).rolling(30, min_periods=8).mean()))
        merged[f"{side}_lu_ops_vs_team"] = merged[col] - base
    if "h_lu_ops_vs_team" in merged and "a_lu_ops_vs_team" in merged:
        merged["d_lu_ops_vs_team"] = (merged["h_lu_ops_vs_team"]
                                      - merged["a_lu_ops_vs_team"])
    return merged


def main() -> None:
    out = build()
    if out.empty:
        return
    print("\nLineup quality distributions (home side):")
    for c in ("h_lu_ops", "h_lu_top5_ops", "h_lu_k_rate", "h_lu_pa_experience",
              "h_lu_n_known"):
        s = out[c].dropna()
        if len(s):
            print(f"  {c:<22} mean {s.mean():>8.4f}  sd {s.std():>7.4f}  "
                  f"[{s.min():.3f}, {s.max():.3f}]")


if __name__ == "__main__":
    main()
