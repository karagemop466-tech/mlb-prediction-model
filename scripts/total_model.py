"""Learned expected-total model, replacing the hand-coded heuristic.

WHY THIS EXISTS
---------------
The first market backtest gave totals a skill score of only +0.0015. The cause
was not the simulator: it was the input. `expected_total_for()` in
backtest_markets.py averaged a few rolling run columns and shrank hard toward
the league mean, producing expected totals with sd 0.373 and correlation 0.116
with the realized total. Feeding a near-constant into the simulator guarantees a
near-constant output, so no totals market could show skill.

This module trains a gradient-boosted regressor on the same point-in-time
features the classifier uses, walk-forward, and is validated against the
heuristic it replaces.

Note the ceiling: single-game run totals are extremely noisy (sd 4.54 against a
mean of 9.05). Even a perfect model of the *expected* total explains a small
share of realized variance. The goal is a better conditional mean, not a
low-error prediction of any single game.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"home_win", "total_runs", "home_score", "away_score", "season",
            "month", "dow"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_"))
            and c not in drop and df[c].dtype.kind in "fi"]


class TotalModel:
    """Blend of a GBM and a ridge, shrunk toward the training-set mean.

    Shrinkage matters. An unshrunk regressor on this target overfits the noise
    and produces expected totals that swing far more than the true conditional
    mean, which then miscalibrates every over/under line.
    """

    def __init__(self, shrink: float = 0.75):
        self.shrink = shrink

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_ = float(y.mean())
        self.gbm_ = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.03, max_depth=3,
            min_samples_leaf=200, l2_regularization=2.0,
            max_leaf_nodes=12, random_state=42).fit(X, y)
        self.ridge_ = Ridge(alpha=50.0).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = 0.5 * self.gbm_.predict(X) + 0.5 * self.ridge_.predict(X)
        # Shrink toward the league mean, then clip to a plausible band.
        p = self.mean_ + self.shrink * (p - self.mean_)
        return np.clip(p, 6.5, 12.5)


def heuristic_total(row, league_mean: float) -> float:
    """The original hand-coded version, kept for comparison."""
    parts = []
    for side in ("h", "a"):
        rf, ra = row.get(f"{side}_rf_50"), row.get(f"{side}_ra_50")
        if pd.notna(rf):
            parts.append(float(rf))
        if pd.notna(ra):
            parts.append(float(ra))
    if not parts:
        return league_mean
    est = float(np.mean(parts)) * 2.0
    return float(np.clip(0.55 * est + 0.45 * league_mean, 6.5, 12.0))


def walk_forward(seasons=(2021, 2022, 2023, 2024, 2025, 2026),
                 shrink: float = 0.75) -> dict:
    df = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    df["total"] = df.home_score + df.away_score
    fc = feature_cols(df)

    rows = []
    preds_all, act_all, heur_all = [], [], []

    for s in seasons:
        tr, te = df[df.season < s], df[df.season == s]
        if len(tr) < 2000 or len(te) < 100:
            continue
        med = tr[fc].median()
        Xtr, Xte = tr[fc].fillna(med).values, te[fc].fillna(med).values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = TotalModel(shrink).fit((Xtr - mu) / sd, tr["total"].values)
        p = m.predict((Xte - mu) / sd)

        lg = float(tr["total"].mean())
        h = np.array([heuristic_total(r, lg) for _, r in te.iterrows()])
        a = te["total"].values

        rows.append({
            "season": int(s), "n": len(te),
            "model_corr": float(np.corrcoef(p, a)[0, 1]),
            "heur_corr": float(np.corrcoef(h, a)[0, 1]),
            "model_sd": float(p.std()), "heur_sd": float(h.std()),
            "model_mae": float(np.abs(p - a).mean()),
            "heur_mae": float(np.abs(h - a).mean()),
            "model_bias": float(p.mean() - a.mean()),
        })
        preds_all.append(p); act_all.append(a); heur_all.append(h)

    P = np.concatenate(preds_all); A = np.concatenate(act_all)
    H = np.concatenate(heur_all)
    summary = {
        "n": int(len(A)), "shrink": shrink,
        "model_corr": float(np.corrcoef(P, A)[0, 1]),
        "heur_corr": float(np.corrcoef(H, A)[0, 1]),
        "model_sd": float(P.std()), "heur_sd": float(H.std()),
        "actual_sd": float(A.std()),
        "model_mae": float(np.abs(P - A).mean()),
        "heur_mae": float(np.abs(H - A).mean()),
        "model_bias": float(P.mean() - A.mean()),
    }
    return {"per_season": rows, "summary": summary}


class SideModel:
    """Predicts home runs and away runs SEPARATELY.

    The simulator previously derived both scoring rates from P(home win) plus a
    single expected total, which forces the split to follow the win probability.
    But home scoring depends on home offense AND away pitching, and those are
    measured separately. Predicting each side directly lets the simulator handle
    a matchup like "strong offense vs strong pitching" (high win prob, low total)
    that a single-rate inversion cannot represent.
    """

    def __init__(self, shrink: float = 0.85):
        self.shrink = shrink

    def fit(self, X: np.ndarray, y_home: np.ndarray, y_away: np.ndarray):
        self.mh_ = float(y_home.mean())
        self.ma_ = float(y_away.mean())
        mk = lambda: HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.03, max_depth=3,
            min_samples_leaf=200, l2_regularization=2.0,
            max_leaf_nodes=12, random_state=42)
        self.gh_, self.ga_ = mk().fit(X, y_home), mk().fit(X, y_away)
        self.rh_ = Ridge(alpha=50.0).fit(X, y_home)
        self.ra_ = Ridge(alpha=50.0).fit(X, y_away)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = 0.5 * self.gh_.predict(X) + 0.5 * self.rh_.predict(X)
        a = 0.5 * self.ga_.predict(X) + 0.5 * self.ra_.predict(X)
        h = self.mh_ + self.shrink * (h - self.mh_)
        a = self.ma_ + self.shrink * (a - self.ma_)
        return np.clip(h, 2.5, 8.0), np.clip(a, 2.5, 8.0)


def fit_side_production(shrink: float = 0.85):
    df = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    df = df.dropna(subset=["home_win"])
    fc = feature_cols(df)
    med = df[fc].median()
    X = df[fc].fillna(med).values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = SideModel(shrink).fit((X - mu) / sd,
                              df.home_score.values, df.away_score.values)
    return m, fc, med, mu, sd


def fit_production(shrink: float = 0.75):
    """Fit on all available data, for live prediction."""
    df = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    df = df.dropna(subset=["home_win"])
    df["total"] = df.home_score + df.away_score
    fc = feature_cols(df)
    med = df[fc].median()
    X = df[fc].fillna(med).values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    model = TotalModel(shrink).fit((X - mu) / sd, df["total"].values)
    return model, fc, med, mu, sd


def main() -> None:
    print("Learned total model vs the hand-coded heuristic\n")
    best = None
    for shrink in (0.55, 0.75, 0.95):
        out = walk_forward(shrink=shrink)
        s = out["summary"]
        print(f"shrink={shrink}:  corr {s['model_corr']:.4f} (heuristic "
              f"{s['heur_corr']:.4f})   sd {s['model_sd']:.3f} "
              f"(heuristic {s['heur_sd']:.3f})   MAE {s['model_mae']:.4f} "
              f"(heuristic {s['heur_mae']:.4f})")
        if best is None or out["summary"]["model_corr"] > best["summary"]["model_corr"]:
            best = out

    s = best["summary"]
    print(f"\nBest shrink={s['shrink']}, n={s['n']:,}")
    print(f"  correlation with actual total: {s['model_corr']:.4f} "
          f"vs heuristic {s['heur_corr']:.4f}  "
          f"({s['model_corr']/max(s['heur_corr'],1e-9):.2f}x)")
    print(f"  spread across games (sd):      {s['model_sd']:.3f} "
          f"vs heuristic {s['heur_sd']:.3f}")
    print(f"  bias:                          {s['model_bias']:+.4f}")
    print(f"  actual total sd is {s['actual_sd']:.3f} — single-game totals are")
    print("  irreducibly noisy; this improves the conditional mean, not per-game error.")

    print("\nPer season:")
    for r in best["per_season"]:
        print(f"  {r['season']}  n={r['n']:>5}  corr {r['model_corr']:+.4f} "
              f"(heur {r['heur_corr']:+.4f})  sd {r['model_sd']:.3f}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "total_model.json").write_text(json.dumps(best, indent=2))
    print("\n-> reports/total_model.json")


if __name__ == "__main__":
    main()
