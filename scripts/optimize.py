"""Continuous improvement loop.

Searches model configs with walk-forward validation, selecting on LOG LOSS
(not accuracy). Reason: for betting, calibrated probability quality determines
profit; raw accuracy does not. A model that is 57% accurate but overconfident
loses to one that is 56% accurate and honest.

Every candidate is evaluated ONLY on future seasons relative to its training
data, so the search cannot overfit by peeking ahead.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"


def feature_cols(df):
    drop = {"home_win", "total_runs", "home_score", "away_score", "season", "month", "dow"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_", "sc_"))
            and c not in drop and df[c].dtype.kind in "fi"]


def walk_forward(df, fc, build_fn, seasons) -> dict:
    lls, aucs, accs, ns = [], [], [], []
    for s in seasons:
        tr, te = df[df.season < s], df[df.season == s]
        if len(tr) < 2000 or len(te) < 50:
            continue
        med = tr[fc].median()
        Xtr, Xte = tr[fc].fillna(med).values, te[fc].fillna(med).values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr, yte = tr.home_win.values, te.home_win.values
        m = build_fn()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        lls.append(log_loss(yte, p)); aucs.append(roc_auc_score(yte, p))
        accs.append(((p > .5).astype(int) == yte).mean()); ns.append(len(te))
    if not ns:
        return {"log_loss": 9.99, "auc": 0.5, "accuracy": 0.5}
    w = np.array(ns) / sum(ns)
    return {"log_loss": float(np.dot(lls, w)), "auc": float(np.dot(aucs, w)),
            "accuracy": float(np.dot(accs, w)), "n": int(sum(ns))}


def main() -> None:
    df = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    df = df.dropna(subset=["home_win"])
    fc = feature_cols(df)
    print(f"[optimize] {len(df):,} games, {len(fc)} features")
    seasons = [s for s in sorted(df.season.unique()) if s >= 2019]

    trials = []

    # --- Logistic regularization sweep
    for C in (0.005, 0.01, 0.03, 0.1, 0.3):
        r = walk_forward(df, fc, lambda C=C: LogisticRegression(max_iter=3000, C=C), seasons)
        trials.append({"name": f"logistic_C{C}", **r})
        print(f"  logistic C={C:<6} ll={r['log_loss']:.5f} auc={r['auc']:.4f} acc={r['accuracy']:.4f}")

    # --- GBM sweep
    grid = itertools.product((0.015, 0.025, 0.04), (3, 4), (60, 100, 160))
    for lr, depth, leaf in grid:
        def bf(lr=lr, depth=depth, leaf=leaf):
            return HistGradientBoostingClassifier(
                max_iter=300, learning_rate=lr, max_depth=depth,
                min_samples_leaf=leaf, l2_regularization=1.5,
                max_leaf_nodes=15, random_state=42)
        r = walk_forward(df, fc, bf, seasons)
        trials.append({"name": f"gbm_lr{lr}_d{depth}_leaf{leaf}", **r})
        print(f"  gbm lr={lr} d={depth} leaf={leaf:<4} ll={r['log_loss']:.5f} "
              f"auc={r['auc']:.4f} acc={r['accuracy']:.4f}")

    # --- Best GBM, isotonic-calibrated
    best_gbm = min([t for t in trials if t["name"].startswith("gbm")],
                   key=lambda t: t["log_loss"])
    parts = best_gbm["name"].split("_")
    lr = float(parts[1][2:]); depth = int(parts[2][1:]); leaf = int(parts[3][4:])

    def cal_fn():
        base = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=lr, max_depth=depth, min_samples_leaf=leaf,
            l2_regularization=1.5, max_leaf_nodes=15, random_state=42)
        return CalibratedClassifierCV(base, method="isotonic", cv=4)
    r = walk_forward(df, fc, cal_fn, seasons)
    trials.append({"name": f"{best_gbm['name']}_isotonic", **r})
    print(f"  calibrated       ll={r['log_loss']:.5f} auc={r['auc']:.4f} acc={r['accuracy']:.4f}")

    # --- Ensemble: logistic + gbm average
    best_log = min([t for t in trials if t["name"].startswith("logistic")],
                   key=lambda t: t["log_loss"])
    C = float(best_log["name"].split("C")[1])

    class Ens:
        def fit(self, X, y):
            self.a = LogisticRegression(max_iter=3000, C=C).fit(X, y)
            self.b = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=lr, max_depth=depth,
                min_samples_leaf=leaf, l2_regularization=1.5,
                max_leaf_nodes=15, random_state=42).fit(X, y)
            return self

        def predict_proba(self, X):
            return 0.5 * self.a.predict_proba(X) + 0.5 * self.b.predict_proba(X)

    r = walk_forward(df, fc, Ens, seasons)
    trials.append({"name": "ensemble_log_gbm", **r})
    print(f"  ensemble         ll={r['log_loss']:.5f} auc={r['auc']:.4f} acc={r['accuracy']:.4f}")

    trials.sort(key=lambda t: t["log_loss"])
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "optimize.json").write_text(json.dumps(trials, indent=2))

    print(f"\n[optimize] BEST: {trials[0]['name']}")
    print(f"           log_loss={trials[0]['log_loss']:.5f}  auc={trials[0]['auc']:.4f} "
          f"acc={trials[0]['accuracy']:.4f}")
    print(f"[optimize] -> reports/optimize.json")


if __name__ == "__main__":
    main()
