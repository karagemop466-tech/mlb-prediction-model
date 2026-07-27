"""Accuracy-focused model search, walk-forward validated.

Differences from optimize.py (which selected on log loss for betting value):
  1. Selection metric is ACCURACY.
  2. The decision threshold is tuned on TRAINING data only, then applied to the
     test season. Accuracy depends on where you cut; 0.5 is not always optimal
     when a class is imbalanced (home teams win ~53%).
  3. Every result carries a standard error, because accuracy differences below
     ~0.7% on this sample size are statistical noise, not real improvements.

Selecting the max of many noisy candidates is itself a form of overfitting, so
the winner is re-validated on held-out seasons the search never touched.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"


def feature_cols(df):
    drop = {"home_win", "total_runs", "home_score", "away_score", "season", "month", "dow"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_", "sc_"))
            and c not in drop and df[c].dtype.kind in "fi"]


def best_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing accuracy on TRAINING predictions only."""
    grid = np.arange(0.40, 0.61, 0.005)
    accs = [accuracy_score(y_true, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(accs))])


def walk_forward(df, fc, build_fn, seasons, tune_threshold=True) -> dict:
    accs, lls, aucs, ns, thrs = [], [], [], [], []
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

        thr = 0.5
        if tune_threshold:
            # in-sample training predictions -> threshold only, never test data
            thr = best_threshold(ytr, m.predict_proba(Xtr)[:, 1])
        p = m.predict_proba(Xte)[:, 1]

        accs.append(accuracy_score(yte, (p >= thr).astype(int)))
        lls.append(log_loss(yte, p))
        aucs.append(roc_auc_score(yte, p))
        ns.append(len(te))
        thrs.append(thr)

    if not ns:
        return {"accuracy": 0.5, "log_loss": 9.9, "auc": 0.5, "n": 0, "se": 0}
    w = np.array(ns) / sum(ns)
    acc = float(np.dot(accs, w))
    n = int(sum(ns))
    return {
        "accuracy": acc,
        "log_loss": float(np.dot(lls, w)),
        "auc": float(np.dot(aucs, w)),
        "n": n,
        "se": float(np.sqrt(acc * (1 - acc) / n)),
        "mean_threshold": float(np.mean(thrs)),
        "per_season_acc": [float(a) for a in accs],
    }


def main() -> None:
    df = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    df = df.dropna(subset=["home_win"])
    fc = feature_cols(df)
    print(f"[opt-acc] {len(df):,} games, {len(fc)} features")

    # Search seasons vs. final holdout seasons the search never sees.
    search_seasons = [s for s in sorted(df.season.unique()) if 2019 <= s <= 2023]
    holdout_seasons = [s for s in sorted(df.season.unique()) if s >= 2024]
    print(f"[opt-acc] search on {search_seasons}, holdout {holdout_seasons}\n")

    trials = []

    def add(name, build_fn, tune=True):
        r = walk_forward(df, fc, build_fn, search_seasons, tune_threshold=tune)
        trials.append({"name": name, "tuned_threshold": tune, **r})
        print(f"  {name:<34} acc={r['accuracy']:.4f} (±{1.96*r['se']:.4f}) "
              f"thr={r['mean_threshold']:.3f} auc={r['auc']:.4f}")

    for C in (0.003, 0.005, 0.01, 0.03, 0.1):
        add(f"logistic_C{C}", lambda C=C: LogisticRegression(max_iter=3000, C=C))

    for lr, depth, leaf in itertools.product((0.015, 0.025), (3, 4), (100, 160, 240)):
        add(f"gbm_lr{lr}_d{depth}_l{leaf}",
            lambda lr=lr, d=depth, l=leaf: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=lr, max_depth=d, min_samples_leaf=l,
                l2_regularization=1.5, max_leaf_nodes=15, random_state=42))

    add("rf_600_d8", lambda: RandomForestClassifier(
        n_estimators=600, max_depth=8, min_samples_leaf=40,
        n_jobs=-1, random_state=42))

    class Ens:
        def __init__(self, C, lr, d, l):
            self.C, self.lr, self.d, self.l = C, lr, d, l

        def fit(self, X, y):
            self.a = LogisticRegression(max_iter=3000, C=self.C).fit(X, y)
            self.b = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=self.lr, max_depth=self.d,
                min_samples_leaf=self.l, l2_regularization=1.5,
                max_leaf_nodes=15, random_state=42).fit(X, y)
            return self

        def predict_proba(self, X):
            return 0.5 * self.a.predict_proba(X) + 0.5 * self.b.predict_proba(X)

    add("ensemble_log_gbm", lambda: Ens(0.005, 0.015, 3, 160))
    add("ensemble_log_gbm_nothresh", lambda: Ens(0.005, 0.015, 3, 160), tune=False)

    trials.sort(key=lambda t: -t["accuracy"])
    top = trials[0]

    print(f"\n[opt-acc] search winner: {top['name']}  acc={top['accuracy']:.4f}")
    print(f"[opt-acc] within 1 SE of winner (statistically tied):")
    for t in trials:
        if top["accuracy"] - t["accuracy"] <= top["se"]:
            print(f"     {t['name']:<34} {t['accuracy']:.4f}")

    # Re-validate the winner on seasons the search never touched.
    print(f"\n[opt-acc] HOLDOUT re-validation on {holdout_seasons}:")
    name_to_fn = {
        "ensemble_log_gbm": lambda: Ens(0.005, 0.015, 3, 160),
        "ensemble_log_gbm_nothresh": lambda: Ens(0.005, 0.015, 3, 160),
    }
    fn = name_to_fn.get(top["name"])
    if fn is None:
        if top["name"].startswith("logistic"):
            C = float(top["name"].split("C")[1])
            fn = lambda: LogisticRegression(max_iter=3000, C=C)
        elif top["name"].startswith("gbm"):
            parts = top["name"].split("_")
            lr = float(parts[1][2:]); d = int(parts[2][1:]); l = int(parts[3][1:])
            fn = lambda: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=lr, max_depth=d, min_samples_leaf=l,
                l2_regularization=1.5, max_leaf_nodes=15, random_state=42)
        else:
            fn = lambda: RandomForestClassifier(
                n_estimators=600, max_depth=8, min_samples_leaf=40,
                n_jobs=-1, random_state=42)

    hold = walk_forward(df, fc, fn, holdout_seasons,
                        tune_threshold=top["tuned_threshold"])
    print(f"     accuracy={hold['accuracy']:.4f} (±{1.96*hold['se']:.4f}) "
          f"auc={hold['auc']:.4f}  n={hold['n']:,}")
    drop = top["accuracy"] - hold["accuracy"]
    print(f"     search-to-holdout drop: {drop:+.4f} "
          f"({'expected noise' if abs(drop) < 2*hold['se'] else 'LARGER THAN NOISE'})")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "optimize_accuracy.json").write_text(json.dumps(
        {"search": trials, "winner": top["name"], "holdout": hold,
         "holdout_seasons": [int(s) for s in holdout_seasons]}, indent=2))
    print(f"\n[opt-acc] -> reports/optimize_accuracy.json")


if __name__ == "__main__":
    main()
