"""Walk-forward backtest: train on the past, predict the next season, never look ahead.

For each season S in 2018..2026:
    train on every game before season S
    predict season S
    never refit inside S (no lookahead)

Reports accuracy, AUC, log-loss and Brier score. Calibration matters more than
accuracy: a betting model needs probabilities that mean what they say.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"home_win", "total_runs", "home_score", "away_score", "season", "month", "dow"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_"))
            and c not in drop and df[c].dtype.kind in "fi"]


def make_model(kind: str):
    if kind == "logistic":
        return LogisticRegression(max_iter=3000, C=0.03)
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=340, learning_rate=0.024, max_depth=4,
            min_samples_leaf=95, l2_regularization=1.4,
            max_leaf_nodes=17, random_state=42,
        )
    if kind == "gbm_cal":
        base = HistGradientBoostingClassifier(
            max_iter=340, learning_rate=0.024, max_depth=4,
            min_samples_leaf=95, l2_regularization=1.4,
            max_leaf_nodes=17, random_state=42,
        )
        return CalibratedClassifierCV(base, method="isotonic", cv=4)
    raise ValueError(kind)


def home_field_baseline(train_y: np.ndarray, n: int) -> np.ndarray:
    """Naive benchmark: always predict the historical home win rate."""
    return np.full(n, train_y.mean())


def run(kind: str = "gbm_cal", start_season: int = 2018, verbose: bool = True) -> dict:
    df = pd.read_parquet(PROC / "features.parquet").sort_values("date").reset_index(drop=True)
    fc = feature_cols(df)
    df = df.dropna(subset=["home_win"])

    seasons = sorted(s for s in df["season"].unique() if s >= start_season)
    rows, oof = [], []

    for s in seasons:
        tr = df[df["season"] < s]
        te = df[df["season"] == s]
        if len(tr) < 2000 or len(te) < 50:
            continue

        med = tr[fc].median()
        Xtr = tr[fc].fillna(med).values
        Xte = te[fc].fillna(med).values
        ytr, yte = tr["home_win"].values, te["home_win"].values

        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

        model = make_model(kind)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]

        base = home_field_baseline(ytr, len(yte))
        rows.append({
            "season": int(s),
            "n_train": len(tr),
            "n_test": len(te),
            "accuracy": float(((p > 0.5).astype(int) == yte).mean()),
            "auc": float(roc_auc_score(yte, p)),
            "log_loss": float(log_loss(yte, p)),
            "brier": float(brier_score_loss(yte, p)),
            "base_acc": float(((base > 0.5).astype(int) == yte).mean()),
            "base_log_loss": float(log_loss(yte, base)),
            "base_brier": float(brier_score_loss(yte, base)),
        })
        d = te[["game_id", "date", "season", "home", "away", "home_win"]].copy()
        d["p_home"] = p
        oof.append(d)

        if verbose:
            r = rows[-1]
            print(f"  {s}  n={r['n_test']:>5}  acc={r['accuracy']:.4f} "
                  f"(base {r['base_acc']:.4f})  auc={r['auc']:.4f}  "
                  f"ll={r['log_loss']:.4f} (base {r['base_log_loss']:.4f})  "
                  f"brier={r['brier']:.4f}")

    res = pd.DataFrame(rows)
    allp = pd.concat(oof, ignore_index=True)
    REPORTS.mkdir(exist_ok=True)
    allp.to_parquet(PROC / f"oof_{kind}.parquet", index=False)

    w = res["n_test"] / res["n_test"].sum()
    summary = {
        "model": kind,
        "seasons": [int(x) for x in res["season"]],
        "total_games": int(res["n_test"].sum()),
        "accuracy": float((res["accuracy"] * w).sum()),
        "auc": float((res["auc"] * w).sum()),
        "log_loss": float((res["log_loss"] * w).sum()),
        "brier": float((res["brier"] * w).sum()),
        "base_accuracy": float((res["base_acc"] * w).sum()),
        "base_log_loss": float((res["base_log_loss"] * w).sum()),
        "base_brier": float((res["base_brier"] * w).sum()),
    }
    summary["ll_improvement_pct"] = float(
        100 * (summary["base_log_loss"] - summary["log_loss"]) / summary["base_log_loss"]
    )
    return {"per_season": res, "summary": summary, "oof": allp}


def main() -> None:
    out = {}
    for kind in ("logistic", "gbm", "gbm_cal"):
        print(f"\n=== {kind} ===")
        r = run(kind)
        s = r["summary"]
        print(f"  WEIGHTED: acc={s['accuracy']:.4f} (base {s['base_accuracy']:.4f})  "
              f"auc={s['auc']:.4f}  ll={s['log_loss']:.4f}  "
              f"improvement over base {s['ll_improvement_pct']:+.2f}%")
        out[kind] = s

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps(out, indent=2))
    print(f"\n[backtest] -> reports/backtest.json")


if __name__ == "__main__":
    main()
