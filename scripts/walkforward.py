"""Day-by-day walk-forward simulation: strategy formation, then live replay.

WHAT THIS FIXES
---------------
The existing backtest retrains at SEASON boundaries. That means a game on
2024-09-30 is predicted by a model that has never seen a single 2024 game --
2,429 games of in-season information discarded. It answers "how good was last
winter's model all year?" rather than "how good is the system if operated
properly?"

This module replays history the way the system would actually be run:

    PHASE 1 (strategy formation)  seasons <= FORM_END
        Choose the model configuration and retrain cadence using ONLY these
        seasons. Nothing after FORM_END influences these choices.

    PHASE 2 (live replay)         seasons > FORM_END
        Walk forward one DAY at a time. On each day:
          - predict every game using a model trained only on prior days
          - after the day completes, those results join the training pool
          - retrain on the configured cadence
        The strategy is frozen. This is a forward test conducted inside
        history, which is the only way to get a large forward sample without
        waiting years.

WHY THE SPLIT MATTERS
---------------------
Selecting a configuration and evaluating it on the same data inflates results.
Phase 1 spends 2016-2022 on the selection; phase 2 never informs it. If phase 2
accuracy holds near phase 1 accuracy, the strategy generalises. If it collapses,
the selection was overfitting -- and we learn that instead of shipping it.

COST CONTROL
------------
A full ensemble fit takes ~19s, so daily retraining across 4 seasons would run
~8 hours. The random forest is 80% of that (600 trees). Using 200 trees cuts a
fit to ~8s, making weekly cadence practical (~30 min). Cadence is itself a
strategy parameter selected in phase 1.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"

FORM_END = 2022          # last season usable for strategy formation
REPLAY_START = 2023      # first season of the frozen live replay


# --------------------------------------------------------------- strategies
@dataclass
class Strategy:
    """A complete, frozen recipe for operating the model."""
    name: str
    cadence_days: int              # retrain every N days (0 = season boundary only)
    train_window_games: int | None  # None = use all history
    rf_trees: int = 200
    recency_halflife: int | None = None   # sample weighting, in games
    min_train: int = 4000

    def build(self):
        return _Ensemble(self.rf_trees)


class _Ensemble:
    def __init__(self, rf_trees: int = 200):
        self.rf_trees = rf_trees

    def fit(self, X, y, w=None):
        self.a = LogisticRegression(max_iter=3000, C=0.03).fit(X, y, sample_weight=w)
        self.b = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.015, max_depth=3, min_samples_leaf=240,
            l2_regularization=1.5, max_leaf_nodes=15,
            random_state=42).fit(X, y, sample_weight=w)
        self.c = RandomForestClassifier(
            n_estimators=self.rf_trees, max_depth=8, min_samples_leaf=40,
            n_jobs=-1, random_state=42).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return (self.a.predict_proba(X) + self.b.predict_proba(X)
                + self.c.predict_proba(X)) / 3


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"home_win", "total_runs", "home_score", "away_score", "season",
            "month", "dow", "total", "abs_margin", "f5_total", "f5_margin",
            "f5_home_lead"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_"))
            and c not in drop and df[c].dtype.kind in "fi"]


# ------------------------------------------------------------------ replay
def replay(df: pd.DataFrame, fc: list[str], strat: Strategy,
           start_season: int, end_season: int,
           verbose: bool = False) -> dict:
    """Day-by-day walk-forward over [start_season, end_season].

    Training data is everything strictly before the current day. The model is
    refit on the strategy's cadence; between refits the existing model predicts.
    """
    df = df.sort_values("date").reset_index(drop=True)
    test_mask = (df.season >= start_season) & (df.season <= end_season)
    days = sorted(df.loc[test_mask, "date"].unique())
    if not days:
        return {}

    model = None
    med = mu = sd = None
    last_fit: pd.Timestamp | None = None
    n_fits = 0
    rows = []
    t0 = time.time()

    for day in days:
        hist = df[df.date < day]
        if len(hist) < strat.min_train:
            continue

        need_fit = (
            model is None
            or (strat.cadence_days > 0
                and (day - last_fit).days >= strat.cadence_days)
            or (strat.cadence_days == 0
                and last_fit is not None
                and pd.Timestamp(day).year != pd.Timestamp(last_fit).year)
        )

        if need_fit:
            tr = hist
            if strat.train_window_games:
                tr = tr.tail(strat.train_window_games)
            med = tr[fc].median()
            Xtr = tr[fc].fillna(med).values
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            w = None
            if strat.recency_halflife:
                age = np.arange(len(tr))[::-1]
                w = 0.5 ** (age / strat.recency_halflife)
            model = strat.build().fit((Xtr - mu) / sd, tr.home_win.values, w)
            last_fit = day
            n_fits += 1

        today = df[df.date == day]
        Xte = today[fc].fillna(med).values
        p = model.predict_proba((Xte - mu) / sd)[:, 1]
        for (_, g), prob in zip(today.iterrows(), p):
            rows.append({"date": day, "season": int(g.season),
                         "p": float(prob), "y": int(g.home_win),
                         "train_n": len(hist)})

    if not rows:
        return {}
    r = pd.DataFrame(rows)
    acc = accuracy_score(r.y, (r.p >= 0.5).astype(int))
    n = len(r)
    out = {
        "strategy": strat.name,
        "cadence_days": strat.cadence_days,
        "train_window": strat.train_window_games,
        "rf_trees": strat.rf_trees,
        "halflife": strat.recency_halflife,
        "n_games": n,
        "n_fits": n_fits,
        "accuracy": float(acc),
        "se": float(np.sqrt(acc * (1 - acc) / n)),
        "auc": float(roc_auc_score(r.y, r.p)),
        "log_loss": float(log_loss(r.y, r.p)),
        "brier": float(np.mean((r.p - r.y) ** 2)),
        "seconds": round(time.time() - t0, 1),
    }
    by = []
    for s, g in r.groupby("season"):
        a = accuracy_score(g.y, (g.p >= 0.5).astype(int))
        by.append({"season": int(s), "n": len(g), "accuracy": float(a),
                   "auc": float(roc_auc_score(g.y, g.p))})
    out["per_season"] = by
    if verbose:
        print(f"    {strat.name:<28} acc={acc:.4f} auc={out['auc']:.4f} "
              f"fits={n_fits} ({out['seconds']:.0f}s)")
    return out, r


def main() -> None:
    df = pd.read_parquet(PROC / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_win"]).sort_values("date")
    fc = feature_cols(df)
    print(f"{len(df):,} games, {len(fc)} features")
    print(f"PHASE 1 formation: seasons <= {FORM_END}")
    print(f"PHASE 2 replay:    seasons >= {REPLAY_START}\n")

    candidates = [
        Strategy("season_boundary(baseline)", 0, None),
        Strategy("retrain_30d", 30, None),
        Strategy("retrain_14d", 14, None),
        Strategy("retrain_7d", 7, None),
        Strategy("retrain_14d_win8000", 14, 8000),
        Strategy("retrain_14d_win12000", 14, 12000),
        Strategy("retrain_14d_hl4000", 14, None, recency_halflife=4000),
        Strategy("retrain_14d_hl8000", 14, None, recency_halflife=8000),
    ]

    print("=== PHASE 1: strategy formation (2019-%d) ===" % FORM_END)
    form = []
    for s in candidates:
        res = replay(df, fc, s, 2019, FORM_END, verbose=True)
        if res:
            form.append(res[0])

    form.sort(key=lambda r: -r["accuracy"])
    best = form[0]
    bar = 1.96 * best["se"]
    print(f"\n  best: {best['strategy']} acc={best['accuracy']:.4f} "
          f"(±{bar:.4f}, n={best['n_games']:,})")
    tied = [f for f in form if best["accuracy"] - f["accuracy"] <= best["se"]]
    print(f"  within 1 SE ({len(tied)} strategies): "
          + ", ".join(f["strategy"] for f in tied))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "walkforward_formation.json").write_text(json.dumps(form, indent=2))
    print(f"\n-> reports/walkforward_formation.json")


if __name__ == "__main__" and "--phase2" not in sys.argv:
    main()


def phase2() -> None:
    """Frozen live replay. Strategy chosen in phase 1, never re-tuned."""
    df = pd.read_parquet(PROC / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_win"]).sort_values("date")
    fc = feature_cols(df)

    form = json.loads((REPORTS / "walkforward_formation.json").read_text())
    form.sort(key=lambda r: -r["accuracy"])
    winner = form[0]
    baseline = next(f for f in form if f["strategy"].startswith("season_boundary"))

    chosen = Strategy(winner["strategy"], winner["cadence_days"],
                      winner["train_window"], winner["rf_trees"],
                      winner["halflife"])
    base = Strategy(baseline["strategy"], baseline["cadence_days"],
                    baseline["train_window"], baseline["rf_trees"],
                    baseline["halflife"])

    print("=== PHASE 2: frozen live replay (%d+) ===" % REPLAY_START)
    print(f"strategy selected in phase 1: {chosen.name}")
    print(f"phase 1 accuracy was {winner['accuracy']:.4f}\n")

    out = {}
    for s, tag in ((chosen, "selected"), (base, "baseline")):
        res = replay(df, fc, s, REPLAY_START, 2026, verbose=False)
        if not res:
            continue
        r, raw = res
        out[tag] = r
        print(f"  {tag:<10} {s.name:<26} acc={r['accuracy']:.4f} "
              f"(±{1.96*r['se']:.4f})  auc={r['auc']:.4f}  n={r['n_games']:,}")
        for ps in r["per_season"]:
            print(f"      {ps['season']}  n={ps['n']:>5}  acc={ps['accuracy']:.4f}")
        raw.to_parquet(REPORTS / f"walkforward_replay_{tag}.parquet", index=False)

    if "selected" in out:
        drop = winner["accuracy"] - out["selected"]["accuracy"]
        se = out["selected"]["se"]
        print(f"\n  phase1 -> phase2 change: {-drop:+.4f}")
        print(f"  (noise band ±{1.96*se:.4f})")
        print(f"  verdict: {'HOLDS UP' if abs(drop) < 1.96*se else 'DEGRADED beyond noise'}")
    if "selected" in out and "baseline" in out:
        d = out["selected"]["accuracy"] - out["baseline"]["accuracy"]
        print(f"  selected vs baseline in replay: {d:+.4f}")

    (REPORTS / "walkforward_replay.json").write_text(json.dumps(out, indent=2))
    print("\n-> reports/walkforward_replay.json")


if __name__ == "__main__" and "--phase2" in sys.argv:
    phase2()
