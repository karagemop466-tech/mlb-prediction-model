"""Does weather improve the model beyond features it already has?

Correlation is not the bar. `h_park_factor` already encodes that Coors is a
hitters' park, and rolling run averages partly absorb seasonal temperature. The
question is INCREMENTAL value, measured walk-forward with a significance gate --
the same protocol that rejected bullpen fatigue and pitcher Statcast.

Three targets, because weather could plausibly move any of them:
    total runs  (regression, correlation with realized total)
    home win    (classification, accuracy)
    |margin|    (run-line relevance)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

WEATHER_COLS = [
    "air_density_index", "temp_f", "humidity", "pressure_hpa",
    "wind_out", "wind_cross", "gust_out", "gust_excess",
    "dew_point_f", "cloud_cover", "precip", "is_closed",
    "mlb_wind_out", "mlb_wind_mph",
]


def base_cols(df):
    drop = {"home_win", "total_runs", "home_score", "away_score", "season",
            "month", "dow", "total", "abs_margin"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_"))
            and c not in drop and df[c].dtype.kind in "fi"]


class Ens:
    def fit(self, X, y):
        self.a = LogisticRegression(max_iter=3000, C=0.03).fit(X, y)
        self.b = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.015, max_depth=3, min_samples_leaf=240,
            l2_regularization=1.5, max_leaf_nodes=15, random_state=42).fit(X, y)
        self.c = RandomForestClassifier(
            n_estimators=600, max_depth=8, min_samples_leaf=40,
            n_jobs=-1, random_state=42).fit(X, y)
        return self

    def predict_proba(self, X):
        return (self.a.predict_proba(X) + self.b.predict_proba(X)
                + self.c.predict_proba(X)) / 3


def reg_model():
    return HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.03, max_depth=3, min_samples_leaf=200,
        l2_regularization=2.0, max_leaf_nodes=12, random_state=42)


def wf_classify(df, cols):
    accs, aucs, ns = [], [], []
    for s in SEASONS:
        tr, te = df[df.season < s], df[df.season == s]
        if len(tr) < 2000 or len(te) < 100:
            continue
        med = tr[cols].median()
        Xtr, Xte = tr[cols].fillna(med).values, te[cols].fillna(med).values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = Ens().fit((Xtr - mu) / sd, tr.home_win.values)
        p = m.predict_proba((Xte - mu) / sd)[:, 1]
        accs.append(accuracy_score(te.home_win, (p >= .5).astype(int)))
        aucs.append(roc_auc_score(te.home_win, p))
        ns.append(len(te))
    w = np.array(ns) / sum(ns)
    acc = float(np.dot(accs, w))
    n = int(sum(ns))
    return {"accuracy": acc, "auc": float(np.dot(aucs, w)), "n": n,
            "se": float(np.sqrt(acc * (1 - acc) / n))}


def wf_regress(df, cols, target):
    corrs, maes, ns = [], [], []
    for s in SEASONS:
        tr, te = df[df.season < s], df[df.season == s]
        if len(tr) < 2000 or len(te) < 100:
            continue
        med = tr[cols].median()
        Xtr, Xte = tr[cols].fillna(med).values, te[cols].fillna(med).values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        a = reg_model().fit((Xtr - mu) / sd, tr[target].values)
        b = Ridge(alpha=50.0).fit((Xtr - mu) / sd, tr[target].values)
        p = 0.5 * a.predict((Xte - mu) / sd) + 0.5 * b.predict((Xte - mu) / sd)
        corrs.append(float(np.corrcoef(p, te[target].values)[0, 1]))
        maes.append(float(np.abs(p - te[target].values).mean()))
        ns.append(len(te))
    w = np.array(ns) / sum(ns)
    return {"corr": float(np.dot(corrs, w)), "mae": float(np.dot(maes, w)),
            "n": int(sum(ns))}


def main() -> None:
    feats = pd.read_parquet(PROC / "features.parquet")
    wx = pd.read_parquet(PROC / "weather_games.parquet")
    df = feats.merge(wx, on="game_id", how="left").sort_values("date")
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    df["total"] = df.home_score + df.away_score
    df["abs_margin"] = (df.home_score - df.away_score).abs()

    bc = base_cols(df)
    wc = [c for c in WEATHER_COLS if c in df.columns]
    cov = float(df["air_density_index"].notna().mean())
    print(f"{len(df):,} games, base {len(bc)} features, weather {len(wc)} "
          f"(coverage {cov:.1%})\n")

    results = {}

    print("=== TARGET: TOTAL RUNS ===")
    b = wf_regress(df, bc, "total")
    a = wf_regress(df, bc + wc, "total")
    print(f"  base            corr {b['corr']:.4f}  MAE {b['mae']:.4f}")
    print(f"  base + weather  corr {a['corr']:.4f}  MAE {a['mae']:.4f}")
    print(f"  delta corr {a['corr']-b['corr']:+.4f}   delta MAE {a['mae']-b['mae']:+.4f}")
    results["total"] = {"base": b, "with_weather": a,
                        "delta_corr": a["corr"] - b["corr"]}

    print("\n=== TARGET: HOME WIN ===")
    b = wf_classify(df, bc)
    a = wf_classify(df, bc + wc)
    bar = 1.96 * b["se"]
    delta = a["accuracy"] - b["accuracy"]
    print(f"  base            acc {b['accuracy']:.4f} (±{bar:.4f})  auc {b['auc']:.4f}")
    print(f"  base + weather  acc {a['accuracy']:.4f}          auc {a['auc']:.4f}")
    print(f"  delta acc {delta:+.4f}  bar {bar:+.4f}  "
          f"-> {'PROMOTE' if delta > bar else 'REJECT (within noise)'}")
    results["home_win"] = {"base": b, "with_weather": a, "delta_acc": delta,
                           "bar": bar, "promoted": bool(delta > bar)}

    print("\n=== TARGET: |MARGIN| (run line) ===")
    b = wf_regress(df, bc, "abs_margin")
    a = wf_regress(df, bc + wc, "abs_margin")
    print(f"  base            corr {b['corr']:.4f}  MAE {b['mae']:.4f}")
    print(f"  base + weather  corr {a['corr']:.4f}  MAE {a['mae']:.4f}")
    print(f"  delta corr {a['corr']-b['corr']:+.4f}")
    results["abs_margin"] = {"base": b, "with_weather": a,
                             "delta_corr": a["corr"] - b["corr"]}

    # Which weather variables actually carry the signal?
    print("\n=== ABLATION on total runs (add one family at a time) ===")
    fams = {
        "density only": ["air_density_index"],
        "raw thermo": ["temp_f", "humidity", "pressure_hpa"],
        "wind only": ["wind_out", "wind_cross"],
        "gusts only": ["gust_out", "gust_excess"],
        "MLB official wind": ["mlb_wind_out", "mlb_wind_mph"],
        "density + wind": ["air_density_index", "wind_out", "wind_cross"],
        "everything": wc,
    }
    base_corr = results["total"]["base"]["corr"]
    abl = {}
    for name, cols in fams.items():
        cols = [c for c in cols if c in df.columns]
        r = wf_regress(df, bc + cols, "total")
        abl[name] = r["corr"]
        print(f"  {name:<20} corr {r['corr']:.4f}  ({r['corr']-base_corr:+.4f})")
    results["ablation_total"] = abl

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "weather_research.json").write_text(json.dumps(results, indent=2))
    print(f"\n-> reports/weather_research.json")


if __name__ == "__main__":
    main()
