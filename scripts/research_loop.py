"""Self-improving research loop.

Maintains a registry of hypotheses, tests each with an identical walk-forward
protocol, applies a statistical significance gate, and promotes only what passes.
Every result -- especially failures -- is appended to reports/research_log.json
so knowledge accumulates instead of being re-discovered.

WHY THE GATE MATTERS
--------------------
Testing many ideas and shipping the best-looking one is itself overfitting. This
project already has a case study: pitcher-level Statcast looked like +1.0%
accuracy on one season (n=2,189, CI +/-2.08%) and came out at -0.01% across four
seasons (n=7,939). Without a gate it would have shipped.

Rules enforced here:
  1. Improvement must exceed the 95% CI of the baseline. Point estimates alone
     never promote an idea.
  2. Every idea is evaluated on the SAME seasons with the SAME protocol.
  3. A Bonferroni-style penalty widens the bar as more ideas are tested in a
     single session, because the chance of a fluke rises with the count.
  4. Results are logged whether they pass or fail.

Add a hypothesis by writing a function that takes the games frame and returns a
DataFrame of new feature columns keyed by game_id, then registering it below.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"
LOG = REPORTS / "research_log.json"

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


# ---------------------------------------------------------------- model
class Ensemble:
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


def base_feature_cols(df):
    drop = {"home_win", "total_runs", "home_score", "away_score", "season",
            "month", "dow"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_"))
            and c not in drop and df[c].dtype.kind in "fi"]


def evaluate(df: pd.DataFrame, cols: list[str], seasons=TEST_SEASONS) -> dict:
    accs, lls, aucs, ns = [], [], [], []
    for s in seasons:
        tr, te = df[df.season < s], df[df.season == s]
        if len(tr) < 2000 or len(te) < 100:
            continue
        med = tr[cols].median()
        Xtr, Xte = tr[cols].fillna(med).values, te[cols].fillna(med).values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = Ensemble().fit((Xtr - mu) / sd, tr.home_win.values)
        p = m.predict_proba((Xte - mu) / sd)[:, 1]
        accs.append(accuracy_score(te.home_win, (p >= .5).astype(int)))
        lls.append(log_loss(te.home_win, p))
        aucs.append(roc_auc_score(te.home_win, p))
        ns.append(len(te))
    if not ns:
        return {"accuracy": .5, "auc": .5, "log_loss": 9.9, "n": 0, "se": 0}
    w = np.array(ns) / sum(ns)
    acc = float(np.dot(accs, w))
    n = int(sum(ns))
    return {"accuracy": acc, "auc": float(np.dot(aucs, w)),
            "log_loss": float(np.dot(lls, w)), "n": n,
            "se": float(np.sqrt(acc * (1 - acc) / n))}


# ---------------------------------------------------------------- hypotheses
def h_bullpen_load(games: pd.DataFrame) -> pd.DataFrame:
    """Relief innings absorbed over the prior 3 days.

    Rationale: a bullpen that threw heavily in the last two games has its best
    arms unavailable. Not captured by any team-form feature, which averages over
    10-100 games. This is the top remaining idea from FINDINGS.md.

    Proxy: (total runs allowed) - (runs allowed attributable to the starter is
    unknown here), so we use games played in a short window plus recent runs
    allowed as a fatigue signal.
    """
    long = []
    for _, r in games.iterrows():
        long.append((r.game_id, r.date, r.home, r.away_score))
        long.append((r.game_id, r.date, r.away, r.home_score))
    L = pd.DataFrame(long, columns=["game_id", "date", "team", "ra"])
    L = L.sort_values(["team", "date"])
    out = []
    for team, g in L.groupby("team", sort=False):
        g = g.sort_values("date").copy()
        idx = g.set_index("date")
        g["bp_ra_3d"] = idx["ra"].shift(1).rolling("3D").sum().values
        g["bp_games_3d"] = idx["ra"].shift(1).rolling("3D").count().values
        g["bp_ra_5d"] = idx["ra"].shift(1).rolling("5D").sum().values
        out.append(g)
    L = pd.concat(out, ignore_index=True)
    cols = ["bp_ra_3d", "bp_games_3d", "bp_ra_5d"]
    h = L.merge(games[["game_id", "home"]], on="game_id")
    h = h[h.team == h.home].set_index("game_id")[cols].add_prefix("h_")
    a = L.merge(games[["game_id", "away"]], on="game_id")
    a = a[a.team == a.away].set_index("game_id")[cols].add_prefix("a_")
    res = h.join(a, how="outer")
    for c in cols:
        res[f"d_{c}"] = res[f"h_{c}"] - res[f"a_{c}"]
    return res


def h_travel_fatigue(games: pd.DataFrame) -> pd.DataFrame:
    """Consecutive road games and home/road transitions.

    Rationale: long road trips and the first game back are physically distinct
    situations that season-long home/road splits average away.
    """
    long = []
    for _, r in games.iterrows():
        long.append((r.game_id, r.date, r.home, 1))
        long.append((r.game_id, r.date, r.away, 0))
    L = pd.DataFrame(long, columns=["game_id", "date", "team", "is_home"])
    L = L.sort_values(["team", "date"])
    out = []
    for team, g in L.groupby("team", sort=False):
        g = g.sort_values("date").copy()
        prev = g["is_home"].shift(1)
        grp = (prev != prev.shift()).cumsum()
        g["trip_len"] = prev.groupby(grp).cumcount() + 1
        g["just_switched"] = (prev != g["is_home"]).astype(float)
        g["road_trip_len"] = np.where(prev == 0, g["trip_len"], 0)
        out.append(g)
    L = pd.concat(out, ignore_index=True)
    cols = ["trip_len", "just_switched", "road_trip_len"]
    h = L[L.is_home == 1].set_index("game_id")[cols].add_prefix("h_")
    a = L[L.is_home == 0].set_index("game_id")[cols].add_prefix("a_")
    res = h.join(a, how="outer")
    for c in cols:
        res[f"d_{c}"] = res[f"h_{c}"] - res[f"a_{c}"]
    return res


def h_recent_form_momentum(games: pd.DataFrame) -> pd.DataFrame:
    """Short-window form relative to long-window form.

    Rationale: teams have hot and cold stretches driven by injuries and callups
    that a 100-game average lags badly. Tests whether the RATIO of 7-game to
    50-game performance carries signal beyond the levels themselves.
    """
    long = []
    for _, r in games.iterrows():
        long.append((r.game_id, r.date, r.home, r.home_win, r.home_score - r.away_score))
        long.append((r.game_id, r.date, r.away, 1 - r.home_win, r.away_score - r.home_score))
    L = pd.DataFrame(long, columns=["game_id", "date", "team", "won", "diff"])
    L = L.sort_values(["team", "date"])
    out = []
    for team, g in L.groupby("team", sort=False):
        g = g.sort_values("date").copy()
        s = g["won"].shift(1)
        d = g["diff"].shift(1)
        g["form7"] = s.rolling(7, min_periods=4).mean()
        g["form50"] = s.rolling(50, min_periods=20).mean()
        g["form_delta"] = g["form7"] - g["form50"]
        g["rdiff7"] = d.rolling(7, min_periods=4).mean()
        g["rdiff50"] = d.rolling(50, min_periods=20).mean()
        g["rdiff_delta"] = g["rdiff7"] - g["rdiff50"]
        out.append(g)
    L = pd.concat(out, ignore_index=True)
    cols = ["form_delta", "rdiff_delta", "form7", "rdiff7"]
    h = L.merge(games[["game_id", "home"]], on="game_id")
    h = h[h.team == h.home].set_index("game_id")[cols].add_prefix("h_")
    a = L.merge(games[["game_id", "away"]], on="game_id")
    a = a[a.team == a.away].set_index("game_id")[cols].add_prefix("a_")
    res = h.join(a, how="outer")
    for c in cols:
        res[f"d_{c}"] = res[f"h_{c}"] - res[f"a_{c}"]
    return res


HYPOTHESES = {
    "bullpen_load": (h_bullpen_load,
                     "Relief workload over prior 3-5 days depletes late-inning quality"),
    "travel_fatigue": (h_travel_fatigue,
                       "Road trip length and home/road transitions affect performance"),
    "form_momentum": (h_recent_form_momentum,
                      "Short-vs-long form divergence captures injuries/callups"),
}


# ---------------------------------------------------------------- loop
def load_log() -> list:
    if LOG.exists():
        try:
            return json.loads(LOG.read_text())
        except Exception:
            return []
    return []


def main() -> None:
    feats = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    feats = feats.dropna(subset=["home_win"])
    games = pd.read_parquet(PROC / "games.parquet").sort_values("date")

    base_cols = base_feature_cols(feats)
    print(f"Baseline: {len(feats):,} games, {len(base_cols)} features")
    base = evaluate(feats, base_cols)
    print(f"  accuracy {base['accuracy']:.4f} (±{1.96*base['se']:.4f})  "
          f"auc {base['auc']:.4f}  n={base['n']:,}\n")

    history = load_log()
    tested = {h["hypothesis"] for h in history}
    todo = [k for k in HYPOTHESES if k not in tested] or list(HYPOTHESES)

    # Multiple-comparisons penalty: more ideas per session -> higher bar.
    k = len(todo)
    z = 1.96 + 0.45 * np.log(max(k, 1))
    print(f"Testing {k} hypotheses. Significance bar: {z:.2f} SE "
          f"(1.96 base + multiple-comparison penalty)\n")

    results = []
    for name in todo:
        fn, rationale = HYPOTHESES[name]
        print(f"[{name}] {rationale}")
        try:
            new = fn(games)
        except Exception as err:
            print(f"  ERROR building features: {type(err).__name__}: {err}\n")
            results.append({"hypothesis": name, "status": "error",
                            "error": f"{type(err).__name__}: {err}"})
            continue

        merged = feats.merge(new, left_on="game_id", right_index=True, how="left")
        new_cols = [c for c in new.columns if merged[c].dtype.kind in "fi"]
        cov = float(merged[new_cols[0]].notna().mean()) if new_cols else 0.0
        res = evaluate(merged, base_cols + new_cols)

        delta = res["accuracy"] - base["accuracy"]
        bar = z * base["se"]
        passed = delta > bar

        print(f"  +{len(new_cols)} features, coverage {cov:.1%}")
        print(f"  accuracy {res['accuracy']:.4f} vs {base['accuracy']:.4f}  "
              f"delta {delta:+.4f}  bar {bar:+.4f}")
        print(f"  auc {res['auc']:.4f} vs {base['auc']:.4f} "
              f"({res['auc']-base['auc']:+.4f})")
        print(f"  VERDICT: {'PROMOTE' if passed else 'REJECT (within noise)'}\n")

        results.append({
            "hypothesis": name, "rationale": rationale,
            "tested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_features": len(new_cols), "coverage": cov,
            "baseline_accuracy": base["accuracy"], "accuracy": res["accuracy"],
            "delta_accuracy": delta, "baseline_auc": base["auc"], "auc": res["auc"],
            "delta_auc": res["auc"] - base["auc"],
            "significance_bar": bar, "n": res["n"],
            "status": "promoted" if passed else "rejected",
        })

    history.extend(results)
    REPORTS.mkdir(exist_ok=True)
    LOG.write_text(json.dumps(history, indent=2))

    promoted = [r for r in results if r.get("status") == "promoted"]
    print("=" * 68)
    print(f"SESSION: {len(results)} tested, {len(promoted)} promoted")
    if promoted:
        for r in promoted:
            print(f"  PROMOTED {r['hypothesis']}: {r['delta_accuracy']:+.4f}")
    else:
        print("  Nothing cleared the significance bar. Baseline unchanged.")
        print("  This is the expected outcome most of the time and is not a failure;")
        print("  it is the gate doing its job.")
    print(f"-> {LOG}")


if __name__ == "__main__":
    main()
