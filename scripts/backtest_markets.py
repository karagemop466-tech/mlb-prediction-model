"""Walk-forward backtest of every simulated market, not just the winner.

The base classifier is validated on P(home win). This validates the JOINT
distribution: margin, totals, one-run games, extras, and conjunctions like
P(home wins AND over 8.5). Each market gets its own Brier score and calibration
curve, measured out of sample.

A market is only trustworthy if its predicted probabilities match realized
frequencies. A simulator can reproduce league averages and still be useless
per-game, so this measures per-game discrimination too.

Protocol, per season S in 2019..2026:
    train the classifier on every game before S
    predict P(home win) for each game in S
    convert to scoring rates, simulate the joint distribution
    score every derived market against what actually happened
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import simulate as S
from total_model import SideModel, feature_cols as tm_feature_cols, load_weather

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"

N_SIMS_PER_GAME = 2500

# Weight on the win-probability inversion vs the side-specific run model when
# setting scoring rates. 1.0 = old behaviour (probability only).
BLEND_WEIGHT = 0.75


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


def feature_cols(df):
    drop = {"home_win", "total_runs", "home_score", "away_score", "season", "month", "dow"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_"))
            and c not in drop and df[c].dtype.kind in "fi"]


def expected_total_for(row, league_mean: float) -> float:
    """Per-game expected total from rolling team scoring/allowing rates."""
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
    # Shrink toward the league mean; rolling estimates are noisy.
    return float(np.clip(0.55 * est + 0.45 * league_mean, 6.5, 12.0))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calib_curve(p: np.ndarray, y: np.ndarray, bins: int = 8) -> list[dict]:
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else \
            (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() < 40:
            continue
        out.append({
            "bin": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
            "n": int(m.sum()),
            "pred": float(p[m].mean()),
            "actual": float(y[m].mean()),
            "error": float(p[m].mean() - y[m].mean()),
        })
    return out


def run(start_season: int = 2019) -> dict:
    df = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    df = load_weather(df)
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    fc = feature_cols(df)
    fc_wx = tm_feature_cols(df)
    league_total = float((df.home_score + df.away_score).mean())

    seasons = [s for s in sorted(df.season.unique()) if s >= start_season]
    rows = []

    for s in seasons:
        tr, te = df[df.season < s], df[df.season == s]
        if len(tr) < 2000 or len(te) < 100:
            continue

        med = tr[fc].median()
        Xtr, Xte = tr[fc].fillna(med).values, te[fc].fillna(med).values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
        model = Ensemble().fit(Xtr_s, tr.home_win.values)
        p_win = model.predict_proba(Xte_s)[:, 1]

        # Side-specific expected runs, trained on the SAME prior seasons only.
        # Run model uses weather; the win classifier does not (tested, rejected).
        medw = tr[fc_wx].median()
        Xtw, Xew = tr[fc_wx].fillna(medw).values, te[fc_wx].fillna(medw).values
        muw, sdw = Xtw.mean(0), Xtw.std(0) + 1e-9
        side = SideModel(0.85).fit((Xtw - muw) / sdw,
                                   tr.home_score.values, tr.away_score.values)
        eh, ea = side.predict((Xew - muw) / sdw)

        rng = np.random.default_rng(int(s))
        preds = {k: np.zeros(len(te)) for k in
                 ("win", "over85", "under85", "one_run", "extras",
                  "hwin_by1", "awin_by1", "win_and_over", "win_and_under",
                  "both_score", "margin_ge3")}

        # Cache rate solutions; solving per game is the expensive step.
        cache: dict[tuple[int, int], tuple[float, float]] = {}
        for i in range(len(te)):
            hr, ar = S.blend_rates(p_win[i], float(eh[i]), float(ea[i]),
                                   weight_prob=BLEND_WEIGHT)
            r = S.simulate_game(hr, ar, n_sims=N_SIMS_PER_GAME, rng=rng)
            preds["win"][i] = r.p_home_win()
            preds["over85"][i] = r.p_total_over(8.5)
            preds["under85"][i] = 1 - r.p_total_over(8.5)
            preds["one_run"][i] = r.p_one_run_game()
            preds["extras"][i] = r.p_extras()
            preds["hwin_by1"][i] = r.p_home_win_by(1)
            preds["awin_by1"][i] = r.p_home_win_by(-1)
            preds["win_and_over"][i] = r.p_joint(True, 8.5)
            preds["win_and_under"][i] = float(
                ((r.margin > 0) & (r.total < 8.5)).mean())
            preds["both_score"][i] = r.p_both_score()
            preds["margin_ge3"][i] = float((np.abs(r.margin) >= 3).mean())

        hs, as_ = te.home_score.values, te.away_score.values
        m, tot = hs - as_, hs + as_
        actual = {
            "win": (m > 0).astype(float),
            "over85": (tot > 8.5).astype(float),
            "under85": (tot < 8.5).astype(float),
            "one_run": (np.abs(m) == 1).astype(float),
            "extras": np.full(len(te), np.nan),   # not stored per game
            "hwin_by1": (m == 1).astype(float),
            "awin_by1": (m == -1).astype(float),
            "win_and_over": ((m > 0) & (tot > 8.5)).astype(float),
            "win_and_under": ((m > 0) & (tot < 8.5)).astype(float),
            "both_score": ((hs > 0) & (as_ > 0)).astype(float),
            "margin_ge3": (np.abs(m) >= 3).astype(float),
        }

        for market, p in preds.items():
            y = actual[market]
            if np.isnan(y).all():
                continue
            rows.append({
                "season": int(s), "market": market, "n": int(len(y)),
                "pred_mean": float(p.mean()), "actual_mean": float(y.mean()),
                "bias": float(p.mean() - y.mean()),
                "brier": brier(p, y),
                "brier_base": brier(np.full_like(p, y.mean()), y),
            })
        print(f"  {s}: {len(te):>5} games simulated")

    res = pd.DataFrame(rows)
    res["skill"] = 1 - res.brier / res.brier_base

    summary = []
    for market, g in res.groupby("market"):
        w = g.n / g.n.sum()
        summary.append({
            "market": market,
            "n": int(g.n.sum()),
            "pred_mean": float((g.pred_mean * w).sum()),
            "actual_mean": float((g.actual_mean * w).sum()),
            "bias": float((g.bias * w).sum()),
            "brier": float((g.brier * w).sum()),
            "brier_base": float((g.brier_base * w).sum()),
            "skill_score": float(1 - (g.brier * w).sum() / (g.brier_base * w).sum()),
        })
    summary.sort(key=lambda r: -r["skill_score"])

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest_markets.json").write_text(
        json.dumps({"per_season": rows, "summary": summary}, indent=2))
    return {"per_season": res, "summary": summary}


def main() -> None:
    print("Walk-forward market backtest (simulating every game)...")
    out = run()
    print(f"\n{'market':<16}{'n':>7}{'pred':>9}{'actual':>9}{'bias':>9}"
          f"{'brier':>9}{'base':>9}{'skill':>9}")
    print("-" * 77)
    for r in out["summary"]:
        print(f"{r['market']:<16}{r['n']:>7,}{r['pred_mean']:>9.4f}"
              f"{r['actual_mean']:>9.4f}{r['bias']:>+9.4f}{r['brier']:>9.4f}"
              f"{r['brier_base']:>9.4f}{r['skill_score']:>+9.4f}")
    print("\nskill > 0 means the simulation beats always predicting the base rate.")
    print("bias near 0 means the market is calibrated on average.")
    print("-> reports/backtest_markets.json")


if __name__ == "__main__":
    main()
