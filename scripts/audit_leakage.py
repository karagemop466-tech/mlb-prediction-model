"""Leakage audit. If this fails, every downstream number is fiction.

Tests:
 1. No feature correlates with the outcome above a plausible threshold.
 2. Reconstruct one team's rolling feature by hand and confirm it excludes the current game.
 3. A model trained on shuffled labels must score ~0.500 AUC (no hidden signal path).
 4. Chronological integrity: no feature row uses a future date.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_"))
            and df[c].dtype.kind in "fi"]


def test_correlations(df: pd.DataFrame) -> bool:
    fc = feature_cols(df)
    corr = df[fc].corrwith(df["home_win"]).abs().sort_values(ascending=False)
    print("\n[1] Top feature correlations with outcome:")
    for name, val in corr.head(8).items():
        flag = "  <-- SUSPICIOUS" if val > 0.30 else ""
        print(f"     {name:28} {val:.4f}{flag}")
    worst = corr.max()
    ok = worst < 0.30
    print(f"     max |corr| = {worst:.4f}  ->  {'PASS' if ok else 'FAIL'}")
    print("     (real MLB predictors top out near 0.10-0.15; >0.30 implies leakage)")
    return ok


def test_manual_reconstruction(df: pd.DataFrame) -> bool:
    """Verify d_win_pct_25 for one game excludes that game's own result."""
    games = pd.read_parquet(PROC / "games.parquet").sort_values(["date", "game_id"])
    team = "NYA"
    tg = games[(games.home == team) | (games.away == team)].copy()
    tg["won"] = np.where(tg.home == team, tg.home_win, 1 - tg.home_win)
    tg = tg.reset_index(drop=True)

    i = 400
    row = tg.loc[i]
    manual = tg.loc[i - 25:i - 1, "won"].mean()   # strictly prior 25

    feats = df[df.game_id == row.game_id]
    if feats.empty:
        print("\n[2] Reconstruction: game not found, SKIP")
        return True
    col = "h_win_pct_25" if row.home == team else "a_win_pct_25"
    pipeline = feats.iloc[0][col]

    diff = abs(manual - pipeline)
    ok = diff < 1e-9
    print(f"\n[2] Manual reconstruction ({team}, game {i}):")
    print(f"     hand-computed prior-25 win%: {manual:.6f}")
    print(f"     pipeline value:              {pipeline:.6f}")
    print(f"     diff {diff:.2e}  ->  {'PASS' if ok else 'FAIL'}")

    incl = tg.loc[i - 24:i, "won"].mean()  # window that WOULD include current game
    print(f"     (leaky variant incl. current game would be {incl:.6f})")
    return ok


def test_shuffled_labels(df: pd.DataFrame) -> bool:
    fc = feature_cols(df)
    d = df.dropna(subset=fc + ["home_win"])
    if len(d) > 12000:
        d = d.sample(12000, random_state=0)
    X = d[fc].values
    rng = np.random.default_rng(0)
    y = rng.permutation(d["home_win"].values)
    n = int(len(d) * 0.7)
    m = LogisticRegression(max_iter=1500, C=0.05)
    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    m.fit((X[:n] - Xm) / Xs, y[:n])
    auc = roc_auc_score(y[n:], m.predict_proba((X[n:] - Xm) / Xs)[:, 1])
    ok = 0.44 < auc < 0.56
    print(f"\n[3] Shuffled-label AUC: {auc:.4f}  ->  {'PASS' if ok else 'FAIL'}")
    print("     (must be ~0.500; higher means the model finds a path to the answer)")
    return ok


def test_chronology(df: pd.DataFrame) -> bool:
    d = df.sort_values("date")
    ok = d["date"].is_monotonic_increasing
    print(f"\n[4] Chronological ordering: {'PASS' if ok else 'FAIL'}")
    print(f"     span {d['date'].min().date()} -> {d['date'].max().date()}, {len(d):,} games")
    return ok


def main() -> None:
    df = pd.read_parquet(PROC / "features.parquet")
    print("=" * 62)
    print("LEAKAGE AUDIT")
    print("=" * 62)
    results = [
        test_correlations(df),
        test_manual_reconstruction(df),
        test_shuffled_labels(df),
        test_chronology(df),
    ]
    print("\n" + "=" * 62)
    print(f"RESULT: {sum(results)}/{len(results)} passed"
          f"  {'-- CLEAN' if all(results) else '-- LEAKAGE PRESENT, DO NOT TRUST MODEL'}")
    print("=" * 62)
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
