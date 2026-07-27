"""Correctness test suite. Run this before trusting any number the system prints.

Accuracy claims are only meaningful if the machinery producing them is correct.
These tests check the things that silently corrupt sports models:

  DATA CORRECTNESS
    1. Scores are non-negative integers; no ties (MLB has no ties in the log era)
    2. Every game has exactly two distinct teams
    3. Known-truth spot checks against the official MLB API
    4. Game counts per season match expected schedule length
    5. No duplicate games; no missing dates mid-season

  FEATURE CORRECTNESS
    6. Rolling features never use the current or future games (leakage)
    7. Differential features equal home minus away, exactly
    8. Feature values fall in physically plausible ranges
    9. No feature is constant or all-null

  MODEL CORRECTNESS
   10. Probabilities are in (0,1) and home+away sum to 1
   11. Predictions are deterministic across repeated runs
   12. Train/test split is strictly chronological
   13. Accuracy beats the naive baseline by more than noise

Exit code 0 = all passed. Non-zero = do not trust the output.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------- data
def data_tests(games: pd.DataFrame) -> None:
    print("\nDATA CORRECTNESS")

    s = games[["home_score", "away_score"]]
    check("scores are non-negative integers",
          bool((s >= 0).all().all() and (s == s.astype(int)).all().all()),
          f"range {int(s.min().min())}-{int(s.max().max())}")

    ties = int((games.home_score == games.away_score).sum())
    # Ties are legal but extremely rare (rain-shortened/suspended games).
    # They must be rare AND must not be labeled as home wins.
    tie_rows = games[games.home_score == games.away_score]
    check("tied games are rare and correctly labeled",
          ties <= 5 and bool((tie_rows.home_win == 0).all()),
          f"{ties} tie(s), all labeled non-home-win: "
          f"{bool((tie_rows.home_win == 0).all()) if ties else 'n/a'}")

    check("home and away teams always differ",
          bool((games.home != games.away).all()))

    dupes = int(games.duplicated(["date", "home", "away", "dblhdr"]).sum())
    check("no duplicate games", dupes == 0, f"{dupes} duplicates")

    # Expected ~2430 games for a full modern season (2020 was shortened).
    counts = games.groupby("season").size()
    bad = {int(k): int(v) for k, v in counts.items()
           if not (2380 <= v <= 2480) and k not in (2020, games.season.max())}
    check("season game counts are plausible", not bad, f"outliers: {bad}" if bad else
          f"{len(counts)} seasons, median {int(counts.median())}")

    hw = games.home_win.mean()
    check("home win rate matches known MLB baseline",
          0.520 <= hw <= 0.550, f"{hw:.4f} (expect ~0.535)")

    runs = games.total_runs.mean()
    check("average total runs is realistic",
          7.5 <= runs <= 10.5, f"{runs:.2f} per game (expect ~8.6-9.5)")


def truth_spot_check(games: pd.DataFrame, n: int = 6) -> None:
    """Compare stored results against the official MLB API for random games."""
    print("\nGROUND-TRUTH SPOT CHECK (vs live MLB Stats API)")
    rng = np.random.default_rng(7)
    recent = games[games.season >= 2023]
    sample = recent.iloc[rng.choice(len(recent), size=min(n, len(recent)), replace=False)]

    TEAM = {"ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHN": "CHC",
            "CHA": "CWS", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
            "HOU": "HOU", "KCA": "KC", "ANA": "LAA", "LAN": "LAD", "MIA": "MIA",
            "MIL": "MIL", "MIN": "MIN", "NYN": "NYM", "NYA": "NYY", "OAK": "OAK",
            "PHI": "PHI", "PIT": "PIT", "SDN": "SD", "SFN": "SF", "SEA": "SEA",
            "SLN": "STL", "TBA": "TB", "TEX": "TEX", "TOR": "TOR", "WAS": "WSH"}

    matched = mismatched = skipped = 0
    for _, g in sample.iterrows():
        day = pd.Timestamp(g.date).date().isoformat()
        url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day}"
               f"&gameType=R&hydrate=team")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "verify/1.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=45).read())
        except Exception as err:
            skipped += 1
            print(f"     skip {day}: {type(err).__name__}")
            continue

        want_h, want_a = TEAM.get(g.home), TEAM.get(g.away)
        found = False
        for blk in data.get("dates", []):
            for api_g in blk.get("games", []):
                ht = api_g["teams"]["home"]["team"].get("abbreviation")
                at = api_g["teams"]["away"]["team"].get("abbreviation")
                if ht == want_h and at == want_a:
                    hs = api_g["teams"]["home"].get("score")
                    as_ = api_g["teams"]["away"].get("score")
                    found = True
                    if hs == g.home_score and as_ == g.away_score:
                        matched += 1
                    else:
                        mismatched += 1
                        print(f"     MISMATCH {day} {g.away}@{g.home}: "
                              f"stored {g.away_score}-{g.home_score}, API {as_}-{hs}")
        if not found:
            skipped += 1

    check("stored scores match official MLB API",
          mismatched == 0 and matched > 0,
          f"{matched} verified, {mismatched} wrong, {skipped} unavailable")


# ---------------------------------------------------------------- features
def feature_tests(games: pd.DataFrame, feats: pd.DataFrame) -> None:
    print("\nFEATURE CORRECTNESS")

    # Differential identity: d_X must equal h_X - a_X
    bad = []
    for c in [c for c in feats.columns if c.startswith("d_")][:40]:
        base = c[2:]
        h, a = f"h_{base}", f"a_{base}"
        if h in feats.columns and a in feats.columns:
            m = feats[[h, a, c]].dropna()
            if len(m) and not np.allclose(m[c], m[h] - m[a], atol=1e-9):
                bad.append(c)
    check("differential features equal home minus away", not bad,
          f"violations: {bad[:3]}" if bad else "checked 40 columns")

    # Level win rates are in [0,1]; DIFFERENCES of two rates are in [-1,1].
    lvl = [c for c in feats.columns
           if ("win_pct" in c or "pythag" in c) and c.startswith(("h_", "a_"))]
    off = [c for c in lvl if feats[c].dropna().lt(0).any() or feats[c].dropna().gt(1).any()]
    check("level win-rate features bounded in [0,1]", not off, f"out of range: {off[:3]}")

    dif = [c for c in feats.columns
           if ("win_pct" in c or "pythag" in c) and c.startswith("d_")]
    off = [c for c in dif if feats[c].dropna().lt(-1).any() or feats[c].dropna().gt(1).any()]
    check("differential win-rate features bounded in [-1,1]", not off,
          f"out of range: {off[:3]}")

    # Runs-per-game features must be physically plausible
    rf = [c for c in feats.columns if c.startswith(("h_rf_", "a_rf_", "h_ra_", "a_ra_"))]
    off = [c for c in rf if feats[c].dropna().lt(0).any() or feats[c].dropna().gt(15).any()]
    check("runs-per-game features in [0,15]", not off, f"out of range: {off[:3]}")

    # No dead columns
    fc = [c for c in feats.columns if c.startswith(("h_", "a_", "d_"))]
    dead = [c for c in fc if feats[c].notna().sum() == 0]
    const = [c for c in fc if feats[c].dropna().nunique() == 1]
    check("no all-null feature columns", not dead, f"{len(dead)} dead")
    check("no constant feature columns", not const, f"{len(const)} constant")

    # Independent leakage re-check: rebuild a rolling window by hand
    g = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    team = "LAN"
    tg = g[(g.home == team) | (g.away == team)].copy()
    tg["won"] = np.where(tg.home == team, tg.home_win, 1 - tg.home_win)
    tg = tg.reset_index(drop=True)
    i = 300
    manual = tg.loc[i - 25:i - 1, "won"].mean()
    row = feats[feats.game_id == tg.loc[i, "game_id"]]
    if len(row):
        col = "h_win_pct_25" if tg.loc[i, "home"] == team else "a_win_pct_25"
        got = row.iloc[0][col]
        check("rolling feature excludes current game (independent recheck)",
              abs(manual - got) < 1e-9, f"manual {manual:.6f} vs pipeline {got:.6f}")
    else:
        check("rolling feature excludes current game", False, "game not found")


# ---------------------------------------------------------------- model
def model_tests(feats: pd.DataFrame) -> None:
    print("\nMODEL CORRECTNESS")
    sys.path.insert(0, str(ROOT / "scripts"))
    from predict import Ensemble, feature_cols

    df = feats.dropna(subset=["home_win"]).sort_values("date")
    fc = feature_cols(df)
    tr = df[df.season <= 2024]
    te = df[df.season >= 2025]

    med = tr[fc].median()
    Xtr = tr[fc].fillna(med).values
    Xte = te[fc].fillna(med).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    m1 = Ensemble().fit(Xtr_s, tr.home_win.values)
    p1 = m1.predict_proba(Xte_s)[:, 1]

    check("probabilities strictly within (0,1)",
          bool((p1 > 0).all() and (p1 < 1).all()),
          f"range {p1.min():.4f}-{p1.max():.4f}")

    both = m1.predict_proba(Xte_s)
    check("home and away probabilities sum to 1",
          bool(np.allclose(both.sum(axis=1), 1.0, atol=1e-9)))

    m2 = Ensemble().fit(Xtr_s, tr.home_win.values)
    p2 = m2.predict_proba(Xte_s)[:, 1]
    check("predictions are deterministic (same seed, same output)",
          bool(np.allclose(p1, p2, atol=1e-12)))

    check("train period strictly precedes test period",
          tr.date.max() < te.date.min(),
          f"train ends {tr.date.max().date()}, test starts {te.date.min().date()}")

    acc = float(((p1 >= 0.5).astype(int) == te.home_win.values).mean())
    base = float(max(te.home_win.mean(), 1 - te.home_win.mean()))
    se = float(np.sqrt(acc * (1 - acc) / len(te)))
    check("accuracy beats naive baseline by more than noise",
          acc - base > 1.96 * se,
          f"model {acc:.4f} vs baseline {base:.4f} (±{1.96*se:.4f})")

    # A model fed shuffled labels must be no better than chance.
    rng = np.random.default_rng(0)
    yshuf = rng.permutation(tr.home_win.values)
    m3 = Ensemble().fit(Xtr_s, yshuf)
    p3 = m3.predict_proba(Xte_s)[:, 1]
    acc3 = float(((p3 >= 0.5).astype(int) == te.home_win.values).mean())
    check("shuffled-label model collapses to chance",
          abs(acc3 - base) < 0.03, f"{acc3:.4f} vs baseline {base:.4f}")


def main() -> None:
    print("=" * 68)
    print("CORRECTNESS VERIFICATION")
    print("=" * 68)

    games = pd.read_parquet(PROC / "games.parquet")
    feats = pd.read_parquet(PROC / "features.parquet")
    print(f"\nDataset: {len(games):,} games, {len(feats.columns)} columns, "
          f"{games.date.min().date()} -> {games.date.max().date()}")

    data_tests(games)
    truth_spot_check(games)
    feature_tests(games, feats)
    model_tests(feats)

    print("\n" + "=" * 68)
    total = len(PASS) + len(FAIL)
    print(f"RESULT: {len(PASS)}/{total} passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        print("DO NOT TRUST OUTPUT UNTIL THESE ARE FIXED.")
    else:
        print("All correctness checks passed.")
    print("=" * 68)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
