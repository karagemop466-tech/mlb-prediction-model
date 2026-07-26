"""Forward-testing: predict today's/upcoming games and log them BEFORE they finish.

This is the honest test. Backtests can be fooled by subtle leakage or by
researcher degrees of freedom; a timestamped forward log cannot. Predictions are
appended to reports/forward_log.csv and scored later by `--score`.

    python scripts/predict.py            # predict today's slate
    python scripts/predict.py --date 2026-07-27
    python scripts/predict.py --score    # grade past logged predictions
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import features as F

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
REPORTS = ROOT / "reports"
LOG = REPORTS / "forward_log.csv"

TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHN",
    "CWS": "CHA", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KC": "KCA", "LAA": "ANA", "LAD": "LAN", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYN", "NYY": "NYA", "OAK": "OAK",
    "ATH": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SDN", "SF": "SFN",
    "SEA": "SEA", "STL": "SLN", "TB": "TBA", "TEX": "TEX", "TOR": "TOR",
    "WSH": "WAS",
}
INV = {v: k for k, v in TEAM_MAP.items()}


def feature_cols(df):
    drop = {"home_win", "total_runs", "home_score", "away_score", "season", "month", "dow"}
    return [c for c in df.columns
            if c.startswith(("h_", "a_", "d_", "sc_"))
            and c not in drop and df[c].dtype.kind in "fi"]


class Ensemble:
    """Best config found by optimize.py: logistic + GBM average."""

    def fit(self, X, y):
        self.a = LogisticRegression(max_iter=3000, C=0.005).fit(X, y)
        self.b = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.015, max_depth=3, min_samples_leaf=160,
            l2_regularization=1.5, max_leaf_nodes=15, random_state=42).fit(X, y)
        return self

    def predict_proba(self, X):
        return 0.5 * self.a.predict_proba(X) + 0.5 * self.b.predict_proba(X)


def upcoming(day: date) -> pd.DataFrame:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day.isoformat()}"
           f"&gameType=R&hydrate=probablePitcher,team")
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=60).read())
    rows = []
    for blk in data.get("dates", []):
        for g in blk.get("games", []):
            a, h = g["teams"]["away"], g["teams"]["home"]
            rows.append({
                "game_pk": g["gamePk"],
                "date": pd.Timestamp(g["officialDate"]),
                "away": TEAM_MAP.get(a["team"]["abbreviation"], a["team"]["abbreviation"]),
                "home": TEAM_MAP.get(h["team"]["abbreviation"], h["team"]["abbreviation"]),
                "away_name": a["team"]["name"], "home_name": h["team"]["name"],
                "status": g["status"]["abstractGameState"],
                "away_sp_name": (a.get("probablePitcher") or {}).get("fullName", "TBD"),
                "home_sp_name": (h.get("probablePitcher") or {}).get("fullName", "TBD"),
            })
    return pd.DataFrame(rows)


def latest_team_state(feat: pd.DataFrame) -> dict:
    """Most recent feature vector observed for each team (home and away views)."""
    state = {}
    for _, r in feat.sort_values("date").iterrows():
        for side, pre in (("home", "h_"), ("away", "a_")):
            team = r[side]
            vals = {c[2:]: r[c] for c in feat.columns if c.startswith(pre)}
            state[team] = vals
    return state


def build_matchup_rows(games: pd.DataFrame, feat: pd.DataFrame, fc: list[str]) -> pd.DataFrame:
    state = latest_team_state(feat)
    base = [c[2:] for c in feat.columns if c.startswith("h_")]
    rows = []
    for _, g in games.iterrows():
        h, a = state.get(g["home"], {}), state.get(g["away"], {})
        row = {}
        for c in base:
            hv, av = h.get(c, np.nan), a.get(c, np.nan)
            row[f"h_{c}"] = hv
            row[f"a_{c}"] = av
            row[f"d_{c}"] = (hv - av) if (pd.notna(hv) and pd.notna(av)) else np.nan
        rows.append(row)
    X = pd.DataFrame(rows)
    for c in fc:
        if c not in X.columns:
            X[c] = np.nan
    return X[fc]


def predict(day: date) -> pd.DataFrame:
    feat = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    feat = feat.dropna(subset=["home_win"])
    fc = feature_cols(feat)

    med = feat[fc].median()
    Xtr = feat[fc].fillna(med).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    model = Ensemble().fit((Xtr - mu) / sd, feat["home_win"].values)

    games = upcoming(day)
    if games.empty:
        print(f"[predict] no games scheduled {day}")
        return pd.DataFrame()

    X = build_matchup_rows(games, feat, fc).fillna(med).values
    p = model.predict_proba((X - mu) / sd)[:, 1]

    games["p_home_win"] = p
    games["p_away_win"] = 1 - p
    games["pick"] = np.where(p >= 0.5, games["home_name"], games["away_name"])
    games["confidence"] = np.maximum(p, 1 - p)
    games["fair_ml_home"] = [
        round(-100 * x / (1 - x)) if x >= 0.5 else round(100 * (1 - x) / x) for x in p
    ]
    games["predicted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
    games["model"] = "ensemble_log_gbm"
    return games


def log_predictions(g: pd.DataFrame) -> None:
    if g.empty:
        return
    cols = ["date", "game_pk", "away", "home", "away_name", "home_name",
            "away_sp_name", "home_sp_name", "p_home_win", "pick", "confidence",
            "fair_ml_home", "predicted_at", "model"]
    out = g[cols].copy()
    REPORTS.mkdir(exist_ok=True)
    if LOG.exists():
        old = pd.read_csv(LOG)
        out = pd.concat([old, out], ignore_index=True)
        out = out.drop_duplicates(subset=["game_pk"], keep="last")
    out.to_csv(LOG, index=False)
    print(f"[predict] logged {len(g)} -> {LOG.relative_to(ROOT)} ({len(out)} total)")


def score() -> None:
    if not LOG.exists():
        print("[score] no forward log yet")
        return
    log = pd.read_csv(LOG)
    res = pd.read_parquet(PROC / "games.parquet")
    res["date"] = pd.to_datetime(res["date"])
    log["date"] = pd.to_datetime(log["date"])
    m = log.merge(res[["date", "home", "away", "home_win"]],
                  on=["date", "home", "away"], how="inner")
    if m.empty:
        print("[score] no logged predictions have completed yet")
        return
    m["correct"] = ((m.p_home_win >= 0.5).astype(int) == m.home_win).astype(int)
    ll = -np.mean(m.home_win * np.log(m.p_home_win.clip(1e-6, 1 - 1e-6))
                  + (1 - m.home_win) * np.log((1 - m.p_home_win).clip(1e-6, 1 - 1e-6)))
    print(f"[score] {len(m)} graded  accuracy={m.correct.mean():.4f}  log_loss={ll:.4f}")
    print(f"[score] backtest expectation: accuracy~0.568  log_loss~0.678")
    hi = m[m.confidence >= 0.60]
    if len(hi) >= 10:
        print(f"[score] high-confidence (>=0.60): {len(hi)} games, "
              f"accuracy={hi.correct.mean():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    if args.score:
        score()
        return

    day = date.fromisoformat(args.date) if args.date else date.today()
    g = predict(day)
    if g.empty:
        return

    print(f"\n{'MATCHUP':<46}{'PICK':<24}{'CONF':>7}{'FAIR ML':>9}")
    print("-" * 88)
    for _, r in g.sort_values("confidence", ascending=False).iterrows():
        mu = f"{r.away_name} @ {r.home_name}"
        ml = f"{r.fair_ml_home:+.0f}" if r.p_home_win >= .5 else f"{-r.fair_ml_home:+.0f}"
        print(f"{mu:<46}{r['pick']:<24}{r.confidence:>7.3f}{ml:>9}")
    print(f"\n{len(g)} games. Home-team probabilities from {g.iloc[0]['model']}.")
    print("Edges vs market require your own odds; see scripts/roi.py.")

    if not args.no_log:
        log_predictions(g)


if __name__ == "__main__":
    main()
