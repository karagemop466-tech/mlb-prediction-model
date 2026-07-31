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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import features as F
import simulate as SIM
import pricing as PRICE
from total_model import SideModel, WEATHER_COLS, load_weather
from weather import (density_index, fetch_forecast, wind_cross_component,
                     wind_out_component)

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
    """Production model, selected by optimize_accuracy.py on ACCURACY.

    Three-way average of logistic regression, gradient boosting and random
    forest. All candidates were statistically tied (difference well inside the +/-0.0074
    confidence interval), so the ensemble is preferred because averaging three
    decorrelated learners reduces variance rather than chasing a noisy winner.

    Full walk-forward 2019-2026: 56.92% accuracy, 0.5912 AUC, 17,046 games.
    """

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
        return (self.a.predict_proba(X)
                + self.b.predict_proba(X)
                + self.c.predict_proba(X)) / 3


def live_weather(games: pd.DataFrame) -> pd.DataFrame:
    """Attach forecast weather to today's slate, keyed on venue + first pitch."""
    vpath = PROC / "venues.json"
    if not vpath.exists():
        for c in WEATHER_COLS:
            games[c] = np.nan
        return games
    venues = json.loads(vpath.read_text())
    by_name = {v["name"]: v for v in venues.values()}

    rows = []
    cache: dict[int, pd.DataFrame] = {}
    for _, g in games.iterrows():
        v = by_name.get(g.get("venue_name"))
        rec = {c: np.nan for c in WEATHER_COLS}
        if v and v.get("lat") is not None:
            vid = v["id"]
            if vid not in cache:
                cache[vid] = fetch_forecast(v["lat"], v["lon"], v["tz"])
            hourly = cache[vid]
            closed = 1 if v.get("roof") == "Dome" else 0
            if hourly is not None and not hourly.empty:
                t = pd.to_datetime(g.get("game_time"), errors="coerce", utc=True)
                target = None
                if pd.notna(t):
                    try:
                        target = t.tz_convert(v["tz"]).tz_localize(None)
                    except Exception:
                        target = None
                if target is None:
                    target = pd.Timestamp(g["date"]).replace(hour=19)
                idx = (hourly["time"] - target).abs().idxmin()
                r = hourly.loc[idx]
                az = v.get("azimuth")
                temp = float(r["temperature_2m"]); rh = float(r["relative_humidity_2m"])
                pres = float(r["surface_pressure"]); wspd = float(r["wind_speed_10m"])
                wdir = float(r["wind_direction_10m"]); gust = float(r["wind_gusts_10m"])
                rec.update({
                    "temp_f": temp, "humidity": rh, "pressure_hpa": pres,
                    "dew_point_f": float(r.get("dew_point_2m", np.nan)),
                    "precip": float(r.get("precipitation", 0) or 0),
                    "cloud_cover": float(r.get("cloud_cover", np.nan)),
                    "air_density_index": density_index(temp, rh, pres),
                    "wind_out": wind_out_component(wdir, wspd, az) if az is not None else np.nan,
                    "wind_cross": wind_cross_component(wdir, wspd, az) if az is not None else np.nan,
                    "gust_out": wind_out_component(wdir, gust, az) if az is not None else np.nan,
                    "gust_excess": gust - wspd,
                    "is_closed": closed,
                })
                if closed:
                    for c in ("wind_out", "wind_cross", "gust_out", "gust_excess"):
                        rec[c] = 0.0
        rows.append(rec)
    wxdf = pd.DataFrame(rows, index=games.index)
    for c in WEATHER_COLS:
        games[c] = wxdf[c]
    return games


def upcoming(day: date) -> pd.DataFrame:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day.isoformat()}"
           f"&gameType=R&hydrate=probablePitcher,team,venue")
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
                "game_time": g.get("gameDate", ""),
                "venue_name": (g.get("venue") or {}).get("name"),
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

    # Separate frame for the run model, which DOES use weather.
    feat_wx = load_weather(feat)
    from total_model import feature_cols as tm_cols
    fc_wx = tm_cols(feat_wx)

    med = feat[fc].median()
    Xtr = feat[fc].fillna(med).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_s = (Xtr - mu) / sd
    model = Ensemble().fit(Xtr_s, feat["home_win"].values)
    # Side-specific expected runs, so totals reflect the actual matchup rather
    # than being implied by the win probability alone.
    med_wx = feat_wx[fc_wx].median()
    Xw = feat_wx[fc_wx].fillna(med_wx).values
    mu_w, sd_w = Xw.mean(0), Xw.std(0) + 1e-9
    side = SideModel(0.85).fit((Xw - mu_w) / sd_w,
                               feat_wx.home_score.values, feat_wx.away_score.values)

    games = upcoming(day)
    if games.empty:
        print(f"[predict] no games scheduled {day}")
        return pd.DataFrame()

    X = build_matchup_rows(games, feat, fc).fillna(med).values
    Xs = (X - mu) / sd
    p = model.predict_proba(Xs)[:, 1]

    # Live forecast weather for the run model.
    games = live_weather(games)
    Xrow = build_matchup_rows(games, feat, fc)
    for c in WEATHER_COLS:
        if c in fc_wx:
            Xrow[c] = games[c].values
    Xw_live = Xrow.reindex(columns=fc_wx).fillna(med_wx).values
    exp_h, exp_a = side.predict((Xw_live - mu_w) / sd_w)

    games["p_home_win"] = p
    games["p_away_win"] = 1 - p
    games["pick"] = np.where(p >= 0.5, games["home_name"], games["away_name"])
    games["confidence"] = np.maximum(p, 1 - p)
    games["fair_ml_home"] = [
        round(-100 * x / (1 - x)) if x >= 0.5 else round(100 * (1 - x) / x) for x in p
    ]
    games["predicted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
    games["model"] = "ensemble_log_gbm_rf"

    # --- Simulate the joint distribution for every game.
    rng = np.random.default_rng(20260730)
    sims = []
    for i, (_, g) in enumerate(games.iterrows()):
        hr, ar = SIM.blend_rates(float(p[i]), float(exp_h[i]), float(exp_a[i]),
                                 weight_prob=0.75)
        r = SIM.simulate_game(hr, ar, n_sims=12000, rng=rng)
        sims.append({
            "p_over_8_5": r.p_total_over(8.5),
            "p_over_9_5": r.p_total_over(9.5),
            "p_one_run": r.p_one_run_game(),
            "p_extras": r.p_extras(),
            "p_home_by_1": r.p_home_win_by(1),
            "p_away_by_1": r.p_home_win_by(-1),
            "p_home_win_and_over": r.p_joint(True, 8.5),
            "p_away_win_and_over": r.p_joint(False, 8.5),
            "p_both_score": r.p_both_score(),
            "p_margin_ge3": float((np.abs(r.margin) >= 3).mean()),
            "exp_total": float(r.total.mean()),
            "fair_ml_home_sim": PRICE.prob_to_american(r.p_home_win()),
            "exp_home_runs": float(exp_h[i]),
            "exp_away_runs": float(exp_a[i]),
        })
    for k in sims[0]:
        games[k] = [s_[k] for s_ in sims]
    return games


def log_predictions(g: pd.DataFrame) -> None:
    if g.empty:
        return
    cols = ["date", "game_pk", "away", "home", "away_name", "home_name",
            "away_sp_name", "home_sp_name", "p_home_win", "pick", "confidence",
            "fair_ml_home", "predicted_at", "model",
            "p_over_8_5", "p_over_9_5", "p_one_run", "p_extras",
            "p_home_by_1", "p_away_by_1", "p_home_win_and_over",
            "p_away_win_and_over", "p_both_score", "p_margin_ge3", "exp_total"]
    cols = [c for c in cols if c in g.columns]
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
    # Drop grades from any earlier run so the merge cannot create _x/_y columns.
    log = log.drop(columns=[c for c in ("home_win", "correct") if c in log.columns])
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

    # --- Grade every simulated market, not just the winner.
    mm = m.merge(res[["date", "home", "away", "home_score", "away_score"]],
                 on=["date", "home", "away"], how="left")
    hs = pd.to_numeric(mm["home_score"], errors="coerce").values.astype(float)
    as_ = pd.to_numeric(mm["away_score"], errors="coerce").values.astype(float)
    margin, total = hs - as_, hs + as_

    market_defs = {
        "p_over_8_5": (total > 8.5),
        "p_over_9_5": (total > 9.5),
        "p_one_run": (np.abs(margin) == 1),
        "p_home_by_1": (margin == 1),
        "p_away_by_1": (margin == -1),
        "p_home_win_and_over": ((margin > 0) & (total > 8.5)),
        "p_both_score": ((hs > 0) & (as_ > 0)),
        "p_margin_ge3": (np.abs(margin) >= 3),
    }
    print(f"\n{'market':<22}{'n':>6}{'pred':>9}{'actual':>9}{'bias':>9}{'brier':>9}")
    print("-" * 64)
    rows = [("p_home_win",
             pd.to_numeric(mm["p_home_win"], errors="coerce").values,
             (margin > 0).astype(float))]
    for col, outcome in market_defs.items():
        if col not in mm.columns:
            continue
        pr = pd.to_numeric(mm[col], errors="coerce").values
        ok = ~np.isnan(pr) & ~np.isnan(hs)
        if ok.sum() < 5:
            continue
        rows.append((col, pr[ok], outcome[ok].astype(float)))
    for name, pr, y in rows:
        print(f"{name:<22}{len(y):>6}{pr.mean():>9.4f}{y.mean():>9.4f}"
              f"{pr.mean()-y.mean():>+9.4f}{np.mean((pr-y)**2):>9.4f}")

    # Persist grades back into the forward log so the site can render them.
    graded = m[["game_pk", "home_win", "correct"]]
    merged = log.drop(columns=[c for c in ("home_win", "correct") if c in log.columns])
    merged = merged.merge(graded, on="game_pk", how="left")
    merged.to_csv(LOG, index=False)
    print(f"[score] {len(m)} graded  accuracy={m.correct.mean():.4f}  log_loss={ll:.4f}")
    print(f"[score] backtest expectation: accuracy~0.569  log_loss~0.678")
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

    # Snapshot for the static site
    out = g.copy()
    if "game_time" in out.columns:
        t = pd.to_datetime(out["game_time"], errors="coerce", utc=True)
        out["game_time"] = t.dt.tz_convert("US/Eastern").dt.strftime("%-I:%M %p ET")
    cols = ["date", "game_pk", "away", "home", "away_name", "home_name",
            "away_sp_name", "home_sp_name", "p_home_win", "pick", "confidence",
            "fair_ml_home", "game_time",
            "p_over_8_5", "p_over_9_5", "p_one_run", "p_extras",
            "p_home_by_1", "p_away_by_1", "p_home_win_and_over",
            "p_away_win_and_over", "p_both_score", "p_margin_ge3",
            "exp_total", "exp_home_runs", "exp_away_runs",
            "temp_f", "humidity", "air_density_index", "wind_out",
            "gust_out", "is_closed"]
    cols = [c for c in cols if c in out.columns]
    REPORTS.mkdir(exist_ok=True)
    out[cols].to_csv(REPORTS / "today.csv", index=False)
    print(f"[predict] site snapshot -> reports/today.csv")


if __name__ == "__main__":
    main()
