"""ROI / betting evaluation.

READ THIS FIRST
---------------
ROI is a property of (model, price), not of a model alone. A 57%-accurate model
loses money against -130 favorites and prints money against +110 dogs. Therefore
this module REFUSES to report ROI unless real historical odds are supplied.

Supply odds as CSV at data/raw/odds/odds.csv with columns:
    date,away,home,ml_home,ml_away        (American odds, e.g. -145 / +122)
Team codes may be Retrosheet or common abbreviations; both are matched.

Without that file, we report only the two things that ARE knowable from free
data: calibration quality and break-even sensitivity (what price you would need
in order to profit). Those are honest. A fabricated ROI number is not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
ODDS = ROOT / "data" / "raw" / "odds" / "odds.csv"
REPORTS = ROOT / "reports"


def american_to_prob(ml: float) -> float:
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def american_profit(ml: float, stake: float = 1.0) -> float:
    """Profit on a WON bet of `stake` units."""
    return stake * (ml / 100.0) if ml > 0 else stake * (100.0 / -ml)


def kelly_fraction(p: float, ml: float) -> float:
    b = (ml / 100.0) if ml > 0 else (100.0 / -ml)
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, f)


def breakeven_analysis(oof: pd.DataFrame) -> dict:
    """What does the model need from the market to be profitable?

    For each confidence bucket, report realized win rate and the break-even
    American odds. If the market routinely offers WORSE than break-even, the
    model cannot profit -- regardless of accuracy.
    """
    d = oof.copy()
    d["conf"] = np.where(d.p_home >= 0.5, d.p_home, 1 - d.p_home)
    d["pick_home"] = (d.p_home >= 0.5).astype(int)
    d["correct"] = np.where(d.pick_home == 1, d.home_win, 1 - d.home_win)

    buckets = [(0.50, 0.53), (0.53, 0.56), (0.56, 0.60), (0.60, 0.65), (0.65, 1.01)]
    rows = []
    for lo, hi in buckets:
        m = d[(d.conf >= lo) & (d.conf < hi)]
        if len(m) < 50:
            continue
        wr = m.correct.mean()
        # break-even decimal odds = 1/win_rate  -> convert to American
        be_dec = 1.0 / wr if wr > 0 else np.inf
        be_am = (be_dec - 1) * 100 if be_dec >= 2 else -100 / (be_dec - 1)
        rows.append({
            "bucket": f"{lo:.2f}-{hi:.2f}",
            "n": int(len(m)),
            "predicted": float(m.conf.mean()),
            "actual": float(wr),
            "calib_error": float(m.conf.mean() - wr),
            "breakeven_american": float(be_am),
        })
    return {"buckets": rows}


def calibration_table(oof: pd.DataFrame, bins: int = 10) -> list[dict]:
    d = oof.copy()
    d["bin"] = pd.cut(d.p_home, np.linspace(0, 1, bins + 1), include_lowest=True)
    out = []
    for b, g in d.groupby("bin", observed=True):
        if len(g) < 30:
            continue
        out.append({
            "bin": str(b),
            "n": int(len(g)),
            "mean_pred": float(g.p_home.mean()),
            "actual": float(g.home_win.mean()),
            "error": float(g.p_home.mean() - g.home_win.mean()),
        })
    return out


def load_odds() -> pd.DataFrame | None:
    if not ODDS.exists():
        return None
    df = pd.read_csv(ODDS)
    need = {"date", "away", "home", "ml_home", "ml_away"}
    if not need.issubset(df.columns):
        print(f"[roi] odds.csv missing columns: {need - set(df.columns)}")
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df


def evaluate_with_odds(oof: pd.DataFrame, odds: pd.DataFrame,
                       edge_min: float = 0.02, kelly_mult: float = 0.25,
                       bankroll: float = 1000.0) -> dict:
    d = oof.merge(odds, on=["date", "home", "away"], how="inner")
    if d.empty:
        return {"error": "no games matched between predictions and odds"}

    d["imp_home"] = d.ml_home.apply(american_to_prob)
    d["imp_away"] = d.ml_away.apply(american_to_prob)
    d["vig"] = d.imp_home + d.imp_away - 1
    # de-vigged fair market probability
    d["fair_home"] = d.imp_home / (d.imp_home + d.imp_away)

    d["edge_home"] = d.p_home - d.imp_home
    d["edge_away"] = (1 - d.p_home) - d.imp_away

    bets, bank, curve = [], bankroll, []
    for _, r in d.sort_values("date").iterrows():
        side = None
        if r.edge_home >= edge_min and r.edge_home >= r.edge_away:
            side, p, ml, won = "home", r.p_home, r.ml_home, r.home_win == 1
        elif r.edge_away >= edge_min:
            side, p, ml, won = "away", 1 - r.p_home, r.ml_away, r.home_win == 0
        if side is None:
            continue
        f = kelly_fraction(p, ml) * kelly_mult
        stake = min(bank * f, bank * 0.05)
        if stake <= 0:
            continue
        pnl = american_profit(ml, stake) if won else -stake
        bank += pnl
        bets.append({"date": str(r.date.date()), "side": side, "ml": float(ml),
                     "p": float(p), "stake": float(stake), "pnl": float(pnl),
                     "won": bool(won), "bank": float(bank)})
        curve.append(bank)

    if not bets:
        return {"error": f"no bets cleared edge threshold {edge_min}"}

    b = pd.DataFrame(bets)
    staked = b.stake.sum()
    profit = b.pnl.sum()
    peak = np.maximum.accumulate(curve)
    dd = float(((peak - curve) / peak).max()) if len(curve) else 0.0

    return {
        "n_games_matched": int(len(d)),
        "n_bets": int(len(b)),
        "bet_rate": float(len(b) / len(d)),
        "win_rate": float(b.won.mean()),
        "total_staked": float(staked),
        "profit": float(profit),
        "roi_pct": float(100 * profit / staked),
        "final_bankroll": float(bank),
        "max_drawdown_pct": float(100 * dd),
        "avg_vig": float(d.vig.mean()),
        "clv_proxy": float((d.p_home - d.fair_home).abs().mean()),
        "params": {"edge_min": edge_min, "kelly_mult": kelly_mult},
    }


def main() -> None:
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "gbm_cal"
    path = PROC / f"oof_{kind}.parquet"
    if not path.exists():
        print(f"[roi] {path} not found -- run backtest.py first")
        return
    oof = pd.read_parquet(path)

    print("=" * 66)
    print(f"ROI EVALUATION  (model: {kind}, {len(oof):,} out-of-sample games)")
    print("=" * 66)

    cal = calibration_table(oof)
    print("\nCALIBRATION (does 60% mean 60%?)")
    print(f"{'bin':>14} {'n':>6} {'pred':>8} {'actual':>8} {'error':>8}")
    for c in cal:
        print(f"{c['bin']:>14} {c['n']:>6} {c['mean_pred']:>8.4f} "
              f"{c['actual']:>8.4f} {c['error']:>+8.4f}")
    mce = max(abs(c["error"]) for c in cal) if cal else float("nan")
    print(f"  max calibration error: {mce:.4f}")

    be = breakeven_analysis(oof)
    print("\nBREAK-EVEN REQUIREMENTS (what price you must beat)")
    print(f"{'confidence':>12} {'n':>6} {'pred':>7} {'actual':>7} {'need better than':>18}")
    for b in be["buckets"]:
        print(f"{b['bucket']:>12} {b['n']:>6} {b['predicted']:>7.4f} "
              f"{b['actual']:>7.4f} {b['breakeven_american']:>+18.1f}")

    odds = load_odds()
    report = {"model": kind, "n_oof": len(oof), "calibration": cal,
              "breakeven": be, "max_calib_error": float(mce)}

    if odds is None:
        print("\n" + "!" * 66)
        print("NO ODDS FILE -> ROI NOT COMPUTED (this is intentional).")
        print("Any ROI figure without real historical prices would be fabricated.")
        print(f"Add {ODDS.relative_to(ROOT)} with columns:")
        print("    date,away,home,ml_home,ml_away")
        print("then re-run. Full ROI, Kelly staking and drawdown activate on load.")
        print("!" * 66)
        report["roi"] = None
        report["roi_status"] = "unavailable: no historical odds supplied"
    else:
        print(f"\nOdds file loaded: {len(odds):,} rows")
        best = None
        for edge in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
            r = evaluate_with_odds(oof, odds, edge_min=edge)
            if "error" in r:
                print(f"  edge>={edge:.2f}: {r['error']}")
                continue
            print(f"  edge>={edge:.2f}: bets={r['n_bets']:>5} "
                  f"win={r['win_rate']:.4f} ROI={r['roi_pct']:+.2f}% "
                  f"maxDD={r['max_drawdown_pct']:.1f}%")
            if best is None or r["roi_pct"] > best["roi_pct"]:
                best = r
        report["roi"] = best
        report["roi_status"] = "computed from supplied odds"
        if best:
            print(f"\nBEST: ROI {best['roi_pct']:+.2f}% on {best['n_bets']} bets "
                  f"(edge>={best['params']['edge_min']})")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"roi_{kind}.json").write_text(json.dumps(report, indent=2))
    print(f"\n[roi] -> reports/roi_{kind}.json")


if __name__ == "__main__":
    main()
