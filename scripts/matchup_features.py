"""Combine pitcher arsenals with lineup batter splits into matchup scores.

    matchup = sum over lineup slots  order_weight[slot]
              * sum over pitch families  arsenal_share[p] * batter_xwoba[slot, p, hand]

Everything on both sides is prior-games-only. The batter values are already
empirical-Bayes shrunk (see arsenal.py), which matters a great deal: the median
batter-family-hand cell has only ~15 batted balls of history.

Also produces a DISPERSION term. A starter who neutralises seven hitters but is
punished by two is a materially different proposition from one who is uniformly
average against the whole lineup, even at the same mean.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
LINEUP_RAW = ROOT / "data" / "raw" / "lineups"

FAMILIES = ["FF", "SI", "FC", "SL", "CU", "CH", "KN"]
ORDER_WEIGHTS = np.array([1.12, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94, 0.91, 0.88])

MATCHUP_COLS = ["mu_xwoba", "mu_disp", "mu_worst", "mu_best", "mu_n_known"]


def load_inputs():
    ars = pd.read_parquet(PROC / "pitcher_arsenal.parquet")
    spl = pd.read_parquet(PROC / "batter_pitch_splits.parquet")
    ars["game_date"] = pd.to_datetime(ars["game_date"])
    spl["date"] = pd.to_datetime(spl["date"])
    return ars, spl


def build() -> pd.DataFrame:
    ars, spl = load_inputs()

    # Arsenal lookup: (pitcher, date) -> share vector
    acols = [f"arr_{f}" for f in FAMILIES]
    ars = ars.dropna(subset=["arr_FF"])
    a_lut = {}
    for r in ars.itertuples(index=False):
        a_lut[(r.pitcher, r.game_date)] = np.array(
            [getattr(r, f"arr_{f}") for f in FAMILIES], dtype=float)
    print(f"[matchup] arsenal records: {len(a_lut):,}")

    # Batter split lookup: (batter, date, hand) -> xwoba per family
    spl = spl[spl["bx_xwoba"].notna()]
    s_lut: dict[tuple, dict[str, float]] = {}
    for r in spl.itertuples(index=False):
        s_lut.setdefault((r.batter, r.date, r.p_throws), {})[r.fam] = r.bx_xwoba
    print(f"[matchup] batter-split records: {len(s_lut):,}")

    # Pitcher handedness, from the split table itself
    hand_rows = spl.groupby("p_throws").size()
    print(f"[matchup] hand distribution: {hand_rows.to_dict()}")

    # league fallback per (fam, hand, season)
    lg = (spl.groupby([spl["date"].dt.year, "fam", "p_throws"])["lg_xwoba"]
            .mean().to_dict())

    # Pitcher handedness. Platoon is one of the largest and best-sampled
    # splits in baseball, so using the WRONG hand would corrupt every matchup
    # score. Built from the pitch data itself (modal p_throws per pitcher).
    hpath = PROC / "pitcher_hand.parquet"
    hand_map: dict[int, str] = {}
    if hpath.exists():
        hm = pd.read_parquet(hpath)
        hand_map = {int(i): v for i, v in hm["p_throws"].items()}
    print(f"[matchup] pitcher handedness known for {len(hand_map):,} pitchers")

    rows = []
    for f in sorted(LINEUP_RAW.glob("*.json")):
        try:
            games = json.loads(Path(f).read_text())
        except Exception:
            continue
        for g in games:
            day = pd.Timestamp(g["date"])
            rec = {"game_pk": g["game_pk"], "date": day,
                   "home_abbr": g["home"], "away_abbr": g["away"]}

            # home lineup faces the AWAY starter, and vice versa
            for side, lineup_key, sp_key in (("h", "home_lineup", "away_sp"),
                                             ("a", "away_lineup", "home_sp")):
                pid = g.get(sp_key)
                share = a_lut.get((pid, day)) if pid else None
                for c in MATCHUP_COLS:
                    rec[f"{side}_{c}"] = np.nan
                if share is None:
                    continue

                hand = hand_map.get(int(pid)) if pid else None
                if hand is None:
                    continue
                vals, wts = [], []
                for i, bid in enumerate(g[lineup_key][:9]):
                    fam_map = s_lut.get((bid, day, hand))
                    if not fam_map:
                        continue
                    yr = day.year
                    num = 0.0
                    for j, fam in enumerate(FAMILIES):
                        w = share[j]
                        if not np.isfinite(w) or w <= 0:
                            continue
                        v = fam_map.get(fam)
                        if v is None:
                            v = lg.get((yr, fam, hand))
                        if v is None:
                            continue
                        num += w * v
                    if num > 0:
                        vals.append(num)
                        wts.append(ORDER_WEIGHTS[i] if i < 9 else 1.0)

                rec[f"{side}_mu_n_known"] = len(vals)
                if len(vals) < 4:
                    continue
                w = np.array(wts); w = w / w.sum()
                v = np.array(vals)
                rec[f"{side}_mu_xwoba"] = float(np.average(v, weights=w))
                rec[f"{side}_mu_disp"] = float(v.std())
                rec[f"{side}_mu_worst"] = float(v.min())
                rec[f"{side}_mu_best"] = float(v.max())
            rows.append(rec)

    out = pd.DataFrame(rows)
    for c in MATCHUP_COLS:
        if f"h_{c}" in out and f"a_{c}" in out:
            out[f"d_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "matchup_features.parquet", index=False)
    cov = out["h_mu_xwoba"].notna().mean()
    print(f"[matchup] {len(out):,} games, {cov:.1%} with matchup score")
    return out


def main() -> None:
    out = build()
    if out.empty:
        return
    o = out.dropna(subset=["h_mu_xwoba"])
    print("\nMatchup score distributions:")
    for c in ("h_mu_xwoba", "h_mu_disp", "d_mu_xwoba"):
        if c in o:
            s = o[c].dropna()
            print(f"  {c:<14} mean {s.mean():>8.4f}  sd {s.std():>7.4f}  "
                  f"[{s.min():.4f}, {s.max():.4f}]")
    print("\nCoverage by season:")
    print(o.groupby(o.date.dt.year).size().to_string())


if __name__ == "__main__":
    main()
