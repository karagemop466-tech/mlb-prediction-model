"""Convert the simulated joint distribution into prices, and check coherence.

Two distinct jobs:

1. PRICING — turn probabilities into fair odds (American / decimal), and show
   what a book would quote after applying a vig. This is just arithmetic on the
   simulator's output.

2. COHERENCE — verify the price set is internally consistent. This is the part
   that actually needs the joint distribution. Independent per-market models can
   quote P(home win)=0.60, P(over)=0.55 and P(win AND over)=0.40, which is
   impossible: the conjunction cannot exceed either marginal, and Frechet bounds
   pin it to [0.15, 0.55]. A simulator cannot violate these by construction, so
   it can be used to audit quote sets that can.

NO ROI IS CLAIMED. This module prices a distribution and checks arithmetic
consistency. It does not assert an edge against any real market. Consistent with
every prior finding in this project, no profitability claim is made anywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- odds math
def prob_to_american(p: float) -> int:
    """Fair American odds for probability p (no vig)."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def prob_to_decimal(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return round(1.0 / p, 3)


def american_to_prob(ml: float) -> float:
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def apply_vig(p: float, vig: float = 0.045) -> tuple[int, int]:
    """Two-way market with `vig` overround, returned as American odds."""
    q = 1 - p
    scale = (1 + vig) / (p + q)
    return prob_to_american(p * scale), prob_to_american(q * scale)


def devig_two_way(ml_a: float, ml_b: float) -> tuple[float, float]:
    """Remove the overround from a two-way quote (proportional method)."""
    pa, pb = american_to_prob(ml_a), american_to_prob(ml_b)
    tot = pa + pb
    return pa / tot, pb / tot


# ---------------------------------------------------------------- coherence
@dataclass
class CoherenceIssue:
    rule: str
    detail: str
    severity: str


def frechet_bounds(p_a: float, p_b: float) -> tuple[float, float]:
    """Tightest possible bounds on P(A and B) given only the marginals."""
    return max(0.0, p_a + p_b - 1.0), min(p_a, p_b)


def check_coherence(quotes: dict) -> list[CoherenceIssue]:
    """Audit a set of market probabilities for internal contradictions.

    `quotes` may contain: p_home_win, p_over, p_win_and_over, p_margin_dist,
    p_one_run, p_home_win_by_1, p_away_win_by_1.
    """
    issues: list[CoherenceIssue] = []

    for k, v in quotes.items():
        if isinstance(v, (int, float)) and not (0.0 <= v <= 1.0):
            issues.append(CoherenceIssue(
                "probability range", f"{k}={v:.4f} outside [0,1]", "fatal"))

    pw, po = quotes.get("p_home_win"), quotes.get("p_over")
    pwo = quotes.get("p_win_and_over")
    if None not in (pw, po, pwo):
        lo, hi = frechet_bounds(pw, po)
        if not (lo - 1e-9 <= pwo <= hi + 1e-9):
            issues.append(CoherenceIssue(
                "Frechet bounds",
                f"P(win AND over)={pwo:.4f} outside [{lo:.4f}, {hi:.4f}] "
                f"implied by P(win)={pw:.4f}, P(over)={po:.4f}",
                "fatal"))

    md = quotes.get("p_margin_dist")
    if md:
        tot = sum(md.values())
        if abs(tot - 1.0) > 0.02:
            issues.append(CoherenceIssue(
                "margin distribution sums to 1",
                f"sums to {tot:.4f}", "warning"))
        if pw is not None:
            implied = sum(v for k, v in md.items() if int(k) > 0)
            if abs(implied - pw) > 0.02:
                issues.append(CoherenceIssue(
                    "margin/moneyline consistency",
                    f"margin distribution implies P(win)={implied:.4f} "
                    f"but moneyline says {pw:.4f}", "fatal"))

    p1 = quotes.get("p_one_run")
    ph1, pa1 = quotes.get("p_home_win_by_1"), quotes.get("p_away_win_by_1")
    if None not in (p1, ph1, pa1):
        if abs((ph1 + pa1) - p1) > 0.015:
            issues.append(CoherenceIssue(
                "one-run decomposition",
                f"P(|margin|=1)={p1:.4f} but P(+1)+P(-1)={ph1+pa1:.4f}",
                "fatal"))

    return issues


# ---------------------------------------------------------------- price sheet
def price_game(sim, lines: tuple[float, ...] = (7.5, 8.5, 9.5, 10.5),
               vig: float = 0.045) -> dict:
    """Full price sheet for one simulated game."""
    p_win = sim.p_home_win()
    sheet = {
        "moneyline": {
            "home": {"prob": p_win, "fair": prob_to_american(p_win),
                     "fair_decimal": prob_to_decimal(p_win)},
            "away": {"prob": 1 - p_win, "fair": prob_to_american(1 - p_win),
                     "fair_decimal": prob_to_decimal(1 - p_win)},
            "with_vig": dict(zip(("home", "away"), apply_vig(p_win, vig))),
        },
        "totals": {},
        "margins": {},
        "derivatives": {},
        "correlated": {},
    }

    for ln in lines:
        po = sim.p_total_over(ln)
        sheet["totals"][str(ln)] = {
            "over_prob": po, "over_fair": prob_to_american(po),
            "under_prob": 1 - po, "under_fair": prob_to_american(1 - po),
            "with_vig": dict(zip(("over", "under"), apply_vig(po, vig))),
        }

    for k in range(-6, 7):
        if k == 0:
            continue
        p = sim.p_margin(k)
        if p > 0.002:
            sheet["margins"][str(k)] = {"prob": p, "fair": prob_to_american(p)}

    for name, p in (
        ("one_run_game", sim.p_one_run_game()),
        ("extra_innings", sim.p_extras()),
        ("both_teams_score", sim.p_both_score()),
        ("shutout", sim.p_shutout()),
    ):
        sheet["derivatives"][name] = {"prob": p, "fair": prob_to_american(p)}

    # The markets that only a joint model can price.
    for ln in (8.5, 9.5):
        ph = sim.p_joint(True, ln)
        pa = sim.p_joint(False, ln)
        indep = p_win * sim.p_total_over(ln)
        sheet["correlated"][f"home_win_and_over_{ln}"] = {
            "prob": ph, "fair": prob_to_american(ph),
            "independent_estimate": indep,
            "correlation_effect": ph - indep,
        }
        sheet["correlated"][f"away_win_and_over_{ln}"] = {
            "prob": pa, "fair": prob_to_american(pa),
        }

    sheet["coherence"] = [
        {"rule": i.rule, "detail": i.detail, "severity": i.severity}
        for i in check_coherence({
            "p_home_win": p_win,
            "p_over": sim.p_total_over(8.5),
            "p_win_and_over": sim.p_joint(True, 8.5),
            "p_margin_dist": {str(k): sim.p_margin(k) for k in range(-15, 16)},
            "p_one_run": sim.p_one_run_game(),
            "p_home_win_by_1": sim.p_home_win_by(1),
            "p_away_win_by_1": sim.p_home_win_by(-1),
        })
    ]
    return sheet


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import simulate as S

    print("Price sheet for a 57% home favorite, expected total 9.0\n")
    sim = S.simulate_from_probability(0.57, 9.0, n_sims=120_000,
                                      rng=np.random.default_rng(1))
    sheet = price_game(sim)

    ml = sheet["moneyline"]
    print(f"MONEYLINE   home {ml['home']['fair']:+5d} (fair)   "
          f"{ml['with_vig']['home']:+5d} (with 4.5% vig)")
    print(f"            away {ml['away']['fair']:+5d} (fair)   "
          f"{ml['with_vig']['away']:+5d} (with 4.5% vig)")

    print("\nTOTALS")
    for ln, d in sheet["totals"].items():
        print(f"  {ln:>5}  over {d['over_prob']:.4f} ({d['over_fair']:+5d})   "
              f"under {d['under_prob']:.4f} ({d['under_fair']:+5d})")

    print("\nMARGIN (how they win)")
    for k in ("1", "2", "3", "-1", "-2", "-3"):
        if k in sheet["margins"]:
            d = sheet["margins"][k]
            print(f"  by {k:>3}: {d['prob']:.4f}  ({d['fair']:+6d})")

    print("\nDERIVATIVES")
    for k, d in sheet["derivatives"].items():
        print(f"  {k:<18} {d['prob']:.4f}  ({d['fair']:+6d})")

    print("\nCORRELATED (requires the joint distribution)")
    for k, d in sheet["correlated"].items():
        extra = ""
        if "independent_estimate" in d:
            extra = (f"   independent would say {d['independent_estimate']:.4f}"
                     f"  (off by {d['correlation_effect']:+.4f})")
        print(f"  {k:<26} {d['prob']:.4f} ({d['fair']:+6d}){extra}")

    print(f"\nCOHERENCE: {len(sheet['coherence'])} issues")
    for i in sheet["coherence"]:
        print(f"  [{i['severity']}] {i['rule']}: {i['detail']}")

    print("\n--- Auditing an INCOHERENT quote set (what independent models produce) ---")
    bad = check_coherence({
        "p_home_win": 0.60, "p_over": 0.55, "p_win_and_over": 0.40,
        "p_one_run": 0.28, "p_home_win_by_1": 0.17, "p_away_win_by_1": 0.15,
    })
    for i in bad:
        print(f"  [{i.severity}] {i.rule}\n      {i.detail}")
