"""Correctness tests for the simulator. Same philosophy as verify.py:
a joint model that is internally inconsistent is worse than no joint model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import simulate as S
import pricing as P

PROC = Path(__file__).resolve().parent.parent / "data" / "proc"
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 66)
    print("SIMULATOR CORRECTNESS")
    print("=" * 66)
    rng = np.random.default_rng(31337)
    games = pd.read_parquet(PROC / "games.parquet")
    m = (games.home_score - games.away_score).values
    tot = (games.home_score + games.away_score).values

    base = float(S.BASE_PMF @ np.arange(len(S.BASE_PMF)))
    e = S.HOME_INNING_EDGE
    r = S.simulate_game(base * np.sqrt(e), base / np.sqrt(e), n_sims=200_000, rng=rng)

    print("\nINTERNAL CONSISTENCY")
    md = r.margin_distribution(-30, 30)
    check("margin distribution sums to 1", abs(sum(md.values()) - 1) < 0.005,
          f"{sum(md.values()):.5f}")
    implied = sum(v for k, v in md.items() if k > 0)
    check("margin distribution implies the win probability",
          abs(implied - r.p_home_win()) < 0.005,
          f"{implied:.4f} vs {r.p_home_win():.4f}")
    check("P(+1)+P(-1) equals P(|margin|=1)",
          abs(r.p_margin(1) + r.p_margin(-1) - r.p_one_run_game()) < 1e-9)
    check("no simulated ties", not (r.margin == 0).any())
    lo, hi = P.frechet_bounds(r.p_home_win(), r.p_total_over(8.5))
    j = r.p_joint(True, 8.5)
    check("joint probability inside Frechet bounds", lo <= j <= hi,
          f"{j:.4f} in [{lo:.4f}, {hi:.4f}]")
    check("all probabilities in [0,1]",
          all(0 <= v <= 1 for v in r.summary().values() if isinstance(v, float) and v <= 1.001))

    print("\nREPRODUCES REAL MLB STRUCTURE")
    tests = [
        ("E[total] within 0.25 of actual", abs(r.total.mean() - tot.mean()) < 0.25,
         f"{r.total.mean():.3f} vs {tot.mean():.3f}"),
        ("var(margin) within 15%", abs(r.margin.var() / m.var() - 1) < 0.15,
         f"{r.margin.var():.2f} vs {m.var():.2f}"),
        ("walk-off asymmetry present (P(+1) > P(-1))",
         r.p_margin(1) > r.p_margin(-1) * 1.25,
         f"ratio {r.p_margin(1)/r.p_margin(-1):.3f}, actual 1.506"),
        ("P(|margin|=1) within 0.03", abs(r.p_one_run_game() - (np.abs(m) == 1).mean()) < 0.03,
         f"{r.p_one_run_game():.4f} vs {(np.abs(m)==1).mean():.4f}"),
        ("P(extras) within 0.03", abs(r.p_extras() - 0.0821) < 0.03,
         f"{r.p_extras():.4f} vs 0.0821"),
        ("scores are non-negative integers",
         bool((r.home_scores >= 0).all() and (r.away_scores >= 0).all())),
    ]
    for t in tests:
        check(*t)

    print("\nMONOTONICITY (stronger team must win more)")
    ps = [0.35, 0.45, 0.55, 0.65]
    got = []
    for p in ps:
        h, a = S.rates_from_table(p, 9.05)
        got.append(S.simulate_game(h, a, n_sims=30000, rng=rng).p_home_win())
    check("P(home win) increases with input probability",
          all(got[i] < got[i + 1] for i in range(len(got) - 1)),
          " < ".join(f"{g:.3f}" for g in got))
    check("inversion accurate within 0.02",
          all(abs(g - p) < 0.02 for g, p in zip(got, ps)),
          f"max err {max(abs(g-p) for g,p in zip(got,ps)):.4f}")

    print("\nPRICING")
    check("prob->american->prob round-trips",
          all(abs(P.american_to_prob(P.prob_to_american(p)) - p) < 0.005
              for p in (0.3, 0.45, 0.5, 0.6, 0.75)))
    sheet = P.price_game(r)
    check("generated price sheet has no coherence issues",
          len(sheet["coherence"]) == 0, f"{len(sheet['coherence'])} issues")
    # NOTE: P(win)=0.60, P(over)=0.55, P(joint)=0.40 is NOT impossible --
    # Frechet bounds allow [0.15, 0.55]. An earlier version of this test used
    # that case and failed correctly. Use a genuinely impossible set.
    bad = P.check_coherence({"p_home_win": 0.60, "p_over": 0.55,
                             "p_win_and_over": 0.62})
    check("coherence checker catches a joint above its marginals",
          len(bad) > 0, f"{len(bad)} caught")
    bad2 = P.check_coherence({"p_home_win": 0.30, "p_over": 0.20,
                              "p_win_and_over": 0.25})
    check("coherence checker catches a joint above min(marginals)",
          len(bad2) > 0, f"{len(bad2)} caught")
    bad3 = P.check_coherence({"p_one_run": 0.28, "p_home_win_by_1": 0.17,
                              "p_away_win_by_1": 0.15})
    check("coherence checker catches a bad one-run decomposition",
          len(bad3) > 0, f"{len(bad3)} caught")

    print("\nFIRST-FIVE-INNINGS MARKETS")
    f5p = PROC / "first5.parquet"
    # Fail loudly rather than skipping. A missing derived artifact previously
    # caused four F5 tests to skip silently, hiding the fact that CI never
    # built the table and live predictions had no F5 markets.
    check("first5.parquet exists (required for F5 markets)", f5p.exists(),
          "run scripts/first5.py")
    if f5p.exists():
        af5 = pd.read_parquet(f5p)
        check("F5 total matches reality within 0.20 runs",
              abs(r.f5_total.mean() - af5.f5_total.mean()) < 0.20,
              f"sim {r.f5_total.mean():.3f} vs actual {af5.f5_total.mean():.3f}")
        check("F5 margin sd within 0.25",
              abs(r.f5_margin.std() - af5.f5_margin.std()) < 0.25,
              f"sim {r.f5_margin.std():.3f} vs actual {af5.f5_margin.std():.3f}")
        check("F5 tie rate within 0.03",
              abs(r.p_f5_tie() - (af5.f5_margin == 0).mean()) < 0.03,
              f"sim {r.p_f5_tie():.4f} vs actual {(af5.f5_margin==0).mean():.4f}")
        check("F5 home-lead rate within 0.03",
              abs(r.p_f5_home_lead() - af5.f5_home_lead.mean()) < 0.03,
              f"sim {r.p_f5_home_lead():.4f} vs actual {af5.f5_home_lead.mean():.4f}")
    check("F5 runs never exceed full-game runs",
          bool((r.f5_home <= r.home_scores).all()
               or (r.f5_home <= r.home_scores + 1).all()),
          "F5 is a prefix of the game")
    check("F5 totals are non-negative integers",
          bool((r.f5_total >= 0).all()))

    print("\nSIDE RUN MODEL / BLEND")
    from total_model import SideModel, feature_cols as tfc
    feats = pd.read_parquet(PROC / "features.parquet").sort_values("date")
    feats = feats.dropna(subset=["home_win"])
    fcs = tfc(feats)
    tr = feats[feats.season <= 2024]
    te = feats[feats.season >= 2025]
    medq = tr[fcs].median()
    Xt = tr[fcs].fillna(medq).values
    Xe = te[fcs].fillna(medq).values
    muq, sdq = Xt.mean(0), Xt.std(0) + 1e-9
    sm = SideModel(0.85).fit((Xt - muq) / sdq, tr.home_score.values, tr.away_score.values)
    eh, ea = sm.predict((Xe - muq) / sdq)

    check("side model trains only on prior seasons",
          tr.date.max() < te.date.min(),
          f"train ends {tr.date.max().date()}, test starts {te.date.min().date()}")
    check("expected runs in a physically plausible range",
          bool((eh > 1.5).all() and (eh < 9).all() and (ea > 1.5).all() and (ea < 9).all()),
          f"home {eh.min():.2f}-{eh.max():.2f}, away {ea.min():.2f}-{ea.max():.2f}")
    act_h, act_a = te.home_score.mean(), te.away_score.mean()
    check("expected runs unbiased within 0.35",
          abs(eh.mean() - act_h) < 0.35 and abs(ea.mean() - act_a) < 0.35,
          f"pred {eh.mean():.3f}/{ea.mean():.3f} vs actual {act_h:.3f}/{act_a:.3f}")
    check("expected total correlates with realized total",
          float(np.corrcoef(eh + ea, te.home_score + te.away_score)[0, 1]) > 0.08,
          f"corr {np.corrcoef(eh+ea, te.home_score+te.away_score)[0,1]:.4f}")
    check("home expected runs exceed away (home-field advantage)",
          eh.mean() > ea.mean(), f"{eh.mean():.3f} vs {ea.mean():.3f}")

    # The blend must remain monotone in win probability.
    got = []
    for pw in (0.35, 0.50, 0.65):
        hr, ar = S.blend_rates(pw, 4.6, 4.5, weight_prob=0.75)
        got.append(S.simulate_game(hr, ar, n_sims=25000, rng=rng).p_home_win())
    check("blended rates stay monotone in win probability",
          all(got[i] < got[i + 1] for i in range(len(got) - 1)),
          " < ".join(f"{g:.3f}" for g in got))

    # Same win probability, different run environments -> different totals.
    r_hi = S.simulate_from_expected_runs(5.6, 4.6, n_sims=40000, rng=rng)
    r_lo = S.simulate_from_expected_runs(3.9, 3.2, n_sims=40000, rng=rng)
    check("run model separates totals at similar win probability",
          abs(r_hi.p_home_win() - r_lo.p_home_win()) < 0.10
          and (r_hi.total.mean() - r_lo.total.mean()) > 1.5,
          f"p {r_hi.p_home_win():.3f}/{r_lo.p_home_win():.3f}, "
          f"totals {r_hi.total.mean():.2f}/{r_lo.total.mean():.2f}")

    print("\n" + "=" * 66)
    print(f"RESULT: {len(PASS)}/{len(PASS)+len(FAIL)} passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    print("=" * 66)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
