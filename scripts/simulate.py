"""Inning-level Monte Carlo simulator for correlated MLB outcomes.

See DESIGN_SIMULATION.md for why this is inning-level rather than a copula on
final scores. Short version: home/away runs are uncorrelated (r=0.005), runs are
overdispersed 2.2x (Poisson is wrong), and the walk-off rule creates a +1/-1
margin asymmetry (16.69% vs 11.08%) that no final-score model can reproduce.

The simulator samples half-innings from the empirical distribution, exponentially
tilted to match each team's expected scoring rate, and applies real stopping
rules. Every derived market (winner, margin, total, extras, conjunctions) then
comes from one coherent joint distribution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"

# Empirical half-inning run distribution from INNINGS 1-8 ONLY (25,136 half
# innings). Innings 9+ are excluded because the bottom of the 9th is skipped
# when the home team leads and truncated on a walk-off; including it would bake
# the stopping rule into the base distribution and double-count it below.
# Regenerate with scripts/calibrate_innings.py.
_DIST_PATH = PROC / "inning_dist.json"
if _DIST_PATH.exists():
    _d = json.loads(_DIST_PATH.read_text())
    BASE_PMF = np.array(_d["pmf"], dtype=float)
    HOME_INNING_EDGE = float(_d["home_inning_advantage"])
else:
    BASE_PMF = np.array([
        0.72987, 0.14338, 0.06839, 0.03187, 0.01496,
        0.00641, 0.00290, 0.00127, 0.00056, 0.00016, 0.00020,
    ])
    HOME_INNING_EDGE = 1.0756
BASE_PMF = BASE_PMF / BASE_PMF.sum()

# Fitted parameters (scripts/fit_simulator.py) override the defaults below.
_PARAM_PATH = PROC / "sim_params.json"
_FITTED = json.loads(_PARAM_PATH.read_text()) if _PARAM_PATH.exists() else {}
MAX_RUNS = len(BASE_PMF) - 1
_K = np.arange(len(BASE_PMF))

# Extra innings use the automatic-runner-on-second rule, which raises scoring.
# NOTE: the raw observed extra-inning mean (2.19x) is badly biased upward --
# extras only CONTINUE while tied, so any high-scoring half-inning ends the game
# and never generates another observation. This value is fit so the simulator
# reproduces the observed 8.21% extra-inning rate. See fit_extra_mult().
EXTRA_INNING_MULT = _FITTED.get("EXTRA_INNING_MULT", 1.30)
MAX_EXTRA_INNINGS = 12


def tilted_pmf(target_mean: float, base: np.ndarray = BASE_PMF) -> np.ndarray:
    """Exponentially tilt the empirical PMF to hit `target_mean`.

    p_tilted(k) ∝ p_base(k) * exp(theta * k)

    This is the maximum-entropy way to shift a distribution's mean: it makes the
    smallest possible distortion to the observed shape, preserving zero-inflation
    and the heavy tail. theta=0 returns the empirical distribution unchanged.
    """
    base_mean = float(base @ _K)
    if target_mean <= 0.01:
        out = np.zeros_like(base)
        out[0] = 1.0
        return out
    if abs(target_mean - base_mean) < 1e-9:
        return base.copy()

    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        w = base * np.exp(mid * _K)
        m = float((w @ _K) / w.sum())
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    w = base * np.exp(((lo + hi) / 2) * _K)
    return w / w.sum()


def _sampler(pmf: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorized draws from a discrete PMF via inverse-CDF."""
    cdf = np.cumsum(pmf)
    u = rng.random(n)
    return np.searchsorted(cdf, u).astype(np.int16)


@dataclass
class SimResult:
    """Joint distribution over one game's outcomes."""
    home_scores: np.ndarray
    away_scores: np.ndarray
    went_extras: np.ndarray
    n_sims: int = field(init=False)

    def __post_init__(self):
        self.n_sims = len(self.home_scores)

    # --- primary markets -------------------------------------------------
    @property
    def margin(self) -> np.ndarray:
        return self.home_scores - self.away_scores

    @property
    def total(self) -> np.ndarray:
        return self.home_scores + self.away_scores

    def p_home_win(self) -> float:
        return float((self.margin > 0).mean())

    def p_margin(self, k: int) -> float:
        return float((self.margin == k).mean())

    def p_home_win_by(self, k: int) -> float:
        return float((self.margin == k).mean())

    def p_total_over(self, line: float) -> float:
        return float((self.total > line).mean())

    def p_extras(self) -> float:
        return float(self.went_extras.mean())

    def p_shutout(self) -> float:
        return float(((self.home_scores == 0) | (self.away_scores == 0)).mean())

    def p_both_score(self) -> float:
        return float(((self.home_scores > 0) & (self.away_scores > 0)).mean())

    # --- correlated / conjunctive markets --------------------------------
    def p_joint(self, home_wins: bool, over_line: float) -> float:
        """P(winner AND total over line) — the correlated question."""
        w = (self.margin > 0) if home_wins else (self.margin < 0)
        return float((w & (self.total > over_line)).mean())

    def p_conditional_win_given_extras(self) -> float:
        m = self.went_extras
        return float((self.margin[m] > 0).mean()) if m.any() else float("nan")

    def p_one_run_game(self) -> float:
        return float((np.abs(self.margin) == 1).mean())

    def margin_distribution(self, lo: int = -10, hi: int = 10) -> dict[int, float]:
        return {k: float((self.margin == k).mean()) for k in range(lo, hi + 1)}

    def total_distribution(self, lo: int = 0, hi: int = 25) -> dict[int, float]:
        return {k: float((self.total == k).mean()) for k in range(lo, hi + 1)}

    def summary(self) -> dict:
        return {
            "n_sims": self.n_sims,
            "p_home_win": self.p_home_win(),
            "exp_home_score": float(self.home_scores.mean()),
            "exp_away_score": float(self.away_scores.mean()),
            "exp_total": float(self.total.mean()),
            "p_over_8_5": self.p_total_over(8.5),
            "p_extras": self.p_extras(),
            "p_one_run": self.p_one_run_game(),
            "p_home_win_by_1": self.p_home_win_by(1),
            "p_away_win_by_1": self.p_home_win_by(-1),
            "p_both_score": self.p_both_score(),
            "p_home_win_and_over_8_5": self.p_joint(True, 8.5),
            "p_away_win_and_over_8_5": self.p_joint(False, 8.5),
        }


# Game-level scoring environment dispersion.
#
# Measured: var(8-inning team total) is 1.12x what independent innings predict,
# while lag-1 autocorrelation between adjacent innings is only 0.006. So innings
# are not sequentially correlated -- instead each GAME has a latent scoring
# environment (starter quality that day, park, wind, umpire zone) that raises or
# lowers both teams' rates for the whole game.
#
# Modeled as a per-TEAM-game multiplicative gamma factor with mean 1, drawn
# INDEPENDENTLY for each side. A shared (per-game) factor was tried first and
# rejected: it forces cov(home,away) to ~1.4 when the real value is ~0.05, and
# it inflates totals without widening margins. Independent per-team factors
# reproduce both the per-team variance and the near-zero covariance.
GAME_ENV_SD = _FITTED.get("GAME_ENV_SD", 0.28)

# Share of walk-off wins where the winning hit itself drives in extra runs, so
# the final margin exceeds 1. Measured: margin==1 in 85.8% of such wins.
WALKOFF_MULTIRUN_P = _FITTED.get("WALKOFF_MULTIRUN_P", 0.22)


def simulate_game(
    home_rate: float,
    away_rate: float,
    n_sims: int = 20000,
    rng: np.random.Generator | None = None,
    env_sd: float | None = None,
) -> SimResult:
    """Simulate one matchup n_sims times, inning by inning.

    home_rate / away_rate: expected runs per half-inning (league avg ~0.507).

    Stopping rules applied during simulation:
      - Bottom 9th is NOT played if the home team already leads.
      - Bottom 9th stops the moment the home team takes the lead (walk-off).
      - Extras continue until someone leads after a completed inning.
    """
    rng = rng or np.random.default_rng()
    sd = GAME_ENV_SD if env_sd is None else env_sd

    # Draw a per-game scoring environment (gamma, mean 1). Games are bucketed by
    # this factor so we can still sample innings vectorized per bucket.
    if sd > 0:
        shape = 1.0 / (sd ** 2)
        env_h = rng.gamma(shape, 1.0 / shape, size=n_sims)
        env_a = rng.gamma(shape, 1.0 / shape, size=n_sims)
    else:
        env_h = np.ones(n_sims)
        env_a = np.ones(n_sims)

    n_buckets = 24
    qs_h = np.quantile(env_h, np.linspace(0, 1, n_buckets + 1)[1:-1])
    qs_a = np.quantile(env_a, np.linspace(0, 1, n_buckets + 1)[1:-1])
    bucket_h = np.searchsorted(qs_h, env_h)
    bucket_a = np.searchsorted(qs_a, env_a)

    home = np.zeros(n_sims, dtype=np.int32)
    away = np.zeros(n_sims, dtype=np.int32)
    pmf_h_by_bucket: list[np.ndarray] = [None] * n_buckets   # type: ignore
    pmf_a_by_bucket: list[np.ndarray] = [None] * n_buckets   # type: ignore

    for b in range(n_buckets):
        ih = np.flatnonzero(bucket_h == b)
        if ih.size:
            ph = tilted_pmf(home_rate * float(env_h[ih].mean()))
            pmf_h_by_bucket[b] = ph
            home[ih] = _sampler(ph, rng, ih.size * 8).reshape(ih.size, 8).sum(axis=1)
        ia = np.flatnonzero(bucket_a == b)
        if ia.size:
            pa = tilted_pmf(away_rate * float(env_a[ia].mean()))
            pmf_a_by_bucket[b] = pa
            away[ia] = _sampler(pa, rng, ia.size * 8).reshape(ia.size, 8).sum(axis=1)
            away[ia] += _sampler(pa, rng, ia.size)   # top of the 9th

    # Bottom 9th: only if the home team is not already ahead.
    needs_bottom9 = home <= away
    n_b9 = int(needs_bottom9.sum())
    if n_b9:
        b9 = np.zeros(n_b9, dtype=np.int32)
        bsub = bucket_h[needs_bottom9]
        for b in range(n_buckets):
            m = np.flatnonzero(bsub == b)
            if m.size and pmf_h_by_bucket[b] is not None:
                b9[m] = _sampler(pmf_h_by_bucket[b], rng, m.size)
        deficit = (away - home)[needs_bottom9]          # >= 0
        # Walk-off truncation: once the home team leads, the game ends. If the
        # inning would produce more than enough runs, cap it at deficit+1.
        needed_to_win = deficit + 1
        # Walk-off truncation. The inning ends the moment the winning run
        # scores, so the recorded margin is usually exactly 1. But it is not
        # ALWAYS 1: a multi-run hit (e.g. a slam with the bases loaded down one)
        # clears the fence before play stops. Measured on real games, the margin
        # is 1 run in 85.8% of home wins where the winning run scored in the
        # final half-inning; capping at deficit+1 for every case would force
        # 100% and over-produce one-run games.
        walkoff = b9 >= needed_to_win
        excess = np.maximum(b9 - needed_to_win, 0)
        # Keep the surplus only when the winning hit itself was multi-run.
        keep = rng.random(n_b9) < WALKOFF_MULTIRUN_P
        capped = needed_to_win + np.where(keep, np.minimum(excess, 3), 0)
        b9 = np.where(walkoff, capped, b9)
        home[needs_bottom9] += b9

    # Extra innings while tied.
    went_extras = np.zeros(n_sims, dtype=bool)
    pmf_h_x = tilted_pmf(home_rate * EXTRA_INNING_MULT)
    pmf_a_x = tilted_pmf(away_rate * EXTRA_INNING_MULT)

    tied = home == away
    went_extras |= tied
    for _ in range(MAX_EXTRA_INNINGS):
        idx = np.flatnonzero(tied)
        if idx.size == 0:
            break
        a_add = _sampler(pmf_a_x, rng, idx.size).astype(np.int32)
        h_add = _sampler(pmf_h_x, rng, idx.size).astype(np.int32)
        # Home stops as soon as it leads in the bottom half.
        need = a_add + 1
        exc = np.maximum(h_add - need, 0)
        keep_x = rng.random(idx.size) < WALKOFF_MULTIRUN_P
        h_add = np.where(h_add >= need,
                         need + np.where(keep_x, np.minimum(exc, 3), 0),
                         h_add)
        away[idx] += a_add
        home[idx] += h_add
        tied = np.zeros(n_sims, dtype=bool)
        tied[idx] = home[idx] == away[idx]

    # Guard: nothing should remain tied (MLB has no ties in practice).
    still = home == away
    if still.any():
        home[still] += 1

    return SimResult(home, away, went_extras)


_RATE_TABLE: dict | None = None


def _build_rate_table(n_p: int = 41, n_t: int = 13, n_sims: int = 12000) -> dict:
    """Precompute rate solutions on a (p_win, expected_total) grid.

    Bisecting per game costs ~1s because each step runs a simulation. The map
    from (p_win, total) to rates is smooth, so solving it once on a grid and
    interpolating is ~1000x faster with negligible error.
    """
    path = PROC / "rate_table.json"
    if path.exists():
        try:
            d = json.loads(path.read_text())
            return {
                "p": np.array(d["p"]), "t": np.array(d["t"]),
                "ratio": np.array(d["ratio"]),
            }
        except Exception:
            pass

    ps = np.linspace(0.25, 0.80, n_p)
    ts = np.linspace(6.5, 12.0, n_t)
    ratio = np.zeros((n_p, n_t))
    rng = np.random.default_rng(4242)
    for i, p in enumerate(ps):
        for j, t in enumerate(ts):
            base_rate = t / (2 * 8.9)
            lo, hi = 0.35, 2.9
            for _ in range(22):
                mid = (lo + hi) / 2
                a = 2 * base_rate / (1 + mid)
                res = simulate_game(mid * a, a, n_sims=n_sims, rng=rng)
                if res.p_home_win() < p:
                    lo = mid
                else:
                    hi = mid
            ratio[i, j] = (lo + hi) / 2
    PROC.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "p": ps.tolist(), "t": ts.tolist(), "ratio": ratio.tolist()}))
    return {"p": ps, "t": ts, "ratio": ratio}


def rates_from_table(p_home_win: float, expected_total: float) -> tuple[float, float]:
    """Fast bilinear interpolation of the precomputed rate grid."""
    global _RATE_TABLE
    if _RATE_TABLE is None:
        _RATE_TABLE = _build_rate_table()
    ps, ts, R = _RATE_TABLE["p"], _RATE_TABLE["t"], _RATE_TABLE["ratio"]

    p = float(np.clip(p_home_win, ps[0], ps[-1]))
    t = float(np.clip(expected_total, ts[0], ts[-1]))
    i = int(np.clip(np.searchsorted(ps, p) - 1, 0, len(ps) - 2))
    j = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    fp = (p - ps[i]) / (ps[i + 1] - ps[i])
    ft = (t - ts[j]) / (ts[j + 1] - ts[j])
    ratio = ((1 - fp) * (1 - ft) * R[i, j] + fp * (1 - ft) * R[i + 1, j]
             + (1 - fp) * ft * R[i, j + 1] + fp * ft * R[i + 1, j + 1])

    base_rate = t / (2 * 8.9)
    a = 2 * base_rate / (1 + ratio)
    return float(ratio * a), float(a)


def rates_from_probability(
    p_home_win: float,
    expected_total: float = 9.05,
    innings: float = 8.9,
    tol: float = 1e-4,
    max_iter: int = 40,
) -> tuple[float, float]:
    """Invert P(home win) into per-half-inning scoring rates.

    Given the classifier's win probability and an expected total, find the
    (home_rate, away_rate) pair that reproduces both. Uses bisection on the
    rate ratio with a small simulation at each step.

    `innings` is below 9 because the bottom of the 9th is often skipped.
    """
    base_total_rate = expected_total / (2 * innings)
    rng = np.random.default_rng(12345)

    lo, hi = 0.35, 2.9  # ratio of home rate to away rate
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        # keep the sum of rates fixed so the total stays on target
        a = 2 * base_total_rate / (1 + mid)
        h = mid * a
        res = simulate_game(h, a, n_sims=4000, rng=rng)
        p = res.p_home_win()
        if abs(p - p_home_win) < tol:
            return h, a
        if p < p_home_win:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2
    a = 2 * base_total_rate / (1 + mid)
    return mid * a, a


def simulate_from_probability(
    p_home_win: float,
    expected_total: float = 9.05,
    n_sims: int = 20000,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """Convenience: classifier probability -> full joint distribution."""
    h, a = rates_from_table(p_home_win, expected_total)
    return simulate_game(h, a, n_sims=n_sims, rng=rng)


if __name__ == "__main__":
    print("Self-test: does the simulator reproduce known MLB structure?\n")
    rng = np.random.default_rng(7)

    # League-average matchup.
    res = simulate_game(0.5067, 0.5067, n_sims=200_000, rng=rng)
    s = res.summary()
    print("League-average matchup (equal strength):")
    print(f"  P(home win)      {s['p_home_win']:.4f}   actual MLB ~0.532 "
          f"(equal-strength sim excludes team quality, so ~0.53 from last-bat alone)")
    print(f"  E[total]         {s['exp_total']:.3f}   actual 9.047")
    print(f"  P(extras)        {s['p_extras']:.4f}   actual 0.082")
    print(f"  P(margin=+1)     {res.p_margin(1):.4f}   actual 0.1669")
    print(f"  P(margin=-1)     {res.p_margin(-1):.4f}   actual 0.1108")
    print(f"  walk-off ratio   {res.p_margin(1)/res.p_margin(-1):.3f}   actual 1.506")
    print(f"  P(|margin|=1)    {res.p_one_run_game():.4f}   actual 0.2778")
    print(f"  P(over 8.5)      {s['p_over_8_5']:.4f}   actual 0.5047")

    print("\nCorrelated markets (these need the joint distribution):")
    print(f"  P(home win AND over 8.5) = {s['p_home_win_and_over_8_5']:.4f}")
    print(f"  independent would give   = "
          f"{s['p_home_win'] * s['p_over_8_5']:.4f}")
    print(f"  P(home win | extras)     = {res.p_conditional_win_given_extras():.4f}")

    print("\nStrong home favorite (p=0.65):")
    r2 = simulate_from_probability(0.65, n_sims=60_000, rng=rng)
    s2 = r2.summary()
    print(f"  P(home win)   {s2['p_home_win']:.4f}  (target 0.65)")
    print(f"  E[total]      {s2['exp_total']:.3f}")
    print(f"  P(win by 1)   {r2.p_home_win_by(1):.4f}")
