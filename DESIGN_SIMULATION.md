# Correlated-Outcome Simulation: Design

## The problem with independent models

The existing system predicts one thing: P(home win) = 0.57. That cannot answer:

- Will the home team win **by exactly one run**?
- Will they win **and** the game go over 8.5 total runs?
- What is P(home wins | game goes to extra innings)?

You could train a separate classifier per question, but the answers would be
mutually inconsistent — nothing forces P(win) to equal the sum of P(win by 1) +
P(win by 2) + ... A joint model must be **coherent by construction**.

## Why not a copula on (home_runs, away_runs)?

The obvious approach: fit marginal distributions for each team's runs, then bind
them with a Gaussian copula. Three measurements say no.

**1. There is nothing to correlate.** Measured on 24,349 games:

```
corr(home_score, away_score)          = +0.0052
mean within-park-season correlation   = -0.0007
```

Home and away scores are effectively **independent**. A copula would fit a
parameter that is indistinguishable from zero and add complexity for nothing.

**2. Runs are heavily overdispersed, so Poisson is wrong.**

```
home: mean 4.564  var  9.967   ratio 2.18
away: mean 4.483  var 10.511   ratio 2.35
```

Poisson requires variance == mean. Reality is more than twice that. A Poisson
scoreline model will produce far too few blowouts and far too few shutouts.

**3. The interesting correlation is not between teams — it is between
*questions* about the same game.** Margin, total, and winner are all functions of
one underlying scoreline. If you simulate the scoreline correctly, every derived
question is automatically consistent.

## The walk-off asymmetry: why score-level models fail

Empirical margin distribution over 24,349 games:

| Margin | Count | Share |
|---|---|---|
| −1 | 2,698 | 11.08% |
| **+1** | **4,065** | **16.69%** |

Home teams win by exactly one run **51% more often** than they lose by one.

This is not a team-strength effect. It is a **rule artifact**: the home team bats
last, and the game *stops the instant they take the lead* in the bottom of the
9th or later. A home team that would have scored 3 in the 9th is recorded as
scoring 1. No model of final scores can produce this asymmetry from team
strength parameters — it has to be simulated at the inning level, with the
stopping rule applied.

Any simulator that gets this wrong will systematically misprice one-run markets
and "win by exactly N" questions.

## Chosen architecture: inning-level Monte Carlo

Sample half-innings, not final scores.

Measured from 27,922 half-innings of 2026 play-by-play:

```
0 runs   72.624%      mean 0.5067
1 run    14.630%      var  1.0937
2 runs    6.876%      ratio 2.158   <-- matches game-level 2.18
3 runs    3.227%
4+        2.643%
```

The half-inning distribution is itself overdispersed at the same ratio as the
game totals. **This is the mechanism**: run-scoring arrives in bursts, and
summing 9 bursty innings reproduces game-level overdispersion for free. No
negative-binomial fitting, no copula, no dispersion parameter to tune.

### The generative model

For each simulated game:

1. Convert the model's team-strength estimate into an expected runs-per-inning
   rate for each side (`lambda_home`, `lambda_away`).
2. Innings 1..8: draw each half-inning's runs from the empirical distribution,
   **tilted** by that team's rate (see below).
3. Top of 9th: away bats.
4. Bottom of 9th: **if home already leads, skip it** — this is the walk-off rule.
   Otherwise home bats, and stops as soon as they take the lead.
5. If tied after 9: extra innings, with the modern automatic-runner rule raising
   the scoring rate. Continue until someone leads after a completed inning.

Because the stopping rules are applied during simulation, the +1/−1 asymmetry
emerges rather than being fitted.

### Tilting the empirical distribution

The base distribution has mean 0.5067 runs/half-inning. A strong offense against
a weak pitcher might need 0.68. Rather than assume a parametric family, tilt the
empirical PMF exponentially:

```
p_tilted(k) proportional to p_base(k) * exp(theta * k)
```

Solve for `theta` by bisection so the tilted mean equals the target. This is the
maximum-entropy adjustment: it changes the mean by the minimum distortion to the
observed shape, preserving the zero-inflation and heavy tail that make the
distribution realistic. `theta = 0` recovers the empirical distribution exactly.

## What this unlocks

One simulation produces a full joint distribution, so every derived market is
consistent by construction:

- P(home win) — must match the base classifier
- P(margin = k) for every k
- P(total runs > line) for any line
- P(home wins by 1) — the walk-off-sensitive market
- P(extra innings)
- P(shutout), P(both teams score)
- **Any conjunction**: P(home wins AND over 8.5)
- **Any conditional**: P(home wins | extras)

## Validation plan

A simulator is only useful if calibrated. Backtest, walk-forward, on every
derived market independently:

1. **Marginal calibration** — P(home win) from simulation must match both the
   classifier and reality.
2. **Margin distribution** — simulated vs actual, including the +1/−1 asymmetry.
3. **Total runs** — calibration across the whole line ladder.
4. **Joint outcomes** — P(win AND over) predicted vs realized.
5. **Extra-inning rate** — simulated vs the observed 8.2%.

Every one of these gets a Brier score and a calibration curve, walk-forward, out
of sample. Forward testing logs simulated probabilities for all markets before
first pitch and grades them afterward.

## Self-improvement loop

Rather than me hand-picking the next feature, `research_loop.py` maintains a
registry of hypotheses, tests each with the same walk-forward protocol, applies
a **statistical significance gate** (improvement must exceed the 95% CI), and
promotes only what passes. Everything tested is logged — including failures —
so the negative results accumulate as knowledge instead of being forgotten.

The gate matters. Testing 20 ideas and shipping the best-looking one is how you
overfit; this is exactly the trap the pitcher-Statcast experiment fell into
(+1.0% on one season, −0.0001 across four).

## On pricing

Pricing mechanisms are included because they are the natural way to *express* a
joint distribution: fair odds for each market, and detection of internally
inconsistent quotes. **No ROI is claimed and no betting advice is given** —
consistent with every prior finding in this project. The pricing layer converts
probabilities to odds and flags arbitrage-violating quote sets; it does not
assert an edge over any real market.
