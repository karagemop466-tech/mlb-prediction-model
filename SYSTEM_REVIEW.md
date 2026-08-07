# System Review — Backtesting, Simulation, and Reading the Results

Current as of 2026-08-01. Every number here was regenerated during this review,
not copied from earlier sessions.

---

## 1. What the system actually is

Three layers, each validated separately.

```
   FEATURES          →  115 point-in-time team features + 12 weather
   (features.py)        strictly prior-games-only, leakage-audited

   TWO MODELS        →  classifier: P(home win)      ensemble log+GBM+RF
   (predict.py)         run model:  E[home], E[away] side-specific, uses weather

   SIMULATOR         →  inning-by-inning Monte Carlo, 12,000 sims per game
   (simulate.py)        every market read off ONE joint distribution
```

The reason for the split: weather improves **run** prediction but was tested and
rejected for **win** prediction (−0.0009, inside noise). Feeding it to both would
have degraded the classifier.

---

## 2. Backtesting: what is actually measured

### The protocol

For each season S from 2019 to 2026:
1. Train on **every game before season S**
2. Predict season S
3. Never refit inside S

No shuffling, no k-fold, no peeking. This is the only protocol that matches how
the model would actually be used.

### Four independent validation gates

| Gate | Tests | What it catches |
|---|---|---|
| `audit_leakage.py` | 4/4 | features containing the answer |
| `verify.py` | 21/21 | data integrity, model determinism |
| `verify_sim.py` | **33/33** | simulator internal consistency |
| `verify_weather.py` | 28/28 | physics against reference values |

All four **block deployment** in the daily workflow. If any fails, the site stops
updating rather than publishing corrupted numbers.

### A real gap found during this review

`verify_sim.py` reported 28/28 instead of the expected 33/33. Cause: the four
first-five-innings tests were guarded by `if f5p.exists()` and were **silently
skipping**, because `first5.parquet` is a gitignored derived artifact that the CI
workflow never rebuilt.

That is not cosmetic. It meant **F5 markets were absent from live predictions in
CI** while appearing in the backtest. Two fixes applied:

1. The workflow now runs `first5.py` after `features.py`.
2. The test asserts the file exists rather than skipping, so the failure is loud.

This is what a review is for. The backtest said the market worked; production
was quietly not producing it.

---

## 3. Backtest results: all 13 markets, 17,724 games

| Market | Predicted | Actual | Bias | Skill |
|---|---|---|---|---|
| **win** | 0.5328 | 0.5315 | +0.0013 | **+0.0255** |
| **f5_home_lead** | 0.4481 | 0.4542 | −0.0061 | **+0.0150** |
| **win_and_under** | 0.2751 | 0.2758 | −0.0007 | **+0.0110** |
| **win_and_over** | 0.2577 | 0.2557 | +0.0020 | **+0.0094** |
| **over 8.5** | 0.4981 | 0.5005 | −0.0024 | **+0.0093** |
| under 8.5 | 0.5019 | 0.4995 | +0.0024 | +0.0093 |
| f5_over_4.5 | 0.4979 | 0.5118 | −0.0139 | +0.0055 |
| both_score | 0.8633 | 0.8725 | −0.0092 | +0.0024 |
| margin ≥ 3 | 0.5378 | 0.5409 | −0.0031 | +0.0005 |
| home win by 1 | 0.1753 | 0.1697 | +0.0056 | +0.0000 |
| one_run | 0.2876 | 0.2797 | +0.0079 | −0.0003 |
| f5_tie | 0.1508 | 0.1513 | −0.0005 | −0.0008 |
| away win by 1 | 0.1123 | 0.1100 | +0.0023 | −0.0015 |

### Reading the two columns that matter

**Bias** = predicted minus actual, averaged. Near zero means the market is
*calibrated*: when the model says 45%, it happens about 45% of the time. Every
market here is calibrated within 0.014, most within 0.006.

**Skill** = how much the Brier score beats always predicting the base rate.
This is the harder test. Calibration says the average is right; skill says the
model can tell *which* games differ.

The gap between them is the single most important thing to understand:

> **`one_run` has bias +0.0079 and skill −0.0003.** The model knows that about
> 28% of games are decided by one run, and says so accurately. It has **no
> ability** to tell you which specific game will be close. Treat it as a league
> constant, not a prediction.

Four markets are in that category: `one_run`, `hwin_by1`, `awin_by1`, `f5_tie`.
They are correct on average and uninformative per game.

### Per-season stability

| Market | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| win | +.037 | +.010 | +.029 | +.036 | +.021 | +.027 | +.022 | +.007 |
| f5_home_lead | +.026 | +.010 | +.017 | +.020 | +.013 | +.017 | +.008 | +.001 |
| over 8.5 | −.006 | +.010 | +.014 | +.007 | +.013 | +.010 | **+.016** | **+.012** |
| one_run | −.001 | −.000 | +.001 | +.001 | −.003 | +.000 | +.002 | −.004 |

Two honest observations:

- **`win` and `f5_home_lead` are positive in all 8 seasons** but are trending
  down. 2026 is the weakest year for both (+0.007, +0.001). That is partly a
  partial season (1,637 games) and partly a genuinely harder era — the same
  pattern showed up in the accuracy search, where every model lost ~1.5% on
  2024-26 holdout.
- **`over 8.5` is trending the other way**, and its two best seasons are the two
  most recent. That is the weather integration working: it was +0.0015 before
  weather, +0.0093 after.

---

## 4. Simulation: how it works and why

### Why simulate instead of training 13 classifiers

Separately-trained models are not forced to agree. One could output
P(home win)=0.60, P(over)=0.55, and P(win AND over)=0.62 — arithmetically
impossible, since a conjunction cannot exceed its marginals.

Simulating one scoreline and reading every market off it makes the answers
**coherent by construction**. The margin distribution sums to the win
probability because they are the same object.

### The three measurements that determined the design

| Measurement | Value | Consequence |
|---|---|---|
| corr(home runs, away runs) | **+0.0052** | a copula is pointless |
| variance / mean of runs | **2.18** | Poisson is wrong by 2x |
| P(margin=+1) vs P(−1) | **16.7% vs 11.1%** | walk-off rule |

The last one forced the architecture. Home teams win by exactly one run **51%
more often** than they lose by one, purely because play *stops* when they take
the lead in the ninth. No final-score model reproduces that from team strength;
it has to be simulated with the stopping rule applied.

### The generative model

For each of 12,000 simulations per game:

1. Draw a per-team scoring environment (gamma, sd 0.28) — independent per side,
   because a shared factor forced cov(home,away) to 1.39 when reality is 0.05
2. Innings 1-8: sample each half-inning from the empirical distribution,
   exponentially tilted to that team's rate
3. Top of 9th: away always bats
4. Bottom of 9th: **skipped if home already leads**; otherwise truncated the
   moment they take the lead
5. Extras while tied, with the automatic-runner scoring boost

Fitted parameters live in `data/proc/sim_params.json`; the fit loss improved
from 0.772 to **0.645** over two sessions.

### Validation: does it reproduce reality?

| Metric | Simulated | Actual |
|---|---|---|
| E[total runs] | 9.10 | 9.05 |
| var(margin) | 21.3 | 20.4 |
| P(margin=+1) / P(−1) ratio | 1.57 | 1.51 |
| P(extra innings) | 0.094 | 0.084 |
| **F5 total** | **5.107** | **5.114** |
| **P(tied after 5)** | **0.1519** | **0.1484** |

The last two are the strongest evidence. **First-five-innings scoring was never
a fitting target.** The simulator reproduces it because the inning-level
mechanism is right, not because it was tuned to. Had the model been quietly
overfit to full-game aggregates, F5 would have exposed it.

---

## 5. How to read a simulation result

Example from today's slate:

```
Matchup                P(home)  E[total]  Over 8.5  F5 home  1-run  Win&Over
Nationals @ Braves       0.610     9.29     0.511     0.487   0.290    0.308
```

**P(home) = 0.610** — the only number with strong validation. 56.9% accuracy,
skill +0.0255, calibration error under 0.003 in the 0.50-0.70 band. When it says
61%, that means 61%.

**E[total] = 9.29** — expected combined runs, driven by the weather-aware run
model. The slate range today is 8.5-10.9, and that spread is real signal, not
noise: it doubled after weather was added.

**Over 8.5 = 0.511** — note it does *not* track E[total] linearly, because the
run distribution is right-skewed.

**F5 home = 0.487** — probability the home team leads after five. Second-highest
skill market (+0.0150).

**1-run = 0.290** — **calibrated but not predictive.** Skill is −0.0003. This is
the league rate, lightly adjusted. Do not read it as a pick.

**Win&Over = 0.308** — the market that requires the joint distribution. It is
**not** P(home) × P(over) = 0.312. The small gap is the correlation the
simulator captures and independence misses.

### Three rules for reading any output

1. **Check the skill score before trusting a market.** Four of thirteen have
   skill ≈ 0. They are correct on average and useless per game.
2. **Probabilities near 0.50 mean "coin flip."** Nine of today's fifteen games
   sit inside 0.48-0.55. The model is explicitly saying it does not know.
3. **Expect to be wrong 4 times in 10.** That is what 56.9% accuracy looks like.
   A model that felt more confident would be lying.

---

## 6. Forward testing

`reports/forward_log.csv` holds timestamped predictions committed **before first
pitch**, graded afterward. Unlike a backtest, this cannot be corrupted by
hindsight or researcher degrees of freedom.

Current status: **72 logged, 30 graded, accuracy 0.5667** against a backtest
expectation of 0.5692.

That looks like a perfect match, and it is meaningless at this sample size —
the 95% interval is **±0.177**. Roughly 300 graded games are needed before the
forward test says anything. The site displays this caveat rather than presenting
30 games as validation.

**One gap:** the forward log records 11 markets but not the three F5 markets,
because F5 was added after the logging schema. Worth fixing so the newest and
second-most-skillful market accumulates forward evidence too.

---

## 7. Honest assessment

### What is solid

- Walk-forward protocol with no lookahead, four blocking validation gates
- Every market calibrated within 0.014
- Simulator reproduces MLB structure including effects it was never fitted to
- Eight rejected feature families, all documented with the reason

### What is weak

- **56.9% accuracy is near the sport's ceiling.** Published MLB models cluster
  at 55-58%. Gains from here are tenths of a percent.
- **Skill is declining in recent seasons** for the two best markets. Worth
  monitoring; may be a harder era or may be model drift.
- **Four markets have no per-game skill** and are presented as such.
- **Forward sample is 30 games.** Nothing can be concluded yet.
- **No ROI is claimed** and none can be, without historical closing odds.

### What would move the needle

**Both remaining structural candidates have now been tested and rejected.**

- **Umpires** (22,935 games, 137 umpires): correctly non-redundant, but the
  tendency does not persist. Split-half correlation +0.085 (p=0.38) for runs,
  +0.095 (p=0.35) for strikeouts. Umpire-to-umpire spread is only 1.12x what
  pure chance produces. Walk-forward −0.0004.
- **Bullpen workload** (measured reliever pitches, not the earlier proxy):
  correct sign and significant (−0.035 with home win, p=1.4e-04) but far too
  small. Walk-forward +0.0000.

That closes out the structural-feature hypothesis list. The refined lesson:
non-redundancy is a **screen, not a predictor of value**. Both structural
candidates passed the redundancy check that killed six participant features,
and both still failed — one because the quantity was noise, one because the
effect was hundredths of a run.

Ten rejections, two successes. What remains:

1. **Accept the ceiling.** ~55.9% on 2023-2026 under the honest day-by-day
   protocol, driven down by a 22% fall in team-quality dispersion since 2019.
   This is league parity, not model deficiency.
2. **Inning-level markets** — the simulator already generates them, and F5
   proved the mechanism transfers (+0.0150 skill, second-best market). Innings
   1-3 or 1-7 lines are near-free extensions.
3. **Market breadth over accuracy depth.** Every accuracy avenue tested has
   returned hundredths of a percent; new coherent markets have returned
   hundredths of a *skill point*, which is an order of magnitude better.
