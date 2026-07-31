# MLB Prediction System — Findings

Built and validated 2026-07-26. Every number below is reproducible from the
scripts in this repo. Nothing is estimated, illustrative, or assumed.

---

## The headline you need to read first

**I cannot report an ROI figure, and neither can anyone else using only free data.**

ROI is not a property of a model. It is a property of a model *priced against a
market*. The same 56.8%-accurate model:

- **loses** money betting -150 favorites
- **makes** money betting +120 underdogs

Without real historical closing odds, any ROI number is fabricated. You asked
for no hallucinations, so `scripts/roi.py` **refuses to print an ROI** until you
supply an odds file. It computes calibration and break-even requirements
instead — those are knowable and honest.

I tested the commonly-cited free source (SportsbookReviewsOnline). **It is dead**
— the URLs return HTML error pages, not spreadsheets. Working sources are paid.

---

## What was actually built and verified

### Data — all real, all public

| Source | Coverage | Verified |
|---|---|---|
| Retrosheet game logs | 22,764 games, 2016–2025 | 10 seasons downloaded |
| MLB Stats API | 1,570 games, 2026 season | live, current |
| Baseball Savant pitch-level | 174 chunks, ~2M pitches | xwOBA/EV/barrel aggregated |
| **Total modeling set** | **24,334 games** | 2016-04-03 → 2026-07-25 |

Sanity check: home win rate **0.5319**. The published historical figure is ~0.535.
If this had come out at 0.60 or 0.48, the pipeline would be broken.

### Leakage audit — 4/4 passed

The single most common way these projects produce fake results is leakage. Four
independent tests, all in `scripts/audit_leakage.py`:

1. **Correlation ceiling** — max |corr| with outcome = **0.1475** (`d_rdiff_100`).
   Real MLB predictors top out at 0.10–0.15. Above 0.30 would mean leakage.
2. **Manual reconstruction** — hand-computed a Yankees rolling win% and matched
   the pipeline to **1e-16**, confirming game N uses only games 1…N-1.
3. **Shuffled labels** — AUC **0.4993** on randomized outcomes. Must be ~0.500;
   anything higher means the model found a back channel to the answer.
4. **Chronology** — no feature row references a future date.

### Model performance — walk-forward, 19,476 out-of-sample games

Train on all prior seasons, predict the next, never refit inside the test season.

| Model | Accuracy | AUC | Log loss | vs. baseline |
|---|---|---|---|---|
| Home-field baseline | 0.5312 | 0.500 | 0.6903 | — |
| Logistic (C=0.005) | 0.5644 | 0.5824 | 0.6818 | +1.37% |
| GBM | 0.5620 | 0.5831 | 0.6814 | +1.42% |
| GBM + isotonic | 0.5599 | 0.5828 | 0.6814 | +1.43% |
| **Ensemble (log+GBM)** | **0.5676** | **0.5900** | **0.67833** | **+1.55%** |

Per-season accuracy ranged 0.540–0.586 with no upward drift over time — a
model that "improves" every year is usually leaking.

**This is the real ceiling.** Published academic and industry MLB models cluster
at 55–58% accuracy. Baseball is the least predictable major sport: the best team
loses ~35% of its games. Anyone advertising 65%+ on MLB sides is overfitting,
leaking, or lying.

### Optimization — 24 configs, walk-forward validated

Selected on **log loss, not accuracy**, because calibrated probabilities
determine profit while raw accuracy does not.

The spread across all 24 configs was 0.6783–0.6813 — a 0.4% range. That tightness
is the signal ceiling asserting itself, not a tuning failure. More tuning will
not break through it; better *features* might, marginally.

### Statcast — two negative results

I aggregated ~2M pitches into rolling 30-game team xwOBA, exit velocity, barrel%
and hard-hit%. Values validated against known MLB norms (xwOBA .369, barrel 4.2%,
hard-hit 24.8%).

Head-to-head on identical seasons:

| Feature set | Log loss | AUC |
|---|---|---|
| Without Statcast (115 features) | **0.67743** | **0.5931** |
| With Statcast (139 features) | 0.67772 | 0.5921 |

**Statcast made it slightly worse** (+0.00029 log loss). Team-level
quality-of-contact is already embedded in run differential; adding it contributes
variance, not information. Reporting this rather than hiding it — it's the kind
of result that makes the rest of the numbers credible.

**Pitcher-level Statcast was then tested too, and also failed.**

This was the top roadmap item, so it got a full build: 3.4M pitches, 23,582
starts, 24 rolling-10-start features (xwOBA-against, EV, barrel%, hard-hit%,
whiff%, K%, BB%), joined via the Chadwick Bureau ID crosswalk.

| Feature set | Accuracy | AUC |
|---|---|---|
| Base (115 features) | 0.5570 | 0.5768 |
| Base + pitcher Statcast (139) | 0.5569 | 0.5790 |

Across **7,939 games (2023-2026)** with both starters known: accuracy
**-0.0001**, AUC **+0.0023**, both far inside the +/-1.09% CI. Not shipped.
Enable with `MLB_USE_SP_STATCAST=1` for further research.

Three real bugs surfaced during this experiment and were fixed:
1. `aggregate()` was **silently OOM-killed** on 3.4M rows, producing truncated
   output with no error. Rewritten to reduce each chunk before concatenating.
2. Savant's CSV endpoint **hard-caps at 25,000 rows and truncates without
   warning**. Added cap detection with recursive window splitting.
3. 2026 games key starters by MLBAM ID while Retrosheet seasons use retro IDs;
   the merge **silently dropped all 2026 coverage** until keyed on both.

### Calibration — the part that actually matters for betting

| Predicted | Games | Actual | Error |
|---|---|---|---|
| 0.40–0.50 | 4,831 | 0.4510 | +0.013 |
| 0.50–0.60 | 10,825 | 0.5451 | +0.002 |
| 0.60–0.70 | 3,049 | 0.6300 | −0.001 |

In the 0.50–0.70 range — **71% of all games** — calibration error is under 0.003.
When this model says 60%, it means 60%. The tails (>0.70, <0.30) are thin and
unreliable; don't bet them.

### Break-even table — the practical bottom line

| Confidence | Games | Actual win rate | You must beat |
|---|---|---|---|
| 0.50–0.53 | 5,530 | 0.5007 | −100 |
| 0.53–0.56 | 5,367 | 0.5422 | −118 |
| 0.56–0.60 | 5,002 | 0.5918 | −145 |
| 0.60–0.65 | 2,688 | 0.6194 | −163 |
| 0.65+ | 889 | 0.6760 | −209 |

**How to read this:** in the 0.56–0.60 bucket the model wins 59.2%, so it profits
at any price better than −145. Typical MLB moneylines carry 4–5% vig. The edge,
if it exists, lives in the 0.56–0.65 band — and only against soft prices.

The honest expectation: this model is **near break-even against efficient
markets**. Sharp books price MLB sides very well. Realistic paths to profit are
line shopping, stale numbers, and lower-liquidity markets — not raw model edge.

---

## Forward testing

`scripts/predict.py` writes timestamped predictions to `reports/forward_log.csv`
**before** games start, then grades them with `--score`. Already logged 12 games
for 2026-07-27. This is the only evaluation that cannot be corrupted by hindsight
or researcher degrees of freedom. Let it accumulate 300+ games before drawing
conclusions; expect ~56.8% accuracy and ~0.678 log loss.

---

## To activate ROI

Create `data/raw/odds/odds.csv`:

```csv
date,away,home,ml_home,ml_away
2026-07-27,BOS,OAK,+135,-155
```

Then `python scripts/roi.py`. Kelly staking, drawdown, CLV and an edge-threshold
sweep all activate automatically.

Sources that actually work (all paid): Scottfree Analytics (~28k MLB games),
The Odds API (historical tier), Sports Insights. Kaggle has free sets ending 2021
— usable for methodology validation, too stale for live deployment.

---

## Roadmap, in expected-value order

1. **Individual starting-pitcher Statcast** — pitcher xwOBA-against, whiff rate,
   pitch mix vs. the specific opposing lineup. Highest-upside remaining feature.
2. **Bullpen fatigue** — relief innings over the prior 3 days. Real, underpriced.
3. **Lineup-aware features** — actual posted lineups vs. season-long team stats.
4. **Umpire assignments** — measurable strike-zone effects on run environment.
5. **Weather at first pitch** — wind direction matters at Wrigley and Coors.
6. **Totals and run lines** — often softer than moneylines.

---

## Honest limitations

- **No ROI validated.** Structurally blocked without paid odds data.
- **56.8% accuracy is near the sport's ceiling.** Don't expect more from sides.
- **2020 is anomalous** (60-game season, no fans) and included; excluding it
  changes little but it is a known irregularity.
- **Statcast coverage is partial** (174/275 chunks at time of writing). Since it
  didn't help, completing it is low priority — but `python scripts/statcast.py`
  resumes from cache if you want it.
- **Model degrades on new teams/relocations.** The Athletics' move is mapped
  manually in `TEAM_MAP`.
- **This is a research tool, not betting advice.** A validated edge against
  efficient markets is rare; most people lose. Never stake money you can't lose.

---

## Correlated-outcome simulation (added 2026-07-31)

### Why a simulator, not more classifiers

The classifier answers one question: P(home win). It cannot answer "will they win
by exactly one run" or "will they win AND go over 8.5" coherently, because
separately-trained models are not forced to agree with each other.

Three measurements drove the design:

| Measurement | Value | Consequence |
|---|---|---|
| corr(home_score, away_score) | **+0.0052** | A copula is pointless — nothing to correlate |
| variance / mean of runs | **2.18** | Poisson is wrong by a factor of two |
| P(margin=+1) vs P(margin=−1) | **16.69% vs 11.08%** | Walk-off rule, unreproducible from final scores |

The +1/−1 asymmetry is the decisive one. Home teams win by one run 51% more
often than they lose by one, purely because the game *stops* when they take the
lead in the ninth. That is a rule artifact, so the model has to simulate innings
and apply the stopping rule.

### Architecture

Sample half-innings from the empirical distribution (measured on 25,136
half-innings, **innings 1–8 only** — the 9th is contaminated by the very
stopping rules being modeled), tilt them exponentially to hit each team's
scoring rate, and apply real stopping rules during play.

Fitted parameters (`scripts/fit_simulator.py`, coordinate search on 8 targets):

```
GAME_ENV_SD        0.28   per-team-game scoring dispersion
EXTRA_INNING_MULT  1.30   scoring rate in extras
WALKOFF_MULTIRUN_P 0.22   share of walk-offs clearing by >1 run
```

### Design decisions found by measurement, not assumption

1. **Shared vs independent dispersion.** A per-*game* environment factor was
   tried first. It forced cov(home,away) to 1.39 when the real value is 0.05.
   Switched to independent per-team factors, which reproduce both the per-team
   variance and the near-zero covariance.
2. **Walk-off truncation is not absolute.** Capping every walk-off at
   deficit+1 forces 100% one-run margins. Measured: when the winning run scores
   in the final half-inning, the margin is 1 run **85.8%** of the time — a slam
   clears the fence before play stops.
3. **The optimizer gamed a bad loss.** With low weights on variance it "fixed"
   the margin probabilities by inflating score variance 28% above reality. The
   weights were rebalanced onto relative error so variance cannot be traded away.
4. **The observed extra-inning multiplier (2.19x) is a selection artifact** —
   extras only *continue* while tied, so high-scoring frames end the game and
   never generate another observation. Fitted to the 8.21% extras rate instead.

### Market backtest: 17,060 games, walk-forward

| Market | Pred | Actual | Bias | Skill |
|---|---|---|---|---|
| win | 0.5348 | 0.5317 | +0.0030 | **+0.0247** |
| win_and_under | 0.2762 | 0.2743 | +0.0020 | **+0.0089** |
| win_and_over | 0.2585 | 0.2574 | +0.0011 | **+0.0082** |
| over 8.5 | 0.4976 | 0.5030 | −0.0054 | +0.0015 |
| margin ≥ 3 | 0.5402 | 0.5421 | −0.0018 | +0.0001 |
| away win by 1 | 0.1096 | 0.1098 | −0.0002 | +0.0001 |
| both score | 0.8627 | 0.8732 | −0.0105 | −0.0004 |
| one run | 0.2860 | 0.2792 | +0.0068 | −0.0006 |
| home win by 1 | 0.1764 | 0.1694 | +0.0070 | −0.0012 |

**Every market is calibrated** (worst bias 0.011). But only the winner and the
conjunctions carry real skill. One-run games and extra innings are essentially
**league constants** — the simulator predicts their rate correctly and has almost
no ability to say which specific game will be close. That is a real limit of the
sport, and it is reported rather than dressed up.

### Self-improvement loop

`scripts/research_loop.py` maintains a hypothesis registry, tests each with an
identical walk-forward protocol, and applies a significance gate that widens
with the number of ideas tested (Bonferroni-style). First session:

| Hypothesis | Δ accuracy | Bar | Verdict |
|---|---|---|---|
| bullpen_load | −0.0023 | +0.0104 | REJECT |
| travel_fatigue | +0.0014 | +0.0104 | REJECT |
| form_momentum | +0.0001 | +0.0104 | REJECT |

**0 of 3 promoted** — including bullpen fatigue, which FINDINGS.md previously
rated the most promising remaining idea. The gate exists precisely because this
project already has a case study in what happens without one: pitcher Statcast
looked like +1.0% on one season and was −0.01% across four.

### Pricing

`scripts/pricing.py` converts the joint distribution to fair American/decimal
odds and audits quote sets for internal contradictions via Fréchet bounds and
decomposition identities. It catches, for example, P(win)=0.60, P(over)=0.55,
P(win∧over)=0.62 — arithmetically impossible.

**No ROI is claimed and no edge over any real market is asserted.** This prices a
distribution and checks arithmetic; it does not assert profitability.


### Simulator upgrade: side-specific run model (2026-07-31)

The first market backtest gave totals a skill of only +0.0015. Diagnosis showed
the fault was the **input**, not the simulator: `expected_total_for()` produced
values with sd 0.373 and correlation 0.116 against realized totals. A near
constant in gives a near constant out.

Replaced with a learned model predicting **home and away runs separately**.
Measured justification: home runs correlate with home offense (0.086) and away
defense (0.091) at roughly equal strength, so collapsing both into one rate
derived from win probability discards half the information.

| | Heuristic | Learned side model |
|---|---|---|
| Correlation with realized total | 0.080 | **0.134** |
| Spread across games (sd) | 0.332 | **0.647** |

Market skill, walk-forward on 17,060 games:

| Market | Before | After |
|---|---|---|
| over/under 8.5 | +0.0015 | **+0.0055** |
| both teams score | -0.0004 | **+0.0020** |
| winner bias | +0.0030 | **+0.0012** |
| totals bias | -0.0054 | **+0.0001** |

The blend weight between the win-probability inversion and the run model was
selected by walk-forward search (w=0.75 maximises both winner and totals skill),
not chosen by hand.

**A double-count that measurement prevented.** Park factor is the single
strongest predictor of total runs (r=0.140), so an explicit park multiplier in
the simulator looked obviously correct. Checking first showed the side model
already reproduces the park effect almost exactly &mdash; predicted spread across
park quintiles 1.524 runs vs actual 1.487. Adding a park term would have
double-counted it. Not shipped.

Simulator correctness tests expanded from 19 to **26**, including leakage
protection for the new model and a direct check that two matchups with the same
win probability can produce different totals (0.600/0.596 win probability ->
10.29/7.26 expected total).

### Per-inning scoring profile: real effect, zero market value (2026-07-31)

Discovered that scoring is **not uniform across innings**. Extracted from
Retrosheet line scores (fields 19/20 store innings as digit strings, giving 10
seasons without new downloads): 22,711 games, 362,704 half-innings.

| Inning | Multiplier | Why |
|---|---|---|
| 1 | **1.057** | Top of the order is guaranteed to bat |
| 2 | **0.904** | Bottom of the order |
| 3–6 | ~1.02 | Steady state |
| 7–8 | 0.98 / 0.97 | Relievers replace tiring starters |

The home team's first inning is the single strongest cell at **1.159x**. The
2026 play-by-play showed the same shape independently (inning 1 = 0.544, inning
2 = 0.453), confirming it is structural rather than one season's noise.

**Implemented, tested, and disabled by default.** Across 10 markets and 17,060
games the change in total skill was **exactly 0.0000** (sum 0.0521 either way),
and mean absolute bias got slightly worse (0.0041 vs 0.0029).

The reason is clear in hindsight: markets settle on **end-of-game** totals and
margins. Reshaping which inning runs arrive in barely moves the sum, and the
per-inning sampling made simulation ~3x slower. Kept behind
`USE_INNING_PROFILE` because it would matter for inning-specific questions
(first-5-innings lines, "run in the 1st"), which this system does not price.

**A double-count caught during implementation.** The observed bottom-9th
multiplier is 0.78, which looks like weak late scoring. It is not: the inning is
skipped when the home team leads and truncated the instant it takes the lead.
The simulator already applies both rules, so using the observed value suppressed
home scoring twice — P(home win) fell from 0.528 to 0.519. Replaced with the
8th-inning level as the untruncated estimate.

**Net gain from the session:** refitting with a new `LEVEL_SCALE` parameter cut
fit loss from 0.772 to **0.645**, the best so far. E[total] error fell to +0.05
runs and margin variance to within 5%.
