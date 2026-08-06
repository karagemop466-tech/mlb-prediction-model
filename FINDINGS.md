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

---

## Weather integration (2026-07-31)

### Data sources — both verified, neither assumed

| Source | Provides | Coverage |
|---|---|---|
| Open-Meteo ERA5 archive | temp, humidity, **surface pressure**, wind speed/direction, **gusts**, hourly | 96.4% of games |
| MLB Stats API weather | stadium-relative wind ("12 mph, Out To CF"), roof status | 100% |
| MLB Stats API venues | **azimuthAngle**, elevation, roof type, lat/lon | 33 parks |

Surface pressure is the critical field: it encodes elevation physically (Coors
~848 hPa vs ~1013 at sea level). Sea-level-adjusted pressure would erase the
largest weather effect in baseball.

**A unit bug caught by cross-validation.** The MLB venue API labels its field
`elevation`, but the values are **feet**, not meters — Coors reads 5190. Open-Meteo
independently returns 1582 m for the same coordinates, and 5190 ft = 1582 m.
Using the API value as meters would have inflated Coors' air-density effect 3.3x.
Open-Meteo's elevation is used as the authority.

### Physics

Air density from temperature, humidity and pressure via partial pressures:

    rho = Pd/(Rd*T) + Pv/(Rv*T)

Validated against the reference value: ISA sea level returns **1.2250 kg/m³**.
Note humid air is **less** dense than dry air at equal T and P (water vapour is
18 g/mol vs ~29 for dry air) — the opposite of the common intuition.

Wind is projected onto each park's plate-to-CF axis using the venue azimuth.
Meteorological direction is where wind comes FROM, so a wind blowing out to
center arrives from bearing `azimuth + 180`:

    out_component = speed * cos(wind_from - (azimuth + 180))

**Independent cross-validation** against MLB's own stadium-relative readings:
temperature r = **0.952** (MAE 2.97°F), wind out-component r = **0.614** with
**78.1% sign agreement**, and every MLB label ("Out To CF", "In From LF", ...)
lands with the correct sign and correct ordering.

### Correlations with total runs (20,049 open-air games)

| Variable | r | p |
|---|---|---|
| **air_density_index** | **−0.144** | 1e-93 |
| pressure | −0.115 | 5e-60 |
| elevation | +0.108 | 4e-53 |
| temperature | +0.102 | 4e-47 |
| humidity | −0.081 | 1e-30 |
| wind_out | +0.018 | 0.011 |
| gust_excess | +0.005 | 0.45 |

The composite density index beats every individual component, confirming the
physics is the right construction rather than a curve fit.

**Is it just Coors?** No. Excluding all high-elevation parks the correlation is
still −0.101 (p=7e-45). De-meaning by venue — which removes park identity
entirely — leaves **−0.087** (p=4e-35). This is genuine weather, not a park proxy.

Effect sizes, within park:

- **−0.94 runs** per 0.05 air-density index
- **+0.35 runs** per 10°F
- **+0.18 runs** per 5 mph of wind blowing out

Air-density deciles span **+0.68 to −0.78 runs** versus the same park's average —
a 1.45-run swing, monotonic across all ten deciles.

### The gust hypothesis: tested and rejected

The hypothesis was that gusts act as a high-variance, localized effect. Tested
directly with Levene's test for unequal variance across gust quintiles:

| Measure | Levene W | p | Verdict |
|---|---|---|---|
| gust_excess (gust − sustained) | 0.721 | 0.58 | no effect |
| gust_ratio (gust / sustained) | 0.882 | 0.47 | no effect |
| gust_out (directional) | 0.114 | 0.98 | no effect |

Run variance is flat at 20.4–21.4 across every quintile. Gusts do **not**
measurably change either the mean or the variance of scoring. Included in the
feature set (they cost nothing) but they carry no signal.

### Moneyline and run line

- **Moneyline: no usable effect.** Best correlation with a home win is air
  density at r = +0.016 (p=0.02), which is negligible. Added to the win
  classifier, weather moved accuracy by **−0.0009** against a ±0.0083 noise bar —
  correctly REJECTED by the research gate.
- **Run line: real but small.** Air density correlates −0.042 (p=2e-09) with
  |margin| — thin air widens margins. Kept via the run model.

### Model impact

Weather is used for **run prediction only**, never for the win classifier.

| Target | Before | After | Change |
|---|---|---|---|
| Expected total (corr) | 0.1338 | **0.1569** | **+17%** |
| Home win accuracy | 0.5655 | 0.5646 | rejected |

Ablation on total runs shows where the value is:

| Feature family | Δ correlation |
|---|---|
| density only | +0.0196 |
| raw thermo (T, RH, P) | +0.0186 |
| density + wind | +0.0236 |
| **everything** | **+0.0251** |
| wind only | +0.0031 |
| gusts only | +0.0039 |

Market skill, walk-forward on 17,112 games:

| Market | Before | After |
|---|---|---|
| **over/under 8.5** | +0.0051 | **+0.0097** |
| **win AND under** | +0.0085 | **+0.0103** |
| **win AND over** | +0.0074 | **+0.0092** |
| margin ≥ 3 | −0.0001 | +0.0009 |
| home win by 1 | −0.0007 | +0.0002 |

Totals skill has now roughly **6x'd** across two sessions (+0.0015 → +0.0097).

Roughly 14.2% of games are under a closed roof or dome; all wind terms are
zeroed and flagged for those, so outdoor conditions are never attributed to
indoor games.

Weather correctness suite: **28/28** (`scripts/verify_weather.py`), including
physics checks against published reference values, projection geometry
identities, and the independent MLB cross-check.

---

## Starting lineups and player Statcast (2026-07-31)

Two related hypotheses, both built in full, both rejected by the significance
gate for the same structural reason.

### Data collected

| Asset | Volume |
|---|---|
| Starting lineups | 25,715 games, **97-99% coverage** 2016-2025 |
| Player game logs | 8,184 player-seasons, **524,296 player-game rows** |
| Batter Statcast | 265,578 batter-game rows (2021-2026) |

Lineups come from the schedule endpoint with `hydrate=lineups`, one request per
date (~1,900 calls instead of ~24,000 boxscores). Batter Statcast reuses the
pitch chunks already cached for the pitcher experiment, so no new downloads.

**Leakage control.** Player stats are accumulated from per-game logs and shifted
one game, so a game on date D sees only games before D. Verified by hand:
Ohtani's first 2024 game has NaN prior PAs, and his fifth game shows exactly 19
PAs, matching the manual sum of games 1-4.

### Hypothesis 1: lineup batting quality

30 features -- PA-weighted OPS/OBP/SLG/ISO using real batting-order weights,
top-5 OPS, K and BB rates, PA experience as a callup detector, and deviation
from each team's own 30-game lineup norm.

Raw correlations looked promising, and pointed at the winner rather than totals:

| Feature | vs margin | vs home win | vs total |
|---|---|---|---|
| d_lu_pa_experience | +0.117 | +0.101 | −0.003 |
| d_lu_bb_rate | +0.093 | +0.069 | −0.017 |
| d_lu_ops | +0.085 | +0.069 | +0.007 |

Walk-forward result: **home win −0.0003** against a ±0.0083 bar, **total runs
±0.0000**. Every ablation family (experience only, OPS only, discipline,
deviation only) landed inside noise.

### Hypothesis 2: player Statcast contact quality

The reasoning for a second attempt was specific: OPS is redundant with team run
totals, but xwOBA and barrel rate measure what a hitter *deserved*, stripping
sequencing luck and defensive positioning. That should be information team run
totals cannot contain.

| Feature | vs margin | vs home win |
|---|---|---|
| **d_lsc_xwoba** | **+0.114** | +0.092 |
| d_lsc_ev | +0.089 | +0.065 |
| d_lsc_hardhit | +0.080 | +0.061 |

Lineup xwOBA is the strongest raw lineup signal found. Walk-forward result:
**home win −0.0005**, **total runs ±0.0000**. Rejected.

### Why both failed — measured, not assumed

| Lineup feature | corr with team rolling runs scored |
|---|---|
| lineup OPS | 0.508 |
| **lineup xwOBA** | **0.590** |
| lineup xwOBA vs Pythagorean | 0.528 |

Contact quality turned out to be **more** redundant than OPS, not less. The
reason is mechanical: a team's rolling run production is computed from games
played by these same hitters, so it already integrates their contact quality.
Incremental R² on margin is +0.0047, which does not survive validation.

### Operational constraint, independently disqualifying

Lineups post ~3-4 hours before first pitch. Measured on the 2026-07-31 slate at
18:52 UTC: **3 of 15 games** had lineups posted, all starting within ~4 hours.
The daily workflow runs at 11:00 UTC. Even if lineups had helped, they would be
unavailable for most live predictions without adding a second afternoon run.

### A validation catch worth recording

Aggregate lineup exit velocity computes to **82.7 mph** against a commonly cited
~89. Investigated rather than assumed to be a bug: the raw pitch feed averages
82.25 mph across *all* batted balls, while Savant leaderboards filter to
qualified hitters. xwOBA (.320) and whiff rate (25.2%) match published norms
exactly, confirming the aggregation is faithful and the difference is a
population definition.

### Running tally of rejected feature families

1. Team Statcast — slightly negative
2. Pitcher Statcast — −0.0001 across 7,939 games
3. Per-inning scoring profile — real effect, exactly 0.0000 market value
4. Bullpen load / travel / momentum — all inside noise
5. **Starting lineups — −0.0003**
6. **Player Statcast — −0.0005**

Against one large success: **weather**, which lifted totals skill from +0.0051
to +0.0097. The pattern is consistent — features that describe *who is playing*
are redundant with team form, while features describing *external conditions*
are not.

---

## Pitch arsenal vs lineup matchup (2026-08-01)

The first rejected hypothesis that failed for a **different reason** than the
previous six. Worth reading for that alone.

### The question

Not "is this pitcher good" or "are these hitters good" — both already covered by
team form. Instead: does *this specific arsenal* exploit *these specific
hitters' weaknesses*? A 50%-slider pitcher facing a lineup that cannot hit
sliders is a different proposition from the same pitcher facing a lineup that
crushes them, even at identical OPS and ERA.

### Solving the sample-size problem

Direct batter-vs-pitcher history is unusable — measured at **~1.6 PA per pair**.
Even career-long, most pairs have under 20 PA against a ~200 PA stabilisation
threshold. Anyone modeling batter-vs-pitcher splits directly is fitting noise.

The matchup was therefore factored through pitch **types**, which do have sample
(~120-150 PA per batter-season):

    matchup = sum over families  arsenal_share[pitcher, p] * batter_xwoba[batter, p, hand]

| Component | Volume |
|---|---|
| Pitch chunks downloaded (with `pitch_type`) | 196 |
| Pitcher-game arsenals | 23,555 |
| Batter × family × hand rows | 547,592 |
| Games with a matchup score | 5,212 |

Batter values are empirical-Bayes shrunk toward the league mean per
(family, hand, season) with k=60 batted balls. The median cell holds only ~15
batted balls, so without shrinkage this would be a noise generator with a
credible-sounding name.

**A bug caught before it mattered:** an early version selected the batter's
platoon split by whichever handedness appeared first in the lookup rather than
the actual starter's hand. Platoon is one of the largest splits in baseball;
this would have corrupted every score. Fixed by deriving modal `p_throws` per
pitcher from the pitch data (924 pitchers, 690 R / 234 L).

### The signal is real

| Check | Value |
|---|---|
| corr(d_mu_xwoba, margin) | **+0.0595** (p=1.7e-05) |
| corr(h_mu_xwoba, home runs scored) | +0.039 |
| corr(a_mu_xwoba, away runs scored) | +0.049 |
| Seasonal consistency (early/mid/late) | 0.066 / 0.059 / 0.052 |

Both directional checks carry the correct sign: a favourable matchup produces
more runs *for the correct team*. This is not a spurious fit.

### And it is NOT redundant

| Existing feature | corr with matchup score |
|---|---|
| d_rf_50 (team rolling runs) | **0.155** |
| d_rdiff_100 | 0.121 |
| d_pythag | 0.111 |

Compare with the lineup families that preceded it: lineup OPS correlated 0.508
with team rolling runs, lineup xwOBA 0.590. **This feature contains genuinely new
information.** That is exactly what the hypothesis predicted.

### It still failed — on effect size

| Target | Base | With matchup | Δ |
|---|---|---|---|
| Home win | 0.5568 | 0.5562 | **−0.0006** |
| Total runs (corr) | 0.1536 | 0.1531 | −0.0005 |
| Home win, covered games only | 0.5605 | 0.5568 | −0.0036 |

The arithmetic explains why:

```
matchup spread (1 sd)     0.0129 xwOBA  ~=  0.19 runs/game
margin standard deviation                   4.47 runs
theoretical max correlation                 ~0.042
observed correlation                         0.0595
```

The observed correlation is **at or slightly above** the ceiling implied by the
effect size, so the measurement is if anything better than expected. The
matchup explains **0.35% of margin variance**. A 0.06 correlation cannot move a
56%-accurate classifier.

### Why the effect is small: dilution

A starter throws ~90 of a game's ~300 pitches and faces the lineup 2-3 times.
The bullpen — unmodeled here — throws the rest. The data shows this directly:

| Game type | corr(matchup, margin) |
|---|---|
| Low-scoring (≤7 runs) | +0.023 (n.s.) |
| Mid (8-10) | +0.067 |
| High (11+) | **+0.085** |

The signal is strongest exactly where starters stayed in and got hit, and absent
where they were pulled early — consistent with dilution rather than absence.

### What would make it usable

Two real extensions, neither small:

1. **First-5-innings markets**, where the starter's arsenal operates undiluted.
   The arsenal and split tables built here are the prerequisite for that work.
2. **Bullpen arsenal modeling** to cover the remaining ~210 pitches.

### Updated tally

Seven rejected families, one success:

| Family | Redundant? | Verdict |
|---|---|---|
| Team Statcast | yes | slightly negative |
| Pitcher Statcast | yes | −0.0001 |
| Inning profile | n/a | exactly 0.0000 |
| Bullpen / travel / momentum | yes | inside noise |
| Starting lineups | yes (0.51) | −0.0003 |
| Player Statcast | yes (0.59) | −0.0005 |
| **Pitch matchup** | **no (0.15)** | **−0.0006, effect too small** |
| **Weather** | **no** | **+0.0046 totals skill — shipped** |

The refined lesson: not redundant is necessary but not sufficient. Weather works
because it moves scoring by ~1.5 runs across its range. The matchup is real,
novel, correctly measured — and moves it by 0.19.

---

## First-five-innings markets, and a falsified hypothesis (2026-08-01)

### The dilution hypothesis was wrong

The pitch-arsenal matchup produced a real, non-redundant signal (r=0.060 with
margin) that added nothing to the model. I attributed that to **dilution**: a
starter throws ~90 of ~300 pitches, so the bullpen washes out his contribution.

That was a falsifiable claim, so I tested it. If dilution is the cause, the
matchup must correlate more strongly with first-five-innings scoring, where the
starter is usually still pitching.

| Target | corr with matchup | sd |
|---|---|---|
| F5 margin | +0.0519 | 3.32 |
| Full margin | +0.0553 | 4.44 |
| **Ratio F5/full** | **0.939** | |

**Below 1. The hypothesis is falsified.** The effect in runs is also smaller on
F5 (0.172) than on the full game (0.246).

A control makes this precise. Every feature loses correlation on F5, because it
is a smaller and noisier sample:

| Feature | F5 | Full | Ratio |
|---|---|---|---|
| d_rdiff_100 | 0.136 | 0.166 | 0.820 |
| d_pythag | 0.127 | 0.151 | 0.841 |
| d_rf_50 | 0.088 | 0.106 | 0.825 |
| **d_mu_xwoba (matchup)** | 0.052 | 0.055 | **0.940** |

So the matchup does hold up *relatively* better on F5 than team-quality features
do — dilution is real but weak. It is not why the feature failed. The feature
failed because the effect is small everywhere.

One further detail, from 2026 inning-level data: the matchup signal lives
entirely in the **first time through the order**.

| Innings | corr(matchup, margin) |
|---|---|
| 1-3 | +0.0368 |
| 4-5 | −0.0102 |

Hitters adjust after seeing an arsenal once. Walk-forward on F5 markets
confirmed the conclusion: matchup adds −0.0002 to F5 total, −0.0001 to F5
margin, −0.0002 to F5 home lead. Rejected on every target.

### The F5 infrastructure turned out to be the real find

Extracted first-five scoring for **24,266 games** from Retrosheet line scores
(fields 19/20 store innings as digit strings) plus the 2026 collector.

The simulator reproduces F5 **without any additional fitting**:

| Metric | Simulated | Actual |
|---|---|---|
| F5 total | 5.107 | 5.114 |
| F5 margin sd | 3.505 | 3.398 |
| P(home leads after 5) | 0.4449 | 0.4550 |
| P(tied after 5) | 0.1519 | 0.1484 |

F5 was never a fitting target, so this is a genuine out-of-sample validation of
the inning-level generative model. Had the simulator been quietly overfit to
full-game aggregates, F5 would have exposed it.

### F5 markets have real skill

Walk-forward, 17,620 games:

| Market | Bias | Skill |
|---|---|---|
| win (full game) | +0.0013 | +0.0255 |
| **f5_home_lead** | **−0.0061** | **+0.0150** |
| win_and_under | −0.0007 | +0.0110 |
| win_and_over | +0.0020 | +0.0094 |
| over 8.5 | −0.0024 | +0.0093 |
| **f5_over_4.5** | −0.0139 | **+0.0055** |
| f5_tie | −0.0005 | −0.0008 |

**F5 home lead is now the second-most-skillful market in the system**, ahead of
every totals market. The reason is the mirror image of the dilution story: over
five innings the outcome depends more on the two starters and less on bullpen
and late-inning randomness, so team-quality signal survives better.

Simulator correctness suite expanded to **32/32**.

### Updated tally

| Family | Verdict |
|---|---|
| Team Statcast | rejected |
| Pitcher Statcast | rejected |
| Inning profile | rejected (0.0000) |
| Bullpen / travel / momentum | rejected |
| Starting lineups | rejected |
| Player Statcast | rejected |
| Pitch matchup | rejected (effect too small, dilution ruled out) |
| **Weather** | **shipped: totals skill +0.0051 -> +0.0093** |
| **F5 markets** | **shipped: new market at +0.0150 skill** |

Eight rejections, two successes. Both successes came from modeling **structure**
(external conditions, game phase) rather than **participants**.

---

## Day-by-day walk-forward: strategy formation then frozen replay (2026-08-05)

### The flaw this addresses

The existing backtest retrains at **season boundaries**. A game on 2024-09-30
was predicted by a model that had never seen a single 2024 game — 2,429 games of
in-season information discarded. That answers "how good was last winter's model
all year?" rather than "how good is the system when operated properly?"

### The design

**Phase 1 (2019-2022):** compare 8 operating strategies — retrain cadence from
season-boundary to every 7 days, rolling training windows, recency weighting.

**Phase 2 (2023-2026):** freeze the phase-1 winner and replay day by day.
Predict each day using only prior days; absorb results afterward. The strategy
is never re-tuned. This is a forward test conducted inside history.

### Phase 1: no strategy separated from the field

| Strategy | Accuracy |
|---|---|
| retrain_14d_hl4000 | 0.5821 |
| retrain_14d_hl8000 | 0.5811 |
| **season_boundary (baseline)** | **0.5790** |
| retrain_14d_win12000 | 0.5787 |
| retrain_7d | 0.5779 |
| retrain_14d | 0.5777 |
| retrain_30d | 0.5768 |

**All 8 within 1 SE (±0.0107).** Retraining every 7 days scored *worse* than
retraining once per season.

The reason is instructive: the rolling features already update after every game.
Refitting the model more often adds little, because the information the old
protocol appeared to discard was never actually missing — it lives in the
features, not the coefficients.

### Phase 2: the selected edge reversed sign

| | Phase 1 | Phase 2 | Change |
|---|---|---|---|
| retrain_14d_hl4000 (selected) | 0.5821 | **0.5556** | −0.0265 |
| season_boundary (baseline) | 0.5790 | **0.5586** | −0.0204 |

The phase-1 winner beat the baseline by **+0.0031**. In the frozen replay it
**lost by −0.0030**. An edge that reverses sign out of sample is noise, and this
is exactly what picking the maximum of eight noisy candidates produces.

Both strategies dropped ~2 points, so roughly 77% of the decline is common to
any model rather than caused by the selection.

### Why recent seasons are harder: league parity

| Season | sd(pythag) | sd(run diff, 100g) |
|---|---|---|
| 2019 | 0.1085 | 0.944 |
| 2022 | 0.1075 | 0.928 |
| 2024 | 0.0950 | 0.845 |
| 2026 | **0.0848** | **0.753** |

Team-quality dispersion has fallen ~22%. Teams are converging, so games are
genuinely less predictable. This is a property of the league, not model decay,
and it caps what any model can achieve.

### Implication for the accuracy target

Higher accuracy is **not** available from retrain scheduling or recency
weighting. Both were tested properly under a protocol designed to catch exactly
this, and neither helped. The honest 2023-2026 figure is **~55.9%**, below the
56.9% measured under the older protocol, because the recent era is harder.

### What this leaves behind

A reusable harness. Any future strategy can be formed on 2019-2022 and replayed
frozen on 2023-2026 to get an un-inflated estimate before shipping. That is a
stronger guarantee than the significance gate alone, because it catches
selection overfitting rather than merely penalising it.
