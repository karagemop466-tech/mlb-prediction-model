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

### Statcast — a real negative result

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

Statcast would more plausibly help at the **individual pitcher** level
(starter xwOBA-against vs. that specific lineup), which is the top item on the
roadmap below.

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
