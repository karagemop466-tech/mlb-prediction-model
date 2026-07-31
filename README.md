# MLB Game Prediction — Accuracy-Focused Model

Walk-forward validated MLB prediction model on **24,335 real games (2016–2026)**
from Retrosheet, the MLB Stats API and Baseball Savant Statcast.

**Live site:** https://karagemop466-tech.github.io/mlb-prediction-model/

## Results

| Metric | Value |
|---|---|
| Accuracy (walk-forward 2019–2026) | **56.92%** |
| Naive baseline (always pick home) | 53.19% |
| AUC | 0.5912 |
| Out-of-sample games evaluated | 17,046 |
| Leakage audit | **4/4 passed** |
| Correctness verification | **21/21 passed** |
| Calibration error (0.50–0.70 band) | < 0.003 |

Accuracy is reported with a 95% confidence interval of ±0.74%. Differences
smaller than that between model variants are statistical noise, not improvements.

## Correctness

`scripts/verify.py` runs 21 checks and the daily workflow **refuses to publish**
if any fail:

- **Data** — score validity, duplicates, per-season game counts, home-win rate
  0.5319 against the known ~0.535, average 9.05 runs/game
- **Ground truth** — random stored games re-verified against the live MLB Stats API
- **Features** — differentials equal home−away exactly; all values physically bounded
- **Leakage** — rolling windows independently rebuilt by hand, matched to 1e-16
- **Model** — probabilities sum to 1, deterministic output, chronological split,
  shuffled labels correctly collapse the model to chance

Two genuine bugs were found and fixed by this suite: a mislabeled tie-game
assertion and an incorrect bound applied to differential features.

## Correlated outcomes

The system simulates each game inning by inning (12,000 sims) and reads every
market off one joint distribution, so answers are coherent by construction.

| Market | Backtest bias | Skill |
|---|---|---|
| Winner | +0.0030 | +0.0247 |
| Win AND over 8.5 | +0.0011 | +0.0082 |
| Over 8.5 | −0.0054 | +0.0015 |
| One-run game | +0.0068 | −0.0006 |

Validated on 17,060 out-of-sample games. Reproduces the walk-off asymmetry
(P(margin=+1)=16.7% vs P(−1)=11.1%) that no final-score model can produce.
See [DESIGN_SIMULATION.md](DESIGN_SIMULATION.md).

## Quickstart

```bash
pip install -r requirements.txt
python scripts/build_dataset.py     # Retrosheet 2016-25 + StatsAPI 2026
python scripts/features.py          # 115 point-in-time features
python scripts/audit_leakage.py     # must pass 4/4
python scripts/verify.py            # must pass 21/21
python scripts/predict.py           # today's slate
python scripts/optimize_accuracy.py # accuracy-selected model search
cd scripts
python calibrate_innings.py         # measure half-inning distribution
python fit_simulator.py             # fit simulator to empirical targets
python verify_sim.py                # must pass 19/19
python backtest_markets.py          # walk-forward, every market
python research_loop.py             # test new hypotheses with a significance gate
python pricing.py                   # price sheet + coherence audit
```

## Model

A three-way average of logistic regression, gradient boosting and random forest.
All top candidates were statistically tied, so the ensemble was chosen because
averaging decorrelated learners reduces variance — rather than chasing the
highest number in a noisy search, which is itself a form of overfitting.

## Why 56.9% is near the ceiling

Published MLB models cluster at 55–58%. Baseball is the least predictable major
sport: the best team still loses ~35% of its games. Anyone advertising 65%+ on
MLB moneylines is overfitting, leaking, or lying.
