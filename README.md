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

## Quickstart

```bash
pip install -r requirements.txt
python scripts/build_dataset.py     # Retrosheet 2016-25 + StatsAPI 2026
python scripts/features.py          # 115 point-in-time features
python scripts/audit_leakage.py     # must pass 4/4
python scripts/verify.py            # must pass 21/21
python scripts/predict.py           # today's slate
python scripts/optimize_accuracy.py # accuracy-selected model search
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
