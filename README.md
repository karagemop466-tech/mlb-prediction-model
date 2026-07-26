# MLB Game Prediction — Sabermetric Model

Walk-forward validated MLB prediction system on **24,334 real games (2016–2026)**
from Retrosheet, MLB Stats API and Baseball Savant Statcast.

**Read [FINDINGS.md](FINDINGS.md) first** — especially the section on why ROI is
not reported.

## Results

| Metric | Value |
|---|---|
| Out-of-sample games | 19,476 |
| Accuracy | **56.8%** (baseline 53.1%) |
| AUC | 0.590 |
| Log loss | 0.6783 |
| Leakage audit | **4/4 passed** |
| Calibration error (0.5–0.7 band) | < 0.003 |
| ROI | **not computed — requires real odds** |

## Quickstart

```bash
python scripts/build_dataset.py    # Retrosheet 2016-25 + StatsAPI 2026
python scripts/features.py         # 115 point-in-time features
python scripts/audit_leakage.py    # MUST pass 4/4
python scripts/backtest.py         # walk-forward by season
python scripts/optimize.py         # 24-config search on log loss
python scripts/roi.py              # calibration + break-even
python scripts/predict.py          # today's slate, logged
python scripts/predict.py --score  # grade past predictions
```

## Why no ROI number

ROI depends on the price you get, not just the model. Free historical MLB
closing-odds archives are dead (SportsbookReviewsOnline returns HTML errors);
working sources are paid. Rather than invent a number, `roi.py` reports
calibration and break-even thresholds, and activates full ROI/Kelly/drawdown
the moment you drop in `data/raw/odds/odds.csv`.

## Anti-leakage design

Every rolling feature is `.shift(1)` before aggregation, so game N sees only
games 1…N-1. Verified four ways — see `scripts/audit_leakage.py`.
