# Today's Simulation — 2026-07-31

15 games, each simulated 12,000 times inning by inning. Every column below
comes from the same joint distribution, so the numbers are mutually consistent.

| Matchup | P(home) | E[total] | Over 8.5 | 1-run | Extras | Home by 1 | Win&Over | BTS |
|---|---|---|---|---|---|---|---|---|
| Nationals @ Braves | 0.610 | 9.29 | 0.511 | 0.290 | 0.092 | 0.189 | 0.308 | 0.875 |
| Twins @ Mariners | 0.543 | 8.92 | 0.488 | 0.295 | 0.090 | 0.182 | 0.264 | 0.863 |
| Sox @ Dodgers | 0.538 | 8.67 | 0.468 | 0.295 | 0.098 | 0.176 | 0.245 | 0.858 |
| Sox @ Rays | 0.534 | 8.54 | 0.458 | 0.298 | 0.097 | 0.182 | 0.234 | 0.850 |
| Yankees @ Cubs | 0.526 | 9.00 | 0.492 | 0.296 | 0.100 | 0.182 | 0.258 | 0.871 |
| Royals @ Rockies | 0.522 | 9.78 | 0.554 | 0.277 | 0.093 | 0.168 | 0.281 | 0.887 |
| Giants @ Padres | 0.512 | 8.73 | 0.471 | 0.296 | 0.093 | 0.175 | 0.232 | 0.860 |
| Phillies @ Orioles | 0.508 | 8.67 | 0.464 | 0.298 | 0.101 | 0.177 | 0.223 | 0.861 |
| Diamondbacks @ Guardians | 0.492 | 8.68 | 0.463 | 0.301 | 0.101 | 0.175 | 0.214 | 0.852 |
| Marlins @ Mets | 0.491 | 9.09 | 0.497 | 0.290 | 0.100 | 0.175 | 0.235 | 0.871 |
| Pirates @ Reds | 0.484 | 8.99 | 0.489 | 0.293 | 0.095 | 0.172 | 0.229 | 0.867 |
| Rangers @ Astros | 0.483 | 9.08 | 0.499 | 0.286 | 0.097 | 0.168 | 0.233 | 0.870 |
| Cardinals @ Jays | 0.465 | 8.86 | 0.480 | 0.288 | 0.096 | 0.171 | 0.212 | 0.860 |
| Brewers @ Angels | 0.445 | 8.99 | 0.489 | 0.286 | 0.095 | 0.167 | 0.203 | 0.861 |
| Tigers @ Athletics | 0.365 | 10.90 | 0.634 | 0.241 | 0.082 | 0.139 | 0.227 | 0.902 |

Slate: E[total] 9.08 (range 8.54–10.90), P(home) 0.365–0.610

## How to read this

**P(home)** — probability the home team wins. This is the only column with
strong validation: 56.9% accuracy, skill +0.0253 out of sample. Calibrated to
within 0.003 in the 0.50–0.70 range, so 0.610 genuinely means ~61%.

**E[total]** — expected combined runs. Drives every totals number. Range today
is 8.54–10.90, and that spread is the point: it reflects the actual matchup, not
a league constant.

**Over 8.5** — P(combined runs > 8.5). Skill +0.0051. Note it does *not* simply
track E[total] linearly, because the run distribution is right-skewed.

**1-run / Extras / Home by 1** — structural markets. **Calibrated but not
predictive.** Skill is ≈0 (−0.0008, −0.0005, −0.0007). The model predicts the
*rate* of one-run games correctly across many games but cannot tell you which
specific game will be close. Treat these as league constants, not as picks.

**Win&Over** — P(home wins AND total > 8.5). This is the column that requires a
joint simulator: it is not P(home) × P(over). For the Athletics game the joint
is 0.2208 while independence would say 0.2432 — a −0.0224 error, because a home
win and a high total are negatively related when the home team is weak.

**BTS** — both teams score at least one run.

## Worked example: Tigers @ Athletics

The most interesting game on the slate, and the one that shows why the
side-specific run model matters.

- **P(home) = 0.365** — the Athletics are the biggest underdog today
- **E[total] = 10.90** — and it is also the highest-scoring expected game

Those two facts together are what the old single-rate simulator structurally
could not represent. Expected runs are **home 4.80, away 6.00**: the Athletics
are not bad at scoring, they are bad at *preventing* runs, and Sutter Health
Park inflates offense. Low win probability, high total.

Full price sheet:

```
MONEYLINE  Athletics       +161 fair | +150 with 4.5% vig
           Detroit Tigers  -161 fair | -181 with 4.5% vig

TOTALS      7.5  over 0.699 (-232)   under 0.301 (+232)
            8.5  over 0.634 (-173)   under 0.366 (+173)
            9.5  over 0.544 (-119)   under 0.456 (+119)
           10.5  over 0.480 (+108)   under 0.520 (-108)

MARGIN     HOME by 1  0.1369 (+630)     <- most likely single outcome
           AWAY by 1  0.1021 (+880)
           AWAY by 2  0.0893 (+1019)
           AWAY by 3  0.0787 (+1171)

DERIVATIVES  one_run_game     0.2390 (+318)
             extra_innings    0.0797 (+1155)
             both_teams_score 0.9007 (-907)
             shutout          0.0993 (+907)

CORRELATED   home_win AND over 8.5  0.2208 (+353)
               independence would say 0.2432  (off by -0.0224)
             away_win AND over 8.5  0.4135 (+142)

COHERENCE: 0 issues
```

Note **HOME by 1 (0.1369) exceeds AWAY by 1 (0.1021)** even though the
Athletics are underdogs. That is the walk-off rule: the home team's one-run wins
are inflated because the game stops the moment they take the lead.

## What "fair" odds mean here

Fair odds are the break-even price implied by the simulated probability, with no
margin. The "with vig" column shows what a book would post at 4.5% overround.

**No ROI is claimed and no edge over any real market is asserted.** These are
model probabilities converted to prices. Whether they beat a real sportsbook is
untested — that would require historical closing lines this project does not
have. If a posted price differs from fair, the most likely explanation is that
the market knows something the model does not: today's lineups, a late scratch,
bullpen availability, or weather.

## Confidence caveats

- Fifteen games is far too few to judge anything. Expect the model to be wrong
  on ~4 of 10 picks; that is what a 56.9%-accurate model *looks like*.
- Probabilities near 0.50 (nine of today's fifteen games are inside 0.48–0.55)
  carry essentially no information. The model is saying "coin flip."
- The two highest-confidence picks today are Tigers over Athletics (0.635) and
  Braves over Nationals (0.610). Both are in the range where calibration is
  strongest.
- Starting pitchers are the posted probables and can change.
