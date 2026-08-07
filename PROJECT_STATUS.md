# What This Project Is, and What To Do Next

Written 2026-08-05. Every figure verified against a live run, not recalled.

---

## Part 1: What this actually is

Two connected things live on GitHub:

**`mlb-daily-results`** — a collector and website. Pulls every MLB game result
daily, publishes box scores, line scores and standings. Runs itself.
→ https://karagemop466-tech.github.io/mlb-daily-results/

**`mlb-prediction-model`** — the real project. A prediction and simulation
system for MLB games, with an unusually strict validation regime.
→ https://karagemop466-tech.github.io/mlb-prediction-model/

### The prediction system in one paragraph

It predicts who wins, how many runs score, and thirteen derived markets — not
by training thirteen models, but by simulating each game **inning by inning
12,000 times** and reading every market off one joint distribution. That makes
the answers mutually consistent: the margin probabilities sum to the win
probability because they are the same object. Two upstream models feed it: a
three-way ensemble for P(home win), and a separate weather-aware model for each
side's expected runs.

### Scale

| | |
|---|---|
| Games in the dataset | **24,481** (2016 → today) |
| Point-in-time features | 115 team + 12 weather |
| Code | 35 scripts, ~8,100 lines |
| Commits | 18 |
| Feature families tested | **12** |
| Documented experiments | 8 written up in full |

### What it actually achieves

| Market | Skill | Reading |
|---|---|---|
| **win** | **+0.0255** | 56.9% accuracy vs 53.2% baseline |
| **f5_home_lead** | **+0.0150** | leads after 5 innings |
| win_and_under | +0.0110 | conjunction, needs the joint model |
| win_and_over | +0.0094 | conjunction |
| over/under 8.5 | +0.0093 | totals, weather-driven |
| f5_over_4.5 | +0.0055 | first-five total |
| both_score, margin≥3 | +0.002 / +0.001 | marginal |
| one_run, hwin_by1, awin_by1, f5_tie | **~0.000** | **calibrated but not predictive** |

That last row matters more than it looks. Those four markets are *correct on
average* — the model knows ~28% of games are one-run affairs — but have no
ability to say **which** game. They are league constants, and the site labels
them as such rather than dressing them up.

### The validation regime

Four suites, all blocking deployment:

```
audit_leakage.py    4/4     features containing the answer
verify.py          21/21    data integrity, model determinism, ground truth
verify_sim.py      33/33    simulator internal consistency
verify_weather.py  28/28    physics against published reference values
```

Plus a **day-by-day walk-forward harness** with strategy formation on 2019-2022
and frozen replay on 2023-2026. This is stricter than the usual protocol because
it catches selection overfitting rather than merely penalising it.

### What makes it unusual

Most modeling projects report what worked. This one reports **ten rejections
against two successes**, each with the reason:

| Family | Why it failed |
|---|---|
| Team Statcast | redundant with team form |
| Pitcher Statcast | redundant (−0.0001 over 7,939 games) |
| Per-inning profile | real effect, exactly 0.0000 market value |
| Bullpen (proxy) | redundant |
| Travel / momentum | redundant |
| Starting lineups | redundant (corr 0.51 with team runs) |
| Player Statcast | redundant (corr 0.59) |
| Pitch-arsenal matchup | non-redundant, but effect only 0.19 runs |
| Umpires | **tendency does not persist** (split-half r=0.085, p=0.38) |
| Bullpen (measured) | correct sign, effect too small |
| **Weather** | **SHIPPED** — totals skill +0.0015 → +0.0093 |
| **F5 markets** | **SHIPPED** — new market at +0.0150 skill |

Three distinct failure modes emerged, and knowing which one applies is the
useful part: *redundancy*, *insufficient effect size*, and *measurement
unreliability*.

### The honest ceiling

Under the strictest protocol (day-by-day, frozen strategy, 2023-2026):
**~55.9% accuracy**, not the 56.9% the older season-boundary protocol reported.

The reason is measurable and is not the model's fault:

| Season | sd(team quality) |
|---|---|
| 2019 | 0.1085 |
| 2026 | **0.0848** |

Team-quality dispersion has fallen **22%**. Teams are converging, so games are
genuinely less predictable. Published MLB models cluster at 55-58%; this sits
inside that band and every avenue tried to exceed it has returned hundredths of
a percent.

**No ROI is claimed anywhere**, and none can be without historical closing odds,
which are not freely available.

---

## Part 2: What to work on next

Ten rejections carry information. The pattern says where *not* to look:

- **Participant features are exhausted.** Seven of ten rejections were
  redundancy with team rolling form. Anything describing who is playing is
  already encoded.
- **Structural features are exhausted too.** Weather worked; umpires and
  bullpen were the only other candidates and both failed.
- **Accuracy gains are exhausted.** Retrain cadence, recency weighting, model
  selection — all tested properly, all inside noise.

But note the asymmetry in what *did* work:

```
accuracy avenues tried   ->  hundredths of a PERCENT   (nothing)
new markets added        ->  hundredths of a SKILL POINT
                             weather: +0.0078 on totals
                             F5:      +0.0150, a whole new market
```

New markets have paid off roughly an order of magnitude better than accuracy
work. That should drive the priority order.

### Recommended, in order

**1. More inning-level markets — highest value, lowest cost**

The simulator already generates runs inning by inning. F5 proved the mechanism
transfers and landed as the second-best market in the system (+0.0150). The same
machinery gives, essentially for free:

- innings 1-3 and 1-7 leads and totals
- "will there be a run in the first inning" (a real, commonly quoted market)
- per-inning scoring distributions

The per-inning scoring profile already measured (inning 1 scores 5.7% above
average, inning 2 is 9.6% below) is currently disabled because it had no
full-game value — but it becomes **directly relevant** for inning-specific
markets. That work is already done and sitting behind a flag.

Estimated effort: small. Estimated value: another market or two at F5-like skill.

**2. Grow the forward test — the only unfinished validation**

Currently **30 graded games** against a needed ~300. The 95% interval is ±0.177,
so it confirms nothing yet. It costs nothing but calendar time, and it is the
one number that cannot be gamed by any modeling choice.

Concretely: let the daily workflow run, and revisit in ~6 weeks when the sample
is meaningful. Also worth verifying the newly-added F5 markets are accumulating
forward grades correctly.

**3. Calibration refinement in the thin tails**

Calibration is excellent in the 0.50-0.70 band (error < 0.003) covering 71% of
games, but the tails are thin and less reliable. Isotonic recalibration fitted
only on the tails might tighten them. Low ceiling, but honest and cheap.

**4. If you want ROI — buy odds data**

This is the only genuinely blocked capability. Everything is built and waiting:
`roi.py` computes Kelly staking, drawdown, CLV and edge sweeps the moment a
`data/raw/odds/odds.csv` appears. Sources are paid (~$50-200 one-off for
historical MLB closing lines). Without it, "is this profitable" is unanswerable
and the project correctly refuses to guess.

### What I would not do

- **Chase accuracy past 57%.** Twelve families tested; the ceiling is real and
  the era is getting harder.
- **Deep learning on the same features.** The information is the constraint, not
  the function class. A neural net on 115 redundant features will land in the
  same place with less interpretability.
- **Player-level anything.** Four separate attempts, all redundant.

---

## Part 3: Honest limitations

- **56-57% is near the sport's ceiling.** Baseball's best team loses ~35% of
  its games.
- **Four of thirteen markets have no per-game skill.** Labelled as such.
- **Forward sample is 30 games.** Nothing proven yet.
- **Weather coverage is 96.4%**, and 14% of games are under closed roofs where
  wind is correctly zeroed.
- **No ROI, no betting advice.** The pricing layer converts probabilities to
  odds and audits coherence; it asserts no edge over any real market.
