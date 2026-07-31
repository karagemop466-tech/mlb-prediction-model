# Weather Effects — Today's Slate (2026-07-31)

Live forecast weather at each park, projected onto that park's plate-to-CF axis.

| Matchup | Temp | RH | Density | Wind out | Gust out | Roof | E[total] | Over 8.5 |
|---|---|---|---|---|---|---|---|---|
| Royals @ Rockies | 89°F | 17% | 0.790 | +1.3 | +2.0 | open | 10.92 | 0.641 |
| Tigers @ Athletics | 92°F | 23% | 0.933 | +9.7 | +14.5 | open | 10.59 | 0.608 |
| Nationals @ Braves | 90°F | 34% | 0.905 | +1.9 | +2.2 | open | 9.61 | 0.546 |
| Rangers @ Astros | 91°F | 53% | 0.931 | +10.3 | +18.3 | open | 9.44 | 0.526 |
| Yankees @ Cubs | 80°F | 59% | 0.932 | +4.4 | +7.3 | open | 9.35 | 0.519 |
| Pirates @ Reds | 89°F | 35% | 0.922 | -7.8 | -10.3 | open | 9.31 | 0.519 |
| Brewers @ Angels | 83°F | 54% | 0.941 | +5.5 | +6.6 | open | 9.24 | 0.513 |
| Cardinals @ Jays | 78°F | 45% | 0.950 | +5.8 | +12.3 | open | 9.06 | 0.499 |
| Marlins @ Mets | 78°F | 74% | 0.957 | +6.6 | +10.0 | open | 9.00 | 0.498 |
| Diamondbacks @ Guardians | 83°F | 49% | 0.926 | -4.9 | -6.8 | open | 8.88 | 0.483 |
| Twins @ Mariners | 79°F | 34% | 0.959 | -1.7 | -2.0 | open | 8.87 | 0.481 |
| Sox @ Dodgers | 76°F | 70% | 0.940 | +4.4 | +5.6 | open | 8.78 | 0.476 |
| Phillies @ Orioles | 85°F | 49% | 0.946 | -5.0 | -6.5 | open | 8.71 | 0.469 |
| Giants @ Padres | 72°F | 90% | 0.965 | -3.3 | -3.8 | open | 8.59 | 0.465 |
| Sox @ Rays | 82°F | 88% | 0.947 | +0.0 | +0.0 | CLOSED | 8.46 | 0.444 |

Density range: 0.790 – 0.965
E[total] range: 8.46 – 10.92 (sd 0.693, was 0.329 before weather)

## How to read the weather columns

**Density** — air density relative to standard sea level. Lower means thinner
air and more carry on fly balls. This is the single most predictive weather
variable (r = −0.144 with total runs). It combines temperature, humidity and
pressure, and beats all three individually.

**Wind out** — component of wind blowing from home plate toward center field, in
mph. Positive helps fly balls, negative suppresses them. Computed from the
compass wind direction and the park's `azimuthAngle`. Worth about **+0.18 runs
per 5 mph**.

**Gust out** — the same projection applied to the gust reading. Included for
completeness, but **it carries no measurable signal** (see below).

**Roof** — closed roofs and domes have all wind terms zeroed. About 14% of games.

## What the research established

**Air density is real and it is not just Coors.** The effect survives excluding
every high-elevation park (r = −0.101) and survives de-meaning by venue, which
removes park identity entirely (r = −0.087, p = 4e-35). Within a single park,
the thinnest-air decile scores **+0.68 runs** above that park's average and the
densest **−0.78** — a 1.45-run swing driven purely by conditions.

**Your gust hypothesis was tested directly and rejected.** The idea that gusts
act as a high-variance localized effect is intuitive, so it got a proper test:
Levene's test for unequal variance across gust quintiles.

| Measure | p-value | Result |
|---|---|---|
| gust excess over sustained | 0.58 | no variance effect |
| gust ratio | 0.47 | no variance effect |
| directional gust (out to CF) | 0.98 | no variance effect |

Run variance is essentially identical (20.4–21.4) in every gust quintile. Gusts
change neither the mean nor the spread of scoring.

**Moneyline: no usable weather effect.** Best correlation with a home win is
+0.016. Added to the win classifier, weather changed accuracy by −0.0009 against
a ±0.0083 significance bar, so the research gate rejected it. Weather is used
for run prediction only.

**Run line: small but real.** Thin air widens margins (r = −0.042 with |margin|,
p = 2e-09), which flows through the run model into margin markets.

## Today's extremes

The slate shows the physics working end to end. **Coors Field** sits at density
0.790 — 21% thinner than sea level, driven by 88°F, 17% humidity and 849 hPa —
and carries the highest expected total. **Petco** at 0.965 (72°F, 90% humidity,
sea level) carries one of the lowest. **Tropicana** is flagged closed, so its
wind terms are zeroed rather than inheriting irrelevant outdoor conditions.

Weather widened the spread of expected totals from sd 0.329 to sd 0.693 — the
model now distinguishes scoring environments more than twice as sharply.

*No ROI is claimed. These are model probabilities, not betting advice.*
