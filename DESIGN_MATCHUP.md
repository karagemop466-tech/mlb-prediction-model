# Pitch-Arsenal vs Lineup Matchup: Design

## Why this is not another "who is playing" feature

Six feature families have now been rejected because they were redundant with
team rolling form. All of them measured **aggregate quality** — how good is this
pitcher, how good are these hitters. Team run production already integrates that.

This hypothesis is different in kind. It asks about **interaction**:

> Does *this specific arsenal* exploit *these specific hitters' weaknesses*?

A pitcher who throws 50% sliders facing a lineup that is collectively helpless
against sliders is a different proposition from the same pitcher facing a lineup
that crushes them — even when both lineups have identical OPS and the pitcher
has identical ERA. Team rolling form **cannot** encode that, because it averages
over all the arsenals a team happened to face.

That is the entire bet. If this fails, it fails for a different reason than the
previous six, and that reason is worth knowing.

## The sample-size problem, measured

The naive version — batter X's history against pitcher Y — is dead on arrival:

| Granularity | PA available |
|---|---|
| batter vs specific pitcher | **~1.6 per pair** (measured over 2 days) |
| batter vs pitch TYPE | **~120-150 per season** |

Direct matchup history is noise. Even over a full career most batter-pitcher
pairs have fewer than 20 PA, which is far below the ~200 PA needed for wOBA to
stabilise. Anyone modeling "batter vs pitcher" splits directly is fitting noise.

**The workaround:** decompose the matchup through pitch types. Batter vs pitch
type has usable sample. Pitcher arsenal is directly observable and stable. The
matchup is then reconstructed as a weighted sum:

    expected_quality = sum over pitch types p of  arsenal_share[pitcher, p]
                                                * batter_performance[batter, p]

This is the standard trick for sparse interaction problems: factor a sparse
two-way table through a lower-dimensional shared basis.

## Pitch types are genuinely distinct

Measured league wOBA by pitch type (small sample, illustrative of the spread):

| Pitch | wOBA | xwOBA |
|---|---|---|
| Sweeper (ST) | .212 | .255 |
| Curveball (CU) | .238 | .263 |
| Slider (SL) | .275 | .295 |
| Changeup (CH) | .303 | .298 |
| Four-seam (FF) | .322 | .327 |
| Sinker (SI) | .344 | .337 |
| Cutter (FC) | .359 | .317 |

A 147-point wOBA spread between sweepers and cutters. If batters vary in how
they handle these — and they do — the interaction is real.

## Arsenals are stable and distinct

Measured over two days, pitchers with 40+ pitches:

```
450203:  CU 42%, FF 23%, SI 16%, CH 11%, FC 8%
477132:  SL 51%, FF 36%, CU 11%
500779:  SI 45%, CH 27%, CU 17%, SV 9%
519326:  FF 39%, SL 28%, CH 17%, SI 9%, ST 7%
```

These are not slight variations on a league-average mix. A 51%-slider pitcher
and a 45%-sinker pitcher present fundamentally different problems.

## Design

### 1. Pitcher arsenal (prior games only)

Rolling share of each pitch type over the pitcher's previous 10 starts,
`.shift(1)` before the rolling window so a start never sees itself.

### 2. Batter vs pitch type (prior games only, shrunk)

For each batter and pitch type, cumulative xwOBA allowed on that pitch type,
strictly before the current date.

**Shrinkage is mandatory here.** A batter with 12 PA against sweepers who went
4-for-9 will show a .500 wOBA that means nothing. Empirical-Bayes shrink toward
the league value for that pitch type:

    shrunk = (n * observed + k * league_mean) / (n + k)

with `k` set from the observed between-batter variance. Without shrinkage this
feature family would be a noise generator with a plausible name.

### 3. Platoon split

Handedness matters more than almost any other split, and it is well-sampled.
Batter performance is tracked separately vs LHP and RHP, and the starter's
handedness selects which value to use.

### 4. Matchup score

For each of the 9 lineup slots, the expected quality of that batter against
this pitcher's arsenal, PA-weighted by batting order, then aggregated:

    lineup_vs_arsenal = sum over slots  order_weight[slot]
                        * sum over pitch types  arsenal[p] * batter_xwoba[slot, p]

Plus a **dispersion** term: how much the matchup quality varies across the
lineup. A pitcher who neutralises 7 hitters but is destroyed by 2 is different
from one who is uniformly mediocre.

## What "correlation to game outcome" means here

The output is a per-game number for each side:

- `h_lineup_vs_opp_arsenal` — home lineup's expected quality vs the away starter
- `a_lineup_vs_opp_arsenal` — away lineup's expected quality vs the home starter
- `d_matchup` — the difference, which is the directional signal

Tested against three targets with the same walk-forward protocol and
significance gate used for every prior hypothesis:

1. **total runs** — does a favourable matchup raise scoring?
2. **home win** — does it move the winner?
3. **margin** — does it widen or narrow the result?

## How this could fail

Three honest failure modes, all worth distinguishing:

1. **Redundancy** — the arsenal-weighted score collapses to "good pitcher vs
   good hitters" and correlates ~0.5 with existing features, exactly like the
   previous six. Diagnostic: correlation with `d_pythag` and `d_rf_50`.
2. **Noise** — batter-vs-pitch-type samples are too thin even after shrinkage,
   so the interaction term is mostly estimation error. Diagnostic: does the
   signal strengthen when restricted to high-PA batters?
3. **Real but tiny** — starters face a lineup 2-3 times and throw ~90 pitches of
   a ~300-pitch game. Even a perfectly measured starter matchup is diluted by
   the bullpen, which is not modeled here.

Failure mode 3 is the one I expect. It is a reason to measure the effect on
**first-5-innings scoring** as well as full-game totals, since that is where the
starter's arsenal actually operates.
