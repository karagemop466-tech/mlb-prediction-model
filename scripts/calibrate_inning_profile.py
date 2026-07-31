"""Per-inning scoring profile from 10 seasons of Retrosheet line scores.

DISCOVERY
---------
Scoring is NOT uniform across innings. Measured on 22,711 games (362,704
half-innings), innings 1-8:

    inning 1  0.5380   +5.7% vs mean   <- top of the order is guaranteed to bat
    inning 2  0.4600   -9.6%           <- bottom of the order
    inning 3  0.5202   +2.2%
    inning 4  0.5185   +1.9%
    inning 5  0.5231   +2.8%
    inning 6  0.5209   +2.4%
    inning 7  0.4984   -2.1%           <- relievers arrive
    inning 8  0.4919   -3.3%

The simulator previously drew every inning from one distribution. That is wrong
in a way that matters for derived markets: it under-weights first-inning scoring
(where the best hitters are guaranteed at bats) and over-weights the second.

The 2026 play-by-play showed the same shape independently (inning 1 = 0.544,
inning 2 = 0.453), so this is a stable structural effect, not one season's noise.

Retrosheet game logs store line scores as digit strings in fields 19 (visitor)
and 20 (home), giving 10 seasons of inning detail without extra downloads.
Games with double-digit innings -- rendered as "(10)" -- are skipped; they are
~0.3% and would corrupt character-wise parsing.

Writes data/proc/inning_profile.json.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "gamelogs"
PROC = ROOT / "data" / "proc"

MAX_RUNS = 10
VIS_LINE, HOME_LINE = 19, 20


def collect() -> dict:
    pmf: dict[tuple[str, int], Counter] = defaultdict(Counter)
    games = extras = skipped = 0

    for path in sorted(RAW.glob("gl*.txt")):
        with open(path, encoding="latin-1") as fh:
            for row in csv.reader(fh):
                if len(row) <= HOME_LINE:
                    continue
                vis, home = row[VIS_LINE], row[HOME_LINE]
                if not vis or not home:
                    continue
                if "(" in vis or "(" in home:
                    skipped += 1
                    continue
                games += 1
                if len(vis) > 9:
                    extras += 1
                for i, ch in enumerate(vis[:9], 1):
                    if ch.isdigit():
                        pmf[("top", i)][min(int(ch), MAX_RUNS)] += 1
                for i, ch in enumerate(home[:9], 1):
                    if ch.isdigit():
                        pmf[("bot", i)][min(int(ch), MAX_RUNS)] += 1

    def as_pmf(c: Counter) -> list[float]:
        tot = sum(c.values())
        return [c.get(k, 0) / tot for k in range(MAX_RUNS + 1)] if tot else []

    def mean(c: Counter) -> float:
        tot = sum(c.values())
        return sum(k * v for k, v in c.items()) / tot if tot else 0.0

    # Baseline over innings 1-8, both halves.
    base = Counter()
    for half in ("top", "bot"):
        for i in range(1, 9):
            base.update(pmf[(half, i)])
    base_mean = mean(base)

    profile = {}
    for i in range(1, 10):
        both = Counter()
        both.update(pmf[("top", i)])
        both.update(pmf[("bot", i)])
        profile[str(i)] = {
            "mult": mean(both) / base_mean if base_mean else 1.0,
            "mult_top": mean(pmf[("top", i)]) / base_mean if base_mean else 1.0,
            "mult_bot": mean(pmf[("bot", i)]) / base_mean if base_mean else 1.0,
            "mean": mean(both),
            "n": int(sum(both.values())),
        }

    var = sum((k - base_mean) ** 2 * v for k, v in base.values.__self__.items()) \
        if False else sum((k - base_mean) ** 2 * v for k, v in base.items()) / sum(base.values())

    return {
        "source": "Retrosheet game log line scores, innings 1-9",
        "n_games": games,
        "n_half_innings": int(sum(base.values())),
        "skipped_double_digit_innings": skipped,
        "observed_extras_rate": extras / games if games else 0.0,
        "base_mean": base_mean,
        "base_var": var,
        "base_pmf": as_pmf(base),
        "profile": profile,
    }


def main() -> None:
    d = collect()
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "inning_profile.json").write_text(json.dumps(d, indent=2))

    print(f"games              {d['n_games']:,}")
    print(f"half-innings (1-8) {d['n_half_innings']:,}")
    print(f"baseline mean      {d['base_mean']:.5f}")
    print(f"baseline variance  {d['base_var']:.5f} "
          f"(ratio {d['base_var']/d['base_mean']:.3f})")
    print(f"extras rate        {d['observed_extras_rate']:.4f}")
    print(f"skipped (10+ inn)  {d['skipped_double_digit_innings']:,}")
    print(f"\n{'inn':>4}{'mult':>9}{'top':>9}{'bot':>9}{'mean':>9}{'n':>9}")
    for i in range(1, 10):
        p = d["profile"][str(i)]
        print(f"{i:>4}{p['mult']:>9.4f}{p['mult_top']:>9.4f}"
              f"{p['mult_bot']:>9.4f}{p['mean']:>9.4f}{p['n']:>9,}")
    print(f"\n-> {PROC / 'inning_profile.json'}")


if __name__ == "__main__":
    main()
