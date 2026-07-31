"""Measure the empirical half-inning run distribution and home-field split.

Uses innings 1-8 ONLY. Innings 9+ are contaminated by stopping rules: the bottom
of the 9th is skipped when the home team leads and truncated on a walk-off, so
its observed mean understates true scoring ability. Calibrating on it would bake
the walk-off effect into the base distribution and then double-count it when the
simulator applies the stopping rule.

Source: inning-level play-by-play from the mlb-daily-results collector.
Writes data/proc/inning_dist.json, consumed by simulate.py.
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
DAILY = Path("/home/user/mlb-daily-results/data/games")

MAX_RUNS = 10


def collect() -> dict:
    top, bot = Counter(), Counter()
    n_games = n_extras = 0
    extra_top, extra_bot = Counter(), Counter()

    files = sorted(glob.glob(str(DAILY / "*" / "*.json")))
    if not files:
        raise SystemExit(f"No play-by-play found under {DAILY}")

    for f in files:
        try:
            day = json.loads(Path(f).read_text())
        except Exception:
            continue
        for g in day.get("games", []):
            if not g.get("is_final"):
                continue
            innings = g.get("innings") or []
            if not innings:
                continue
            n_games += 1
            if len(innings) > 9:
                n_extras += 1
            for i in innings:
                num = i.get("num", 0)
                a, h = i.get("away"), i.get("home")
                if num <= 8:
                    if a is not None:
                        top[min(a, MAX_RUNS)] += 1
                    if h is not None:
                        bot[min(h, MAX_RUNS)] += 1
                elif num > 9:
                    if a is not None:
                        extra_top[min(a, MAX_RUNS)] += 1
                    if h is not None:
                        extra_bot[min(h, MAX_RUNS)] += 1

    def pmf(c: Counter) -> list[float]:
        tot = sum(c.values())
        return [c.get(k, 0) / tot for k in range(MAX_RUNS + 1)] if tot else []

    def mean(c: Counter) -> float:
        tot = sum(c.values())
        return sum(k * v for k, v in c.items()) / tot if tot else 0.0

    combined = Counter()
    combined.update(top)
    combined.update(bot)

    m_top, m_bot, m_all = mean(top), mean(bot), mean(combined)
    m_xtop, m_xbot = mean(extra_top), mean(extra_bot)

    out = {
        "source": "innings 1-8 only (uncontaminated by walk-off/skip rules)",
        "n_games": n_games,
        "n_half_innings": sum(combined.values()),
        "pmf": pmf(combined),
        "mean": m_all,
        "mean_top": m_top,
        "mean_bottom": m_bot,
        "home_inning_advantage": (m_bot / m_top) if m_top else 1.0,
        "observed_extras_rate": n_extras / n_games if n_games else 0.0,
        "extra_inning_mult": ((m_xtop + m_xbot) / 2) / m_all if m_all and m_xtop else None,
        "n_extra_half_innings": sum(extra_top.values()) + sum(extra_bot.values()),
    }

    var = sum((k - m_all) ** 2 * v for k, v in combined.items()) / sum(combined.values())
    out["var"] = var
    out["dispersion_ratio"] = var / m_all
    return out


def main() -> None:
    d = collect()
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "inning_dist.json").write_text(json.dumps(d, indent=2))

    print(f"games              {d['n_games']:,}")
    print(f"half-innings (1-8) {d['n_half_innings']:,}")
    print(f"mean runs/half-inn {d['mean']:.5f}")
    print(f"variance           {d['var']:.5f}  (ratio {d['dispersion_ratio']:.3f} — "
          f"Poisson would be 1.0)")
    print(f"top (away)         {d['mean_top']:.5f}")
    print(f"bottom (home)      {d['mean_bottom']:.5f}")
    print(f"home inning edge   {d['home_inning_advantage']:.4f}x")
    print(f"observed extras    {d['observed_extras_rate']:.4f}")
    if d["extra_inning_mult"]:
        print(f"extra-inning mult  {d['extra_inning_mult']:.4f}x "
              f"({d['n_extra_half_innings']:,} half-innings)")
    print(f"\n-> {PROC / 'inning_dist.json'}")


if __name__ == "__main__":
    main()
