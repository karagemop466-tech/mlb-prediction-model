"""First-5-innings scoring, extracted from line scores.

WHY THIS EXISTS
---------------
The pitch-arsenal matchup showed a real, non-redundant signal (r=0.060 with
margin, p=1.7e-05) that nonetheless failed to improve the model. The diagnosis
was DILUTION: a starter throws ~90 of a game's ~300 pitches, and the bullpen --
which the matchup does not model -- throws the rest.

That diagnosis makes a falsifiable prediction. If dilution is the cause, the
matchup should correlate MORE strongly with first-5-innings scoring, where the
starter is usually still pitching, than with full-game scoring.

If F5 correlation is not higher, the dilution story is wrong and the effect is
simply small everywhere. Either answer is worth having.

Sources:
  - Retrosheet game logs, fields 19/20, store inning-by-inning runs as digit
    strings. Summing the first five characters gives F5 runs exactly.
  - 2026 comes from the daily-results collector's inning arrays.

Games with a double-digit inning (rendered "(10)") are skipped for character
parsing; they are ~0.2% and only affect innings 10+, never the first five.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "gamelogs"
PROC = ROOT / "data" / "proc"
DAILY = Path("/home/user/mlb-daily-results/data/games")

IDX = {"date": 0, "dblhdr": 1, "away": 3, "home": 6,
       "away_score": 9, "home_score": 10, "vis_line": 19, "home_line": 20}


def from_retrosheet() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.glob("gl*.txt")):
        with open(path, encoding="latin-1") as fh:
            for r in csv.reader(fh):
                if len(r) <= IDX["home_line"]:
                    continue
                vis, home = r[IDX["vis_line"]], r[IDX["home_line"]]
                if not vis or not home or "(" in vis or "(" in home:
                    continue
                # 'x' marks a half-inning not batted (home leading in the 9th).
                a5 = sum(int(c) for c in vis[:5] if c.isdigit())
                h5 = sum(int(c) for c in home[:5] if c.isdigit())
                # Require at least 5 innings actually played by both sides.
                if len([c for c in vis[:5] if c.isdigit()]) < 5:
                    continue
                d = r[IDX["date"]]
                rows.append({
                    "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                    "away": r[IDX["away"]], "home": r[IDX["home"]],
                    "dblhdr": r[IDX["dblhdr"]],
                    "f5_away": a5, "f5_home": h5,
                })
    return pd.DataFrame(rows)


def from_daily() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(DAILY / "*" / "*.json"))):
        try:
            day = json.loads(Path(f).read_text())
        except Exception:
            continue
        for g in day.get("games", []):
            if not g.get("is_final"):
                continue
            innings = g.get("innings") or []
            if len(innings) < 5:
                continue
            a5 = sum((i.get("away") or 0) for i in innings[:5])
            h5 = sum((i.get("home") or 0) for i in innings[:5])
            rows.append({
                "date": g["date"],
                "away": g["teams"]["away"]["abbr"],
                "home": g["teams"]["home"]["abbr"],
                "dblhdr": "0",
                "f5_away": a5, "f5_home": h5,
            })
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    rs = from_retrosheet()
    dl = from_daily()
    print(f"[f5] retrosheet {len(rs):,} games, daily-collector {len(dl):,} games")

    if not dl.empty:
        from build_dataset import TEAM_MAP
        dl["away"] = dl["away"].map(lambda x: TEAM_MAP.get(x, x))
        dl["home"] = dl["home"].map(lambda x: TEAM_MAP.get(x, x))

    df = pd.concat([rs, dl], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "home", "away", "dblhdr"], keep="first")
    df["f5_total"] = df.f5_away + df.f5_home
    df["f5_margin"] = df.f5_home - df.f5_away
    df["f5_home_lead"] = (df.f5_margin > 0).astype(int)

    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC / "first5.parquet", index=False)
    return df


def main() -> None:
    df = build()
    print(f"\n[f5] {len(df):,} games with first-5 scoring")
    print(f"  F5 total    mean {df.f5_total.mean():.3f}  sd {df.f5_total.std():.3f}")
    print(f"  F5 margin   mean {df.f5_margin.mean():+.3f}  sd {df.f5_margin.std():.3f}")
    print(f"  home leads after 5: {df.f5_home_lead.mean():.4f}")
    print(f"\n  full-game total mean is ~9.05, so F5 captures "
          f"{100*df.f5_total.mean()/9.05:.0f}% of scoring")
    print(f"  F5 margin sd {df.f5_margin.std():.3f} vs full-game 4.47 "
          f"-> {100*df.f5_margin.std()/4.47:.0f}% of the variance to explain")
    print(f"\n-> {PROC / 'first5.parquet'}")


if __name__ == "__main__":
    main()
