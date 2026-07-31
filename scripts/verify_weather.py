"""Correctness tests for the weather pipeline.

Weather is the first feature family in this project built on external physics
rather than pure statistics, so the physics itself must be verified against
known values, not just checked for plausibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from weather import (air_density, density_index, parse_mlb_weather,
                     wind_cross_component, wind_out_component)

PROC = Path(__file__).resolve().parent.parent / "data" / "proc"
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 66)
    print("WEATHER PIPELINE CORRECTNESS")
    print("=" * 66)

    print("\nAIR DENSITY PHYSICS (against published reference values)")
    isa = air_density(59.0, 0.0, 1013.25)
    check("ISA sea level = 1.225 kg/m3", abs(isa - 1.225) < 0.002, f"{isa:.4f}")

    # Humid air is LESS dense than dry air at the same T and P.
    dry, humid = air_density(85, 10, 1013), air_density(85, 90, 1013)
    check("humid air less dense than dry air", humid < dry,
          f"90% RH {humid:.4f} < 10% RH {dry:.4f}")

    cold, hot = air_density(45, 50, 1013), air_density(95, 50, 1013)
    check("cold air denser than hot air", cold > hot, f"{cold:.4f} > {hot:.4f}")

    coors, sea = air_density(85, 30, 848), air_density(85, 30, 1013)
    check("Coors air ~19% thinner than sea level at equal T/RH",
          0.15 < (1 - coors / sea) < 0.20, f"{100*(1-coors/sea):.1f}% thinner")

    check("density index is 1.0 at ISA",
          abs(density_index(59, 0, 1013.25) - 1.0) < 0.002)

    print("\nWIND PROJECTION GEOMETRY")
    # azimuth 0 => CF due north. Wind FROM the south blows out to CF.
    check("wind from south with CF north blows OUT",
          abs(wind_out_component(180, 10, 0) - 10) < 0.01,
          f"{wind_out_component(180, 10, 0):+.2f}")
    check("wind from north with CF north blows IN",
          abs(wind_out_component(0, 10, 0) + 10) < 0.01,
          f"{wind_out_component(0, 10, 0):+.2f}")
    check("perpendicular wind has zero out-component",
          abs(wind_out_component(90, 10, 0)) < 0.01)
    check("perpendicular wind is full crosswind",
          abs(wind_cross_component(90, 10, 0) - 10) < 0.01)
    check("out and cross components obey Pythagoras",
          all(abs(np.hypot(wind_out_component(d, 12, 35),
                           wind_cross_component(d, 12, 35)) - 12) < 0.01
              for d in (0, 45, 130, 210, 300)))
    # Rotating the park by 180 degrees must flip the sign.
    check("rotating stadium 180 degrees flips the sign",
          abs(wind_out_component(180, 10, 0) + wind_out_component(180, 10, 180)) < 0.01)

    print("\nMLB WEATHER STRING PARSING")
    w = parse_mlb_weather({"condition": "Sunny", "temp": "78",
                           "wind": "12 mph, Out To CF"})
    check("parses temperature", w["mlb_temp_f"] == 78)
    check("parses wind speed", w["mlb_wind_mph"] == 12)
    check("out-to-CF is fully positive", abs(w["mlb_wind_out"] - 12) < 0.01,
          f"{w['mlb_wind_out']:+.1f}")
    w2 = parse_mlb_weather({"condition": "Roof Closed", "temp": "72",
                            "wind": "0 mph, None"})
    check("detects closed roof", w2["roof_closed"] == 1)
    w3 = parse_mlb_weather({"condition": "Cloudy", "temp": "65",
                            "wind": "9 mph, In From CF"})
    check("in-from-CF is negative", w3["mlb_wind_out"] < 0,
          f"{w3['mlb_wind_out']:+.1f}")
    check("crosswind label gives zero out-component",
          parse_mlb_weather({"temp": "70", "wind": "8 mph, L To R"})["mlb_wind_out"] == 0)

    print("\nGAME-LEVEL DATA")
    path = PROC / "weather_games.parquet"
    if not path.exists():
        check("weather_games.parquet exists", False)
    else:
        wx = pd.read_parquet(path)
        check("weather coverage above 90%", wx.has_weather.mean() > 0.90,
              f"{100*wx.has_weather.mean():.1f}%")
        o = wx[(wx.has_weather == 1) & (wx.is_closed == 0)]
        check("temperatures physically plausible",
              bool(o.temp_f.between(-10, 125).all()),
              f"[{o.temp_f.min():.1f}, {o.temp_f.max():.1f}]")
        check("humidity within 0-100",
              bool(o.humidity.between(0, 100).all()))
        check("pressures physically plausible",
              bool(o.pressure_hpa.between(780, 1060).all()),
              f"[{o.pressure_hpa.min():.1f}, {o.pressure_hpa.max():.1f}]")
        # ERA5 reports sustained wind and gusts from different model
        # diagnostics, so a handful of hours have gust marginally below
        # sustained. Verified as a reanalysis artifact, not a pipeline bug:
        # 37 of 20,049 rows (0.19%), max shortfall 5.8 mph. Assert it stays rare
        # rather than asserting it never happens.
        bad_gust = int((o.gust_mph < o.wind_mph - 0.01).sum())
        check("gust below sustained wind is rare (ERA5 artifact)",
              bad_gust / len(o) < 0.01,
              f"{bad_gust}/{len(o)} rows ({100*bad_gust/len(o):.2f}%)")
        # Projection identity: |out| <= speed, checked on rows where the wind
        # was not zeroed by the roof rule.
        live = o[(o.wind_mph > 0) & o.wind_out.notna()]
        viol = int((live.wind_out.abs() > live.wind_mph + 0.01).sum())
        check("wind_out never exceeds wind speed", viol == 0,
              f"{viol} violations in {len(live):,} open-air rows")
        check("closed-roof games have wind zeroed",
              bool((wx[wx.is_closed == 1].wind_out.abs() < 1e-9).all()))
        check("density index in a sane range",
              bool(o.air_density_index.between(0.70, 1.15).all()),
              f"[{o.air_density_index.min():.3f}, {o.air_density_index.max():.3f}]")

        # Independent cross-check against MLB's own readings.
        m = o.dropna(subset=["mlb_temp_f", "temp_f"])
        if len(m) > 1000:
            c = float(np.corrcoef(m.temp_f, m.mlb_temp_f)[0, 1])
            check("reanalysis temp matches MLB official (r>0.9)", c > 0.90,
                  f"r={c:.4f}, MAE {abs(m.temp_f-m.mlb_temp_f).mean():.2f}F")
        mo = o.dropna(subset=["mlb_wind_out", "wind_out"])
        mo = mo[mo.mlb_wind_out.abs() > 1]
        if len(mo) > 1000:
            agree = float((np.sign(mo.wind_out) == np.sign(mo.mlb_wind_out)).mean())
            check("derived wind direction agrees with MLB (>70%)", agree > 0.70,
                  f"{100*agree:.1f}% sign agreement")

        # Coors must be the thinnest air in the league.
        if "elevation_m" in o:
            hi = o.loc[o.elevation_m.idxmax()]
            check("highest-elevation park has lowest density",
                  hi.air_density_index < o.air_density_index.median() - 0.05,
                  f"elev {hi.elevation_m:.0f}m -> index {hi.air_density_index:.3f}")

    print("\n" + "=" * 66)
    print(f"RESULT: {len(PASS)}/{len(PASS)+len(FAIL)} passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    print("=" * 66)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
