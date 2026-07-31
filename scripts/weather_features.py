"""Join hourly weather to each game at its actual first-pitch hour.

Two independent weather sources are combined:

  1. Open-Meteo reanalysis  -- continuous physical variables (temperature,
     humidity, SURFACE pressure, wind speed/direction, GUSTS). Gives air density
     and a compass wind vector that is projected onto each park's plate->CF axis
     using the MLB venue azimuth.

  2. MLB Stats API weather  -- the official stadium-relative reading
     ("12 mph, Out To CF") plus the roof status. Used as an independent
     cross-check and to detect closed roofs.

Roof handling is not cosmetic. Roughly 15% of games are played under a closed
roof or in a dome. Attributing outdoor wind to those games would inject pure
noise, so all wind terms are zeroed and flagged.

Output: data/proc/weather_games.parquet, one row per game_id.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from venues import build_resolver
from weather import (density_index, load_hourly, parse_mlb_weather,
                     wind_cross_component, wind_out_component)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
RAW_MLB = ROOT / "data" / "raw" / "mlb_weather"


# ------------------------------------------------------- MLB weather scrape
def fetch_mlb_weather(start: str, end: str) -> pd.DataFrame:
    """Schedule-level weather for a date range (one request per ~30 days)."""
    url = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R"
           f"&startDate={start}&endDate={end}&hydrate=weather,venue,team")
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    rows = []
    for blk in d.get("dates", []):
        for g in blk.get("games", []):
            w = parse_mlb_weather(g.get("weather"))
            rows.append({
                "date": pd.Timestamp(g["officialDate"]),
                "game_pk": g["gamePk"],
                "venue_id": (g.get("venue") or {}).get("id"),
                "start_utc": g.get("gameDate"),
                "away_abbr": g["teams"]["away"]["team"].get("abbreviation"),
                "home_abbr": g["teams"]["home"]["team"].get("abbreviation"),
                "day_night": g.get("dayNight"),
                **w,
            })
    return pd.DataFrame(rows)


def download_mlb_weather(years=range(2016, 2027)) -> pd.DataFrame:
    RAW_MLB.mkdir(parents=True, exist_ok=True)
    frames = []
    for yr in years:
        path = RAW_MLB / f"mlb_wx_{yr}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        parts = []
        for m0, m1 in (("03-01", "05-15"), ("05-16", "07-31"),
                       ("08-01", "11-15")):
            try:
                parts.append(fetch_mlb_weather(f"{yr}-{m0}", f"{yr}-{m1}"))
            except Exception as e:
                print(f"  {yr} {m0}: {type(e).__name__}")
        if parts:
            df = pd.concat(parts, ignore_index=True).drop_duplicates("game_pk")
            df.to_parquet(path, index=False)
            frames.append(df)
            print(f"  MLB weather {yr}: {len(df)} games")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------ hourly lookup
def build() -> pd.DataFrame:
    games = pd.read_parquet(PROC / "games.parquet")
    venues = json.loads((PROC / "venues.json").read_text())
    resolver = build_resolver(venues)
    vmeta = {int(k): v for k, v in venues.items()}

    games = games.copy()
    games["venue_id"] = games["park"].map(resolver)
    games["date"] = pd.to_datetime(games["date"])

    # MLB weather (stadium-relative wind + roof status)
    mlb = download_mlb_weather()
    if not mlb.empty:
        mlb["date"] = pd.to_datetime(mlb["date"])
        mlb["start_hour_local"] = np.nan
        st = pd.to_datetime(mlb["start_utc"], errors="coerce", utc=True)
        mlb["start_utc_ts"] = st

    rows = []
    for (vid, yr), grp in games.dropna(subset=["venue_id"]).groupby(
            ["venue_id", games["date"].dt.year]):
        vid = int(vid)
        hourly = load_hourly(vid, int(yr))
        meta = vmeta.get(vid, {})
        az = meta.get("azimuth")
        roof = meta.get("roof")
        if hourly is None or hourly.empty:
            for _, g in grp.iterrows():
                rows.append({"game_id": g.game_id, "venue_id": vid,
                             "roof_type": roof, "has_weather": 0})
            continue

        hourly = hourly.copy()
        hourly["date"] = hourly["time"].dt.normalize()
        hourly["hour"] = hourly["time"].dt.hour
        elev = float(hourly["venue_elevation_m"].iloc[0]) \
            if "venue_elevation_m" in hourly else np.nan

        # Default first pitch: 19:00 local for night games, 13:00 for day.
        for _, g in grp.iterrows():
            hr = 19 if str(g.get("daynight", "N")).upper().startswith("N") else 13
            sel = hourly[(hourly["date"] == g["date"]) & (hourly["hour"] == hr)]
            if sel.empty:
                sel = hourly[hourly["date"] == g["date"]]
                if sel.empty:
                    rows.append({"game_id": g.game_id, "venue_id": vid,
                                 "roof_type": roof, "has_weather": 0})
                    continue
                sel = sel.iloc[[min(hr, len(sel) - 1)]]
            r = sel.iloc[0]

            temp = float(r["temperature_2m"])
            rh = float(r["relative_humidity_2m"])
            pres = float(r["surface_pressure"])
            wspd = float(r["wind_speed_10m"])
            wdir = float(r["wind_direction_10m"])
            gust = float(r["wind_gusts_10m"])

            di = density_index(temp, rh, pres)
            w_out = wind_out_component(wdir, wspd, az) if az is not None else np.nan
            w_cross = wind_cross_component(wdir, wspd, az) if az is not None else np.nan
            g_out = wind_out_component(wdir, gust, az) if az is not None else np.nan

            rows.append({
                "game_id": g.game_id, "venue_id": vid, "has_weather": 1,
                "roof_type": roof,
                "temp_f": temp, "humidity": rh, "pressure_hpa": pres,
                "dew_point_f": float(r.get("dew_point_2m", np.nan)),
                "precip": float(r.get("precipitation", 0) or 0),
                "cloud_cover": float(r.get("cloud_cover", np.nan)),
                "wind_mph": wspd, "wind_dir": wdir, "gust_mph": gust,
                "gust_excess": gust - wspd,
                "gust_ratio": gust / wspd if wspd > 0.5 else np.nan,
                "air_density_index": di,
                "wind_out": w_out, "wind_cross": w_cross, "gust_out": g_out,
                "elevation_m": elev,
                "azimuth": az,
            })

    wx = pd.DataFrame(rows)

    # Merge MLB's own reading + roof status by (date, teams)
    if not mlb.empty:
        gm = games[["game_id", "date", "home", "away"]].copy()
        from build_dataset import TEAM_MAP
        mlb["home_r"] = mlb["home_abbr"].map(lambda x: TEAM_MAP.get(x, x))
        mlb["away_r"] = mlb["away_abbr"].map(lambda x: TEAM_MAP.get(x, x))
        key = mlb.drop_duplicates(["date", "home_r", "away_r"])
        gm = gm.merge(
            key[["date", "home_r", "away_r", "mlb_temp_f", "mlb_condition",
                 "mlb_wind_mph", "mlb_wind_label", "mlb_wind_out", "roof_closed"]],
            left_on=["date", "home", "away"],
            right_on=["date", "home_r", "away_r"], how="left")
        wx = wx.merge(gm.drop(columns=["date", "home", "away", "home_r", "away_r"]),
                      on="game_id", how="left")

    # A closed roof means outdoor wind is irrelevant. Zero it and flag it.
    closed = (wx.get("roof_closed", 0) == 1)
    if "roof_type" in wx:
        closed = closed | (wx["roof_type"] == "Dome")
    wx["is_closed"] = closed.astype(int)
    for c in ("wind_out", "wind_cross", "gust_out", "gust_excess",
              "wind_mph", "gust_mph", "mlb_wind_out"):
        if c in wx:
            wx.loc[closed, c] = 0.0

    PROC.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(PROC / "weather_games.parquet", index=False)
    return wx


def main() -> None:
    wx = build()
    n = len(wx)
    print(f"\ngames with weather rows: {n:,}")
    print(f"  has_weather=1:    {int(wx.has_weather.sum()):,} "
          f"({100*wx.has_weather.mean():.1f}%)")
    print(f"  closed roof/dome: {int(wx.is_closed.sum()):,} "
          f"({100*wx.is_closed.mean():.1f}%)")
    open_air = wx[(wx.has_weather == 1) & (wx.is_closed == 0)]
    print(f"  open-air w/ wind: {len(open_air):,}")
    if len(open_air):
        print("\nOpen-air distributions:")
        for c in ("temp_f", "humidity", "pressure_hpa", "air_density_index",
                  "wind_mph", "gust_mph", "gust_excess", "wind_out", "wind_cross"):
            if c in open_air:
                s = open_air[c].dropna()
                if len(s):
                    print(f"  {c:<20} mean {s.mean():>8.3f}  sd {s.std():>7.3f}  "
                          f"[{s.min():>7.2f}, {s.max():>7.2f}]")
    print(f"\n-> {PROC / 'weather_games.parquet'}")


if __name__ == "__main__":
    main()
