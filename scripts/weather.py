"""Historical hourly weather at each ballpark, from the Open-Meteo archive.

Open-Meteo's archive API is free, needs no key, and is built on ERA5 / ECMWF
reanalysis. It returns temperature, relative humidity, SURFACE PRESSURE (not
sea-level-adjusted), wind speed, wind direction and WIND GUSTS at hourly
resolution, plus the model grid elevation.

Surface pressure is the important one. Air density depends on actual pressure at
the park, and that is what encodes elevation: Coors Field sits near 848 hPa
versus roughly 1013 hPa at sea level. Using sea-level pressure would erase the
single largest weather effect in baseball.

Data is fetched per venue per season (one request covers a whole season of
hourly data) and cached, so the whole 2016-2026 pull is ~30 venues x 11 seasons.

MLB's own weather field is also parsed separately (see parse_mlb_weather) --
it is stadium-relative ("Out To CF"), which is exactly the physically relevant
axis, and serves as an independent cross-check on the reanalysis wind.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"
RAW = ROOT / "data" / "raw" / "weather"

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ("temperature_2m,relative_humidity_2m,surface_pressure,"
          "wind_speed_10m,wind_gusts_10m,wind_direction_10m,precipitation,"
          "cloud_cover,dew_point_2m")


# ------------------------------------------------------------------ fetching
def fetch_venue_season(vid: str, lat: float, lon: float, tz: str,
                       year: int, retries: int = 3) -> pd.DataFrame | None:
    """One season of hourly weather for one venue. Cached to parquet."""
    path = RAW / f"v{vid}_{year}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)

    # The archive lags real time by a few days, so cap the end date. Requesting
    # a future end_date returns HTTP 400 and loses the whole season.
    import datetime as _dt
    cap = _dt.date.today() - _dt.timedelta(days=6)
    end = min(_dt.date(year, 11, 15), cap)
    if end < _dt.date(year, 3, 1):
        return None
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-03-01", "end_date": end.isoformat(),
        "hourly": HOURLY,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": tz or "UTC",
    }
    url = f"{ARCHIVE}?{urllib.parse.urlencode(params)}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=180).read())
            h = d.get("hourly")
            if not h:
                return None
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"])
            df["venue_elevation_m"] = d.get("elevation")
            RAW.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            return df
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            print(f"    v{vid} {year}: HTTP {e.code}")
            return None
        except Exception as e:
            if attempt == retries - 1:
                print(f"    v{vid} {year}: {type(e).__name__}")
                return None
            time.sleep(5 * (attempt + 1))
    return None


def download(years=range(2016, 2027)) -> None:
    venues = json.loads((PROC / "venues.json").read_text())
    games = pd.read_parquet(PROC / "games.parquet")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from venues import build_resolver
    resolver = build_resolver(venues)
    used = set(games["park"].map(resolver).dropna().astype(int))

    todo = [(k, v) for k, v in venues.items() if int(k) in used]
    print(f"{len(todo)} venues x {len(list(years))} seasons")
    for vid, v in todo:
        if v["lat"] is None:
            continue
        got = 0
        for yr in years:
            df = fetch_venue_season(vid, v["lat"], v["lon"], v["tz"], yr)
            if df is not None:
                got += 1
            time.sleep(0.4)
        print(f"  {v['name'][:32]:<34} {got}/{len(list(years))} seasons")


FORECAST = "https://api.open-meteo.com/v1/forecast"


def fetch_forecast(lat: float, lon: float, tz: str, days: int = 3):
    """Live forecast for upcoming games. Same schema as the archive.

    The archive lags several days behind real time, so predictions for today's
    slate must come from the forecast endpoint.
    """
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": HOURLY,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": tz or "UTC",
        "forecast_days": days,
    }
    url = f"{FORECAST}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
        d = json.loads(urllib.request.urlopen(req, timeout=90).read())
        h = d.get("hourly")
        if not h:
            return None
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"])
        df["venue_elevation_m"] = d.get("elevation")
        return df
    except Exception:
        return None


# ------------------------------------------------------------------ physics
def air_density(temp_f: float, rh_pct: float, pressure_hpa: float) -> float:
    """Air density in kg/m^3 from temperature, relative humidity and pressure.

    Uses the standard formulation: dry-air and water-vapour partial pressures
    combined via the ideal gas law, with saturation vapour pressure from the
    Tetens/Magnus approximation. Humid air is LESS dense than dry air at the
    same temperature and pressure, because water vapour (18 g/mol) is lighter
    than dry air (~29 g/mol) -- a fact that is widely stated backwards.

        rho = (Pd / (Rd * T)) + (Pv / (Rv * T))
    """
    t_c = (temp_f - 32.0) * 5.0 / 9.0
    t_k = t_c + 273.15
    # Saturation vapour pressure (hPa), Magnus form
    es = 6.1078 * 10.0 ** (7.5 * t_c / (237.3 + t_c))
    pv = (rh_pct / 100.0) * es          # actual vapour pressure, hPa
    pd_ = pressure_hpa - pv             # dry air partial pressure, hPa
    rd, rv = 287.058, 461.495           # J/(kg*K)
    return (pd_ * 100.0) / (rd * t_k) + (pv * 100.0) / (rv * t_k)


SEA_LEVEL_STD_DENSITY = air_density(59.0, 0.0, 1013.25)   # ~1.225 kg/m^3


def density_index(temp_f, rh_pct, pressure_hpa) -> float:
    """Air density relative to ISA sea level. <1 means thinner air = more carry."""
    return air_density(temp_f, rh_pct, pressure_hpa) / SEA_LEVEL_STD_DENSITY


def wind_out_component(wind_dir_deg: float, wind_mph: float,
                       azimuth_deg: float) -> float:
    """Component of wind blowing from home plate toward center field, in mph.

    Meteorological wind direction is the direction the wind is coming FROM.
    `azimuth_deg` is the bearing from home plate toward center field.

    A wind blowing out to center travels along the azimuth bearing, so it comes
    FROM the opposite bearing (azimuth + 180). Therefore:

        out_component = wind_speed * cos(wind_from - (azimuth + 180))

    Positive = blowing out toward center (helps fly balls carry).
    Negative = blowing in from center (suppresses).
    """
    if any(pd.isna(x) for x in (wind_dir_deg, wind_mph, azimuth_deg)):
        return np.nan
    blowing_from_out = (azimuth_deg + 180.0) % 360.0
    delta = np.radians(wind_dir_deg - blowing_from_out)
    return float(wind_mph * np.cos(delta))


def wind_cross_component(wind_dir_deg: float, wind_mph: float,
                         azimuth_deg: float) -> float:
    """Crosswind magnitude (mph), perpendicular to the plate-to-CF axis."""
    if any(pd.isna(x) for x in (wind_dir_deg, wind_mph, azimuth_deg)):
        return np.nan
    blowing_from_out = (azimuth_deg + 180.0) % 360.0
    delta = np.radians(wind_dir_deg - blowing_from_out)
    return float(abs(wind_mph * np.sin(delta)))


# ------------------------------------------------- MLB's own weather string
MLB_WIND_RE = re.compile(r"(\d+)\s*mph,?\s*(.*)", re.I)

# MLB reports wind relative to the stadium. Map each label to a component along
# the plate->CF axis, as a fraction of wind speed.
MLB_DIR_FACTOR = {
    "out to cf": 1.0, "out to lf": 0.71, "out to rf": 0.71,
    "in from cf": -1.0, "in from lf": -0.71, "in from rf": -0.71,
    "l to r": 0.0, "r to l": 0.0, "varies": 0.0, "calm": 0.0, "none": 0.0,
}


def parse_mlb_weather(w: dict | None) -> dict:
    """Parse MLB's weather dict into numeric fields.

    Returns temp_f, condition, wind_mph, wind_label, mlb_wind_out (stadium
    relative component) and roof_closed.
    """
    out = {"mlb_temp_f": np.nan, "mlb_condition": None, "mlb_wind_mph": np.nan,
           "mlb_wind_label": None, "mlb_wind_out": np.nan, "roof_closed": 0}
    if not w:
        return out
    cond = (w.get("condition") or "").strip()
    out["mlb_condition"] = cond or None
    if cond.lower() in ("roof closed", "dome"):
        out["roof_closed"] = 1
    try:
        out["mlb_temp_f"] = float(w.get("temp"))
    except (TypeError, ValueError):
        pass
    m = MLB_WIND_RE.match((w.get("wind") or "").strip())
    if m:
        out["mlb_wind_mph"] = float(m.group(1))
        label = m.group(2).strip().lower().rstrip(".")
        out["mlb_wind_label"] = label or None
        f = MLB_DIR_FACTOR.get(label)
        if f is not None:
            out["mlb_wind_out"] = out["mlb_wind_mph"] * f
    return out


# ------------------------------------------------------------------ assembly
def load_hourly(vid: int, year: int) -> pd.DataFrame | None:
    path = RAW / f"v{vid}_{year}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def main() -> None:
    import sys
    if "--download" in sys.argv:
        yrs = [int(a) for a in sys.argv if a.isdigit()] or list(range(2016, 2027))
        download(yrs)
        return

    print("Air density physics self-check")
    print(f"  ISA sea level (59F, 0% RH, 1013 hPa): "
          f"{air_density(59, 0, 1013.25):.4f} kg/m3  (expect ~1.2250)")
    print(f"  Coors Field summer (85F, 30% RH, 848 hPa): "
          f"{air_density(85, 30, 848):.4f}  index "
          f"{density_index(85, 30, 848):.4f}")
    print(f"  Fenway summer      (85F, 70% RH, 1013 hPa): "
          f"{air_density(85, 70, 1013):.4f}  index "
          f"{density_index(85, 70, 1013):.4f}")
    print(f"  Cold sea level     (45F, 60% RH, 1015 hPa): "
          f"{air_density(45, 60, 1015):.4f}  index "
          f"{density_index(45, 60, 1015):.4f}")
    print("\n  Humid air is LESS dense than dry air at equal T and P:")
    print(f"    85F 10% RH: {air_density(85, 10, 1013):.4f}")
    print(f"    85F 90% RH: {air_density(85, 90, 1013):.4f}")

    print("\nWind projection self-check (azimuth 0 = CF due north)")
    for wd, label in ((180, "from south = blowing OUT to CF"),
                      (0, "from north = blowing IN from CF"),
                      (90, "from east = pure crosswind"),
                      (270, "from west = pure crosswind")):
        print(f"  wind from {wd:>3} deg, 10 mph: "
              f"out={wind_out_component(wd, 10, 0):+.2f} "
              f"cross={wind_cross_component(wd, 10, 0):.2f}   {label}")


if __name__ == "__main__":
    main()
