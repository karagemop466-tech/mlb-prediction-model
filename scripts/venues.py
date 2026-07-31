"""Venue geometry, elevation and roof status from the MLB Stats API.

Everything here is retrieved from statsapi.mlb.com, not assumed. The fields that
matter for weather physics:

    azimuthAngle  bearing (degrees) from home plate toward center field. This is
                  what lets a compass wind direction be converted into a
                  stadium-relative vector (out to CF / in from CF / crosswind).
    elevation     meters above sea level. Drives air density directly.
    roofType      Open / Retractable / Dome. Wind and humidity effects must be
                  suppressed for closed roofs, otherwise the model attributes
                  outdoor weather to indoor games.
    dimensions    outfield distances, used to sanity-check park run environment.

Writes data/proc/venues.json.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "proc"

API = ("https://statsapi.mlb.com/api/v1/venues?sportId=1"
       "&hydrate=location,fieldInfo,timezone")

# Retrosheet park codes -> MLB venue ids, for joining to the historical games
# table. Only parks used by MLB teams 2016-2026 are needed.
RETRO_TO_VENUE = {
    "ANA01": 1, "ARL02": 5325, "ATL03": 4705, "BAL12": 2, "BOS07": 3,
    "CHI11": 4, "CHI12": 17, "CIN09": 2602, "CLE08": 5, "DEN02": 19,
    "DET05": 2394, "HOU03": 2392, "KAN06": 7, "LOS03": 22, "MIA02": 4169,
    "MIL06": 32, "MIN04": 3312, "NYC20": 3289, "NYC21": 3313, "OAK01": 10,
    "PHI13": 2681, "PHO01": 15, "PIT08": 31, "SAN02": 2680, "SEA03": 680,
    "SFO03": 2395, "STL10": 2889, "STP01": 12, "TOR02": 14, "WAS11": 3309,
    "SAC01": 2529, "ARL01": 13, "NYC17": 3313, "MIA01": 4169,
    # Globe Life Field (2020+), Tropicana, Truist/SunTrust, and the 2020-21
    # temporary parks. Verified against MLB venue ids.
    "ARL03": 5325, "TAM02": 12, "ATL02": 4705, "BUF05": 5119,
    "SEA02": 680, "DUN01": 2536, "WAS10": 3309,
}


def build_resolver(venues: dict) -> dict:
    """Map every park label seen in the games table to a venue id.

    Retrosheet seasons use 5-character park codes; the 2026 rows come from the
    MLB API and carry venue NAMES instead. Both must resolve.
    """
    by_name = {x["name"]: k for k, x in venues.items()}
    resolver = dict(RETRO_TO_VENUE)
    for name, vid in by_name.items():
        resolver[name] = int(vid)
    return resolver


def fetch() -> dict:
    req = urllib.request.Request(API, headers={"User-Agent": "research/1.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=90).read())

    out = {}
    for v in data.get("venues", []):
        loc = v.get("location") or {}
        coords = loc.get("defaultCoordinates") or {}
        fi = v.get("fieldInfo") or {}
        if not coords:
            continue
        out[str(v["id"])] = {
            "id": v["id"],
            "name": v.get("name"),
            "lat": coords.get("latitude"),
            "lon": coords.get("longitude"),
            "azimuth": loc.get("azimuthAngle"),
            "elevation_m": loc.get("elevation"),
            "tz": (v.get("timeZone") or {}).get("id"),
            "roof": fi.get("roofType"),
            "capacity": fi.get("capacity"),
            "cf": fi.get("center"),
            "lf_line": fi.get("leftLine"),
            "rf_line": fi.get("rightLine"),
        }
    return out


def main() -> None:
    v = fetch()
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "venues.json").write_text(json.dumps(v, indent=2))

    with_az = [x for x in v.values() if x["azimuth"] is not None]
    with_el = [x for x in v.values() if x["elevation_m"] is not None]
    roofs = {}
    for x in v.values():
        roofs[x["roof"]] = roofs.get(x["roof"], 0) + 1

    print(f"venues: {len(v)}  with azimuth: {len(with_az)}  "
          f"with elevation: {len(with_el)}")
    print(f"roof types: {roofs}")

    print("\nHighest elevation venues (air density matters most here):")
    for x in sorted(with_el, key=lambda z: -z["elevation_m"])[:6]:
        print(f"  {x['name']:<28} {x['elevation_m']:>6} m  "
              f"azimuth {x['azimuth']}  roof {x['roof']}")

    print("\nMapped Retrosheet park codes:", len(RETRO_TO_VENUE))
    print(f"-> {PROC / 'venues.json'}")


if __name__ == "__main__":
    main()
