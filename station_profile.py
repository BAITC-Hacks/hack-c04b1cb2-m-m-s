"""Validate a two-turbine station profile without inferring its coordinates."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re

DEFAULT_STATION = {
    "id": "competition",
    "name": "Ветропарк конкурса",
    "locations": {
        "turbine-1": [43.645150, 78.535604],
        "turbine-2": [43.643198, 78.538828],
    },
}
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_TURBINES = {"turbine-1", "turbine-2"}


def validate_station(value: dict) -> dict:
    """Return only normalized, validated profile fields."""
    if not isinstance(value, dict) or set(value) != {"id", "name", "locations"}:
        raise ValueError("station must contain only id, name, and locations")
    station_id = value["id"]
    if not isinstance(station_id, str) or _SLUG.fullmatch(station_id) is None:
        raise ValueError("station id must be a lowercase slug of 1..64 characters")
    name = value["name"]
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
        raise ValueError("station name must contain 1..120 nonblank characters")
    locations = value["locations"]
    if not isinstance(locations, dict) or set(locations) != _TURBINES:
        raise ValueError("station must specify exactly turbine-1 and turbine-2")
    normalized = {}
    for turbine in ("turbine-1", "turbine-2"):
        pair = locations[turbine]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{turbine} must have latitude and longitude")
        latitude, longitude = pair
        if (type(latitude) not in (int, float) or type(longitude) not in (int, float)
                or not math.isfinite(latitude) or not math.isfinite(longitude)
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
            raise ValueError(f"{turbine} coordinates are invalid")
        normalized[turbine] = [float(latitude), float(longitude)]
    return {"id": station_id, "name": name.strip(), "locations": normalized}


def load_station(path: pathlib.Path) -> dict:
    """Read a JSON profile and return its validated normalized form."""
    return validate_station(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def station_digest(profile: dict) -> str:
    normalized = validate_station(profile)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
