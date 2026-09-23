"""Fetch archived ECMWF weather runs known at a past decision time."""
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import math
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

from storage import atomic_write

UTC = dt.timezone.utc
LOCATIONS = {
    "turbine-1": (43.645150, 78.535604),
    "turbine-2": (43.643198, 78.538828),
}
API = "https://single-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ("wind_speed_10m", "wind_speed_100m", "temperature_2m")


def validate_result(saved: dict, day: dt.date, turbine: str, url: str,
                    decision: dt.datetime, run: dt.datetime, *,
                    locations: dict | None = None) -> None:
    """Apply the same chronology and value checks to cache and fresh responses."""
    if not isinstance(saved, dict):
        raise ValueError(f"weather response is not an object: {day} {turbine}")
    if (saved.get("request_url") != url or saved.get("turbine") != turbine
            or saved.get("model_run_utc") != run.strftime("%Y-%m-%dT%H:%M:%SZ")
            or saved.get("decision_time_utc") != decision.strftime("%Y-%m-%dT%H:%M:%SZ")):
        raise ValueError(f"weather provenance mismatch: {day} {turbine}")
    lat, lon = (LOCATIONS if locations is None else locations)[turbine]
    if saved.get("coordinates") != {"latitude": lat, "longitude": lon}:
        raise ValueError(f"weather coordinates mismatch: {day} {turbine}")
    units = saved.get("units") or {}
    if not isinstance(units, dict):
        raise ValueError(f"weather units are invalid: {day} {turbine}")
    if any(units.get(name) not in ("m/s", "ms") for name in ("wind_speed_10m", "wind_speed_100m")):
        raise ValueError(f"weather wind speed must be m/s: {day} {turbine}")
    records = saved.get("forecast")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError(f"weather must contain 48 hours: {day} {turbine}")
    for hour, row in enumerate(records):
        expected = decision.replace(tzinfo=None) + dt.timedelta(hours=hour)
        try:
            stamp = dt.datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
            if stamp.tzinfo is not None:
                stamp = stamp.astimezone(UTC).replace(tzinfo=None)
            values = [row[name] for name in VARIABLES]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid weather hour {hour}: {day} {turbine}") from exc
        if stamp != expected or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                    for value in values) or any(value < 0 for value in values[:2]):
            raise ValueError(f"invalid weather hour {hour}: {day} {turbine}")


def run_for(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Use yesterday's 18 UTC run at today's 06 UTC decision point."""
    decision = dt.datetime.combine(day, dt.time(6), tzinfo=UTC)
    return decision, decision - dt.timedelta(hours=12)


def retrieve(day: dt.date, turbine: str, cache_dir: pathlib.Path, refresh: bool = False,
             *, locations: dict | None = None) -> dict:
    selected_locations = LOCATIONS if locations is None else locations
    if turbine not in selected_locations:
        raise ValueError(f"unknown turbine: {turbine}")
    decision, run = run_for(day)
    lat, lon = selected_locations[turbine]
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(VARIABLES),
        "models": "ecmwf_ifs",
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "timezone": "UTC",
        "forecast_days": 3,
        "wind_speed_unit": "ms",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{day.isoformat()}-{turbine}.json"
    if path.exists() and not refresh:
        saved = json.loads(path.read_text())
        validate_result(saved, day, turbine, url, decision, run,
                        locations=selected_locations)
        return saved
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = json.load(response)
    except http.client.HTTPException as exc:
        raise RuntimeError(f"incomplete or invalid weather HTTP response: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError("weather API response is not an object")
    if body.get("error"):
        raise RuntimeError(body.get("reason", "weather API error"))
    hourly = body.get("hourly") or {}
    times = hourly.get("time") or []
    start = decision.replace(tzinfo=None)
    records = []
    for index, stamp in enumerate(times):
        moment = dt.datetime.fromisoformat(stamp)
        if start <= moment < start + dt.timedelta(hours=48):
            record = {"time_utc": stamp}
            for variable in VARIABLES:
                values = hourly.get(variable) or []
                record[variable] = values[index] if index < len(values) else None
            records.append(record)
    expected = [start + dt.timedelta(hours=i) for i in range(48)]
    if len(records) != 48 or any(
        dt.datetime.fromisoformat(row["time_utc"]) != expected[i]
        or any(row[v] is None for v in VARIABLES)
        for i, row in enumerate(records)
    ):
        raise RuntimeError(f"missing or invalid hourly forecast: {day} {turbine}")
    result = {
        "source": "Open-Meteo Single Runs API / ECMWF IFS HRES 9km",
        "request_url": url,
        "model_run_utc": run.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision_time_utc": decision.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assumed_publication_delay_hours": 6,
        "turbine": turbine,
        "coordinates": {"latitude": lat, "longitude": lon},
        "units": {v: (body.get("hourly_units") or {}).get(v) for v in VARIABLES},
        "forecast": records,
    }
    validate_result(result, day, turbine, url, decision, run,
                    locations=selected_locations)
    if path.exists():
        try:
            if json.loads(path.read_text()) == result:
                return result
        except (OSError, ValueError):
            pass  # A damaged cache is replaced only after fresh data validates.
    atomic_write(path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="decision date, YYYY-MM-DD, UTC")
    parser.add_argument("--days", type=int, default=1, help="number of daily forecasts")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"))
    parser.add_argument("--refresh", action="store_true", help="fetch again when inputs are updated")
    args = parser.parse_args()
    try:
        day = dt.date.fromisoformat(args.date)
        if not 1 <= args.days <= 366:
            raise ValueError("--days must be between 1 and 366")
        for offset in range(args.days):
            current = day + dt.timedelta(days=offset)
            for turbine in LOCATIONS:
                result = retrieve(current, turbine, args.cache_dir, refresh=args.refresh)
                print(f"{current} {turbine}: {len(result['forecast'])} hours; "
                      f"run {result['model_run_utc']}; "
                      f"saved {args.cache_dir / (current.isoformat() + '-' + turbine + '.json')}")
    except (ValueError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        print(f"forecast error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
