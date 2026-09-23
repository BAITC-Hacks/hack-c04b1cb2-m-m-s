"""Fetch archived ECMWF weather runs known at a past decision time."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

UTC = dt.timezone.utc
LOCATIONS = {
    "turbine-1": (43.645150, 78.535604),
    "turbine-2": (43.643198, 78.538828),
}
API = "https://single-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ("wind_speed_10m", "temperature_2m")


def run_for(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Use yesterday's 18 UTC run at today's 06 UTC decision point."""
    decision = dt.datetime.combine(day, dt.time(6), tzinfo=UTC)
    return decision, decision - dt.timedelta(hours=12)


def retrieve(day: dt.date, turbine: str, cache_dir: pathlib.Path) -> dict:
    if turbine not in LOCATIONS:
        raise ValueError(f"unknown turbine: {turbine}")
    decision, run = run_for(day)
    lat, lon = LOCATIONS[turbine]
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
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("request_url") != url:
            raise ValueError(f"cached request differs: {path}")
        return saved
    with urllib.request.urlopen(url, timeout=30) as response:
        body = json.load(response)
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
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="decision date, YYYY-MM-DD, UTC")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"))
    args = parser.parse_args()
    try:
        day = dt.date.fromisoformat(args.date)
        for turbine in LOCATIONS:
            result = retrieve(day, turbine, args.cache_dir)
            print(f"{turbine}: {len(result['forecast'])} hours; run {result['model_run_utc']}; "
                  f"saved {args.cache_dir / (args.date + '-' + turbine + '.json')}")
    except (ValueError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        print(f"forecast error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
