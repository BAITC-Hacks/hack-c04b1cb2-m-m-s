"""Train a reproducible hourly wind-power curve and apply it to archived forecasts.

Only Python's standard library is required. Training uses complete six-sample
hours from each turbine's ten-minute CSV, up to and including --cutoff. The
CSV's time zone and measurement height are undocumented. Prediction uses
forecast wind_speed_100m (m/s), so sensor/forecast domain shift remains a
material source of error; validate this model on historical forecast runs.

Examples:
    python3 power_model.py train --input-dir /path/to/originals \
        --cutoff 2025-12-31 --output model.json
    python3 power_model.py predict --model model.json \
        --weather weather-cache/2026-02-01-turbine-1.json \
        --output prediction.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys

FIELDS = {
    "time": "Статистическое время",
    "wind": "Средняя скорость ветра(m/s)",
    "power": "Нормализованная активная мощность",
    "temperature": "Средняя температура окружающей среды(°C)",
}
BIN_WIDTH = 0.5
MAX_WIND = 30.0
SMOOTHING_BANDWIDTH = 0.75


def parse_row(row: dict) -> tuple[dt.datetime, float, float, float]:
    stamp = dt.datetime.strptime(row[FIELDS["time"]], "%Y-%m-%d %H:%M:%S")
    wind, power, temp = (float(row[FIELDS[key]]) for key in ("wind", "power", "temperature"))
    if not all(math.isfinite(x) for x in (wind, power, temp)):
        raise ValueError("non-finite measurement")
    if not (0 <= wind <= MAX_WIND and 0 <= power <= 1 and -80 <= temp <= 80):
        raise ValueError("measurement outside expected range")
    if stamp.second or stamp.minute not in (0, 10, 20, 30, 40, 50):
        raise ValueError("measurement is not on a ten-minute boundary")
    return stamp, wind, power, temp


def load_hourly(path: pathlib.Path, cutoff: dt.date) -> tuple[list[tuple[float, float, float]], dict]:
    buckets: dict[dt.datetime, dict[int, tuple[float, float, float]]] = collections.defaultdict(dict)
    stats = collections.Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS.values()).issubset(reader.fieldnames or []):
            raise ValueError(f"missing expected CSV columns in {path}")
        for row in reader:
            stats["input_rows"] += 1
            try:
                stamp, wind, power, temp = parse_row(row)
            except (ValueError, KeyError, TypeError):
                stats["invalid_rows"] += 1
                continue
            if stamp.date() > cutoff:
                stats["rows_after_cutoff"] += 1
                continue
            hour = stamp.replace(minute=0, second=0)
            if stamp.minute in buckets[hour]:
                stats["duplicate_rows"] += 1
                continue
            buckets[hour][stamp.minute] = (wind, power, temp)
    hourly = []
    for observations in buckets.values():
        if len(observations) != 6:
            stats["incomplete_hours"] += 1
            continue
        hourly.append(tuple(sum(values[i] for values in observations.values()) / 6 for i in range(3)))
    stats["complete_hours"] = len(hourly)
    if not hourly:
        raise ValueError(f"no complete hours at or before {cutoff} in {path}")
    return hourly, dict(stats)


def fit_curve(hourly: list[tuple[float, float, float]]) -> dict:
    """Gaussian-smooth empirical half-m/s bins; support is weighted by sample count."""
    bins: dict[int, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    for wind, power, _temp in hourly:
        index = min(int(wind / BIN_WIDTH), int(MAX_WIND / BIN_WIDTH))
        bins[index][0] += power
        bins[index][1] += 1
    centers = {i: (i + 0.5) * BIN_WIDTH for i in bins}
    values = []
    for i in range(int(MAX_WIND / BIN_WIDTH) + 1):
        center = (i + 0.5) * BIN_WIDTH
        weights = {j: count * math.exp(-0.5 * ((center - centers[j]) / SMOOTHING_BANDWIDTH) ** 2)
                   for j, (_total, count) in bins.items()}
        denominator = sum(weights.values())
        value = sum(weights[j] * (bins[j][0] / bins[j][1]) for j in bins) / denominator if denominator else 0.0
        values.append(round(max(0.0, min(1.0, value)), 6))
    return {"bin_width_ms": BIN_WIDTH, "max_wind_ms": MAX_WIND,
            "smoothing_bandwidth_ms": SMOOTHING_BANDWIDTH, "values": values,
            "observed_bin_counts": {str(i): int(v[1]) for i, v in sorted(bins.items())}}


def train(input_dir: pathlib.Path, cutoff: dt.date) -> dict:
    turbines = {}
    for turbine in ("turbine-1", "turbine-2"):
        path = input_dir / f"{turbine}.csv"
        hourly, stats = load_hourly(path, cutoff)
        turbines[turbine] = {
            "source_csv": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_quality": stats,
            "training_mean_normalized_power": round(sum(r[1] for r in hourly) / len(hourly), 6),
            "hourly_mean_temperature_c": round(sum(r[2] for r in hourly) / len(hourly), 3),
            "curve": fit_curve(hourly),
        }
    return {
        "model_type": "gaussian_smoothed_empirical_wind_power_curve",
        "model_version": 1,
        "trained_through_inclusive": cutoff.isoformat(),
        "training_granularity": "complete hours of six ten-minute observations",
        "target": "normalized active power, range 0..1",
        "training_wind": "turbine sensor mean wind speed, m/s; measurement height unspecified",
        "prediction_wind": "Open-Meteo archived forecast wind_speed_100m, m/s",
        "csv_timestamp_timezone": "unspecified; cutoff applied to CSV calendar date",
        "limitations": ["sensor wind and 100m forecast wind have uncalibrated domain shift",
                        "temperature is aggregated for quality checks but not used as a model feature"],
        "turbines": turbines,
    }


def curve_value(curve: dict, wind: float) -> float:
    if not math.isfinite(wind) or wind < 0:
        raise ValueError("invalid forecast wind speed")
    width = curve["bin_width_ms"]
    values = curve["values"]
    # Linear interpolation between adjacent smoothed bin centers.
    position = max(0.0, min(len(values) - 1.0, wind / width - 0.5))
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] * (1 - (position - lo)) + values[hi] * (position - lo), 6)


def predict(model: dict, weather: dict) -> dict:
    turbine = weather.get("turbine")
    if turbine not in model["turbines"]:
        raise ValueError(f"weather turbine not in model: {turbine}")
    if weather.get("units", {}).get("wind_speed_100m") not in ("m/s", "ms"):
        raise ValueError("weather wind_speed_100m must be in m/s")
    rows = weather.get("forecast")
    if not isinstance(rows, list) or len(rows) not in (24, 48):
        raise ValueError("weather forecast must contain 24 or 48 hourly records")
    curve = model["turbines"][turbine]["curve"]
    width, values = curve["bin_width_ms"], curve["values"]
    if (not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0
            or not isinstance(values, list) or not values
            or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in values)):
        raise ValueError(f"invalid power curve: {turbine}")
    wind_scale = model["turbines"][turbine].get("forecast_wind_scale", 1.0)
    if not math.isfinite(wind_scale) or not 0.5 <= wind_scale <= 1.5:
        raise ValueError("invalid forecast wind calibration")
    result = []
    previous = None
    for row in rows:
        time = dt.datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
        if previous is not None and time - previous != dt.timedelta(hours=1):
            raise ValueError("forecast timestamps are not consecutive hourly records")
        previous = time
        wind = float(row["wind_speed_100m"])
        result.append({"time_utc": row["time_utc"], "wind_speed_100m_ms": wind,
                       "normalized_power": curve_value(curve, wind * wind_scale)})
    return {
        "turbine": turbine,
        "decision_time_utc": weather.get("decision_time_utc"),
        "weather_model_run_utc": weather.get("model_run_utc"),
        "weather_request_url": weather.get("request_url"),
        "power_model_type": model["model_type"],
        "power_model_trained_through_inclusive": model["trained_through_inclusive"],
        "forecast_wind_scale": wind_scale,
        "forecast": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="fit separate power curves for two turbine CSVs")
    training.add_argument("--input-dir", type=pathlib.Path, required=True)
    training.add_argument("--cutoff", type=dt.date.fromisoformat, required=True,
                          help="last included CSV calendar date, YYYY-MM-DD")
    training.add_argument("--output", type=pathlib.Path, required=True)
    prediction = commands.add_parser("predict", help="apply model to a cached archived weather forecast")
    prediction.add_argument("--model", type=pathlib.Path, required=True)
    prediction.add_argument("--weather", type=pathlib.Path, required=True)
    prediction.add_argument("--output", type=pathlib.Path,
                            help="JSON path; omit to write JSON to stdout")
    args = parser.parse_args()
    try:
        if args.command == "train":
            payload = train(args.input_dir, args.cutoff)
            output = args.output
        else:
            payload = predict(json.loads(args.model.read_text()), json.loads(args.weather.read_text()))
            output = args.output
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized)
            print(f"saved {output}")
        else:
            print(serialized, end="")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"power model error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
