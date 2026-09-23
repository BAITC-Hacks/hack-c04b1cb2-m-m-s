"""Run the wind-farm forecasting cycle for one or more historical decision days.

Each day: get archived weather for both turbine coordinates, use a power model
trained only on measurements before the decision day, predict 48 hourly values, check
the result, and save the output. ``--refresh-weather`` repeats the external
fetch when weather inputs are updated. The output is normalized power, not MWh.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys
import urllib.error

from forecast_weather import LOCATIONS, retrieve
from power_model import predict, train

LATEST_TRAINING_DATE = dt.date(2026, 1, 31)


def combine(day: dt.date, model: dict, weather: dict[str, dict]) -> dict:
    predicted = {turbine: predict(model, data) for turbine, data in weather.items()}
    a = predicted["turbine-1"]["forecast"]
    b = predicted["turbine-2"]["forecast"]
    if len(a) != 48 or len(b) != 48:
        raise ValueError("both turbines must have 48 forecast hours")
    combined = []
    for left, right in zip(a, b):
        if left["time_utc"] != right["time_utc"]:
            raise ValueError("turbine forecast timestamps differ")
        mean = round((left["normalized_power"] + right["normalized_power"]) / 2, 6)
        combined.append({"time_utc": left["time_utc"],
                         "turbine_1_normalized_power": left["normalized_power"],
                         "turbine_2_normalized_power": right["normalized_power"],
                         "farm_equal_capacity_mean_normalized_power": mean})
    values = [row["farm_equal_capacity_mean_normalized_power"] for row in combined]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("predicted normalized power outside 0..1")
    ramps = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
    largest_ramp = max(ramps)
    largest_ramp_index = ramps.index(largest_ramp) + 1
    low_hours = sum(value < 0.1 for value in values)
    signals = []
    if largest_ramp >= 0.2:
        signals.append(f"Изменение мощности ≥ 0,2 перед часом {combined[largest_ramp_index]['time_utc']} UTC")
    if low_hours >= 6:
        signals.append(f"Низкая прогнозная мощность (< 0,1) в {low_hours} часах")
    analysis = {
        "mean_normalized_power": round(statistics.mean(values), 6),
        "minimum_normalized_power": min(values),
        "maximum_normalized_power": max(values),
        "largest_hourly_ramp": round(largest_ramp, 6),
        "largest_hourly_ramp_ending_utc": combined[largest_ramp_index]["time_utc"],
        "low_power_hours_below_0_1": low_hours,
        "first_24h_mean": round(statistics.mean(values[:24]), 6),
        "second_24h_mean": round(statistics.mean(values[24:]), 6),
        "signals": signals,
        "checks": ["48 последовательных часов UTC", "время турбин согласовано",
                   "мощность в диапазоне 0–1"],
    }
    return {
        "forecast_date": day.isoformat(),
        "decision_time_utc": weather["turbine-1"]["decision_time_utc"],
        "training_cutoff_inclusive": model["trained_through_inclusive"],
        "forecast_wind_scale": {name: model["turbines"][name].get("forecast_wind_scale", 1.0)
                                for name in weather},
        "weather_run_utc": {name: data["model_run_utc"] for name, data in weather.items()},
        "weather_request_urls": {name: data["request_url"] for name, data in weather.items()},
        "output_unit": "normalized active power (0..1)",
        "farm_aggregation": "arithmetic mean of two turbine normalized outputs; equal-capacity assumption",
        "analysis": analysis,
        "forecast": combined,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", type=dt.date.fromisoformat, help="first decision date, YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=pathlib.Path,
                        help="directory containing turbine-1.csv and turbine-2.csv")
    source.add_argument("--model", type=pathlib.Path,
                        help="pretrained power-curve JSON; raw CSV not required")
    source.add_argument("--model-dir", type=pathlib.Path,
                        help="directory of power-curve-YYYY-MM-DD.json artifacts")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("predictions"))
    parser.add_argument("--refresh-weather", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.days <= 31:
        parser.error("--days must be between 1 and 31")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        models: dict[dt.date, dict] = {}
        for offset in range(args.days):
            day = args.date + dt.timedelta(days=offset)
            cutoff = min(day - dt.timedelta(days=1), LATEST_TRAINING_DATE)
            if cutoff not in models:
                if args.input_dir:
                    models[cutoff] = train(args.input_dir, cutoff)
                    (args.output_dir / f"power-model-{cutoff}.json").write_text(
                        json.dumps(models[cutoff], ensure_ascii=False, indent=2) + "\n")
                else:
                    path = (args.model_dir / f"power-curve-{cutoff}.json"
                            if args.model_dir else args.model)
                    models[cutoff] = json.loads(path.read_text())
                if dt.date.fromisoformat(models[cutoff]["trained_through_inclusive"]) > cutoff:
                    raise ValueError(f"model contains future measurements for {day}")
            weather = {turbine: retrieve(day, turbine, args.cache_dir,
                                         refresh=args.refresh_weather)
                       for turbine in LOCATIONS}
            output = combine(day, models[cutoff], weather)
            path = args.output_dir / f"{day.isoformat()}-forecast.json"
            path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
            print(f"{day}: 48 hours; mean normalized power "
                  f"{output['analysis']['mean_normalized_power']:.3f}; saved {path}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, urllib.error.URLError) as exc:
        print(f"forecast cycle error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
