"""Compare hourly archived-weather predictions with withheld January measurements.

CSV time zone is undocumented. Report several plausible UTC offsets instead of
silently choosing one from the held-out target data. January is never used to
fit the supplied model.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import pathlib
import statistics

from power_model import FIELDS, parse_row, predict, validate_model


def actual_hours(path: pathlib.Path, first: dt.date, last: dt.date) -> dict[dt.datetime, float]:
    buckets: dict[dt.datetime, dict[int, float]] = collections.defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                stamp, _wind, power, _temp = parse_row(row)
            except (ValueError, KeyError, TypeError):
                continue
            if first <= stamp.date() <= last:
                buckets[stamp.replace(minute=0)][stamp.minute] = power
    return {stamp: statistics.mean(minutes.values()) for stamp, minutes in buckets.items()
            if len(minutes) == 6}


def predicted_hours(model: dict, weather_dir: pathlib.Path, turbine: str,
                    first: dt.date, last: dt.date) -> dict[dt.datetime, float]:
    validate_model(model, first - dt.timedelta(days=1))
    result: dict[dt.datetime, tuple[dt.datetime, float]] = {}
    # Start both baseline and calibrated comparisons with the first evaluation
    # decision. A calibration ending yesterday was unavailable at yesterday's run.
    day = first
    while day <= last:
        path = weather_dir / f"{day}-{turbine}.json"
        payload = predict(model, json.loads(path.read_text()))
        decision = dt.datetime.fromisoformat(payload["decision_time_utc"].replace("Z", "+00:00"))
        for row in payload["forecast"]:
            time = dt.datetime.fromisoformat(row["time_utc"]).replace(tzinfo=dt.timezone.utc)
            if time not in result or decision > result[time][0]:
                result[time] = decision, row["normalized_power"]
        day += dt.timedelta(days=1)
    return {time: value for time, (_decision, value) in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--weather-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"))
    parser.add_argument("--first", type=dt.date.fromisoformat, default=dt.date(2026, 1, 1))
    parser.add_argument("--last", type=dt.date.fromisoformat, default=dt.date(2026, 1, 14))
    parser.add_argument("--offsets", type=int, nargs="+", default=[5, 6, 7],
                        help="hypothetical CSV local time minus UTC, in whole hours")
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    if args.first > args.last:
        parser.error("validation start must precede its end")
    try:
        validate_model(model, args.first - dt.timedelta(days=1))
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    report = {"model_cutoff": model["trained_through_inclusive"],
              "validation_dates": [str(args.first), str(args.last)], "turbines": {}}
    for turbine in ("turbine-1", "turbine-2"):
        measured = actual_hours(args.input_dir / f"{turbine}.csv", args.first, args.last)
        forecast = predicted_hours(model, args.weather_dir, turbine, args.first, args.last)
        offsets = {}
        for offset in args.offsets:
            pairs = [(predicted, actual) for local, actual in measured.items()
                     if (predicted := forecast.get((local - dt.timedelta(hours=offset)).replace(
                         tzinfo=dt.timezone.utc))) is not None]
            errors = [predicted - actual for predicted, actual in pairs]
            if not errors:
                raise ValueError(f"no comparable hours for {turbine}, offset {offset}")
            baseline = model["turbines"][turbine]["training_mean_normalized_power"]
            offsets[str(offset)] = {"hours": len(errors),
                                    "mae_normalized_power": round(statistics.mean(map(abs, errors)), 6),
                                    "rmse_normalized_power": round(math.sqrt(statistics.mean(e * e for e in errors)), 6),
                                    "bias_prediction_minus_actual": round(statistics.mean(errors), 6),
                                    "training_mean_baseline_mae": round(statistics.mean(
                                        abs(baseline - actual) for _predicted, actual in pairs), 6)}
        report["turbines"][turbine] = offsets
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
