"""Fit a small forecast-to-sensor wind adjustment using past forecast runs.

The power curve is fitted on turbine sensor wind, while deployment receives
100 m forecast wind. Select one multiplicative adjustment per turbine on a
separate calibration period; retain the period and CSV time-zone assumption in
the model artifact. Keep a later period untouched for validation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics

from power_model import curve_value
from validate_january import actual_hours

SCALES = tuple(round(0.8 + 0.05 * i, 2) for i in range(11))


def weather_winds(directory: pathlib.Path, turbine: str, first: dt.date,
                  last: dt.date) -> dict[dt.datetime, float]:
    """For each target hour, retain the latest already-issued daily forecast."""
    latest: dict[dt.datetime, tuple[dt.datetime, float]] = {}
    day = first
    while day <= last:
        payload = json.loads((directory / f"{day}-{turbine}.json").read_text())
        decision = dt.datetime.fromisoformat(payload["decision_time_utc"].replace("Z", "+00:00"))
        for row in payload["forecast"]:
            moment = dt.datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=dt.timezone.utc)
            if moment < decision:
                raise ValueError("weather record precedes its decision time")
            wind = float(row["wind_speed_100m"])
            if moment not in latest or decision > latest[moment][0]:
                latest[moment] = decision, wind
        day += dt.timedelta(days=1)
    return {moment: wind for moment, (_decision, wind) in latest.items()}


def calibrate(model: dict, input_dir: pathlib.Path, weather_dir: pathlib.Path,
              first: dt.date, last: dt.date, offset_hours: int) -> dict:
    if first > last:
        raise ValueError("calibration start must precede its end")
    if dt.date.fromisoformat(model["trained_through_inclusive"]) >= first:
        raise ValueError("power model must be trained before calibration period")
    result = json.loads(json.dumps(model))
    for turbine in ("turbine-1", "turbine-2"):
        actual = actual_hours(input_dir / f"{turbine}.csv", first, last)
        winds = weather_winds(weather_dir, turbine, first, last)
        pairs = [(winds[utc], power) for local, power in actual.items()
                 if (utc := (local - dt.timedelta(hours=offset_hours)).replace(
                     tzinfo=dt.timezone.utc)) in winds]
        if len(pairs) < 24:
            raise ValueError(f"too few comparable calibration hours for {turbine}")
        curve = result["turbines"][turbine]["curve"]
        scores = {scale: statistics.mean(abs(curve_value(curve, wind * scale) - power)
                                         for wind, power in pairs) for scale in SCALES}
        best = min(SCALES, key=lambda scale: (scores[scale], abs(scale - 1)))
        result["turbines"][turbine]["forecast_wind_scale"] = best
        result["turbines"][turbine]["forecast_wind_calibration"] = {
            "method": "grid search on hourly MAE; multiply forecast 100 m wind before power curve",
            "candidate_scales": list(SCALES),
            "period": [first.isoformat(), last.isoformat()],
            "csv_local_minus_utc_hours_assumed": offset_hours,
            "matched_hours": len(pairs),
            "calibration_mae_at_scale_1": round(scores[1.0], 6),
            "calibration_mae_selected": round(scores[best], 6),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--weather-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"))
    parser.add_argument("--first", type=dt.date.fromisoformat, default=dt.date(2026, 1, 1))
    parser.add_argument("--last", type=dt.date.fromisoformat, default=dt.date(2026, 1, 14))
    parser.add_argument("--offset-hours", type=int, default=5)
    parser.add_argument("--target-model", type=pathlib.Path,
                        help="apply fitted adjustment to a newer power model after checking it has no future data")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    calibrated = calibrate(model, args.input_dir, args.weather_dir,
                           args.first, args.last, args.offset_hours)
    if args.target_model:
        target = json.loads(args.target_model.read_text())
        if dt.date.fromisoformat(target["trained_through_inclusive"]) < args.last:
            raise ValueError("target power model predates the calibration period")
        for turbine in ("turbine-1", "turbine-2"):
            source = calibrated["turbines"][turbine]
            target["turbines"][turbine]["forecast_wind_scale"] = source["forecast_wind_scale"]
            target["turbines"][turbine]["forecast_wind_calibration"] = {
                **source["forecast_wind_calibration"],
                "calibration_source_model_cutoff": model["trained_through_inclusive"],
            }
        calibrated = target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibrated, ensure_ascii=False, indent=2) + "\n")
    for turbine, payload in calibrated["turbines"].items():
        print(f"{turbine}: scale={payload['forecast_wind_scale']}, "
              f"hours={payload['forecast_wind_calibration']['matched_hours']}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
