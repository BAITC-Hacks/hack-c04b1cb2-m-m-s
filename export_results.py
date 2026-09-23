"""Create one February CSV from overlapping daily 48-hour forecasts.

For every target hour, choose the newest forecast whose decision time has
already passed. The daily JSON files remain the full audit trail.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import io
import pathlib

from storage import atomic_write


def export(input_dir: pathlib.Path, output: pathlib.Path) -> int:
    latest: dict[dt.datetime, tuple[dt.datetime, dict]] = {}
    for index in range(29):
        day = dt.date(2026, 1, 31) + dt.timedelta(days=index)
        payload = json.loads((input_dir / f"{day}-forecast.json").read_text())
        if payload["forecast_date"] != day.isoformat():
            raise ValueError(f"forecast date mismatch: {day}")
        decision = dt.datetime.fromisoformat(payload["decision_time_utc"].replace("Z", "+00:00"))
        if decision != dt.datetime.combine(day, dt.time(6), tzinfo=dt.timezone.utc):
            raise ValueError(f"unexpected decision time: {day}")
        run_times = payload["weather_run_utc"]
        if dt.date.fromisoformat(payload["training_cutoff_inclusive"]) >= day:
            raise ValueError(f"future training data: {day}")
        if any(decision - dt.datetime.fromisoformat(run.replace("Z", "+00:00"))
               < dt.timedelta(hours=6)
               for run in run_times.values()):
            raise ValueError(f"weather run too close to decision: {day}")
        rows = payload["forecast"]
        if len(rows) != 48:
            raise ValueError(f"not 48 hours: {day}")
        for hour, row in enumerate(rows):
            moment = dt.datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=dt.timezone.utc)
            if moment != decision + dt.timedelta(hours=hour):
                raise ValueError(f"forecast hour gap or overlap: {day}, hour {hour}")
            a = row["turbine_1_normalized_power"]
            b = row["turbine_2_normalized_power"]
            farm = row["farm_equal_capacity_mean_normalized_power"]
            if not all(isinstance(value, (int, float)) and math.isfinite(value)
                       and 0 <= value <= 1 for value in (a, b, farm)):
                raise ValueError(f"invalid normalized output: {day}, hour {hour}")
            if abs(farm - (a + b) / 2) > 0.000001:
                raise ValueError(f"farm output does not match turbine mean: {day}, hour {hour}")
            if moment.month != 2:
                continue
            entry = {
                "time_utc": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_forecast_date": day.isoformat(),
                "decision_time_utc": payload["decision_time_utc"],
                "weather_run_turbine_1_utc": run_times["turbine-1"],
                "weather_run_turbine_2_utc": run_times["turbine-2"],
                "training_cutoff_inclusive": payload["training_cutoff_inclusive"],
                "turbine_1_normalized_power": row["turbine_1_normalized_power"],
                "turbine_2_normalized_power": row["turbine_2_normalized_power"],
                "farm_equal_capacity_mean_normalized_power":
                    row["farm_equal_capacity_mean_normalized_power"],
            }
            if moment not in latest or decision > latest[moment][0]:
                latest[moment] = decision, entry
    start = dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)
    expected = [start + dt.timedelta(hours=hour) for hour in range(28 * 24)]
    if set(latest) != set(expected):
        missing = sorted(set(expected) - set(latest))
        raise ValueError(f"February coverage incomplete; missing {len(missing)} hours")
    output.parent.mkdir(parents=True, exist_ok=True)
    with io.StringIO(newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(latest[expected[0]][1]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(latest[moment][1] for moment in expected)
        atomic_write(output, stream.getvalue())
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("results/february-hourly.csv"))
    args = parser.parse_args()
    count = export(args.input_dir, args.output)
    print(f"saved {count} February hours to {args.output}")


if __name__ == "__main__":
    main()
