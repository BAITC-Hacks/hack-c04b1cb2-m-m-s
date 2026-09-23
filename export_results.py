"""Create one February CSV from overlapping daily 48-hour forecasts.

For every target hour, choose the newest forecast whose decision time has
already passed. The daily JSON files remain the full audit trail.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib


def export(input_dir: pathlib.Path, output: pathlib.Path) -> int:
    latest: dict[dt.datetime, tuple[dt.datetime, dict]] = {}
    for index in range(29):
        day = dt.date(2026, 1, 31) + dt.timedelta(days=index)
        payload = json.loads((input_dir / f"{day}-forecast.json").read_text())
        if payload["forecast_date"] != day.isoformat():
            raise ValueError(f"forecast date mismatch: {day}")
        decision = dt.datetime.fromisoformat(payload["decision_time_utc"].replace("Z", "+00:00"))
        run_times = payload["weather_run_utc"]
        if dt.date.fromisoformat(payload["training_cutoff_inclusive"]) >= day:
            raise ValueError(f"future training data: {day}")
        if any(dt.datetime.fromisoformat(run.replace("Z", "+00:00")) >= decision
               for run in run_times.values()):
            raise ValueError(f"weather run not earlier than decision: {day}")
        rows = payload["forecast"]
        if len(rows) != 48:
            raise ValueError(f"not 48 hours: {day}")
        for row in rows:
            moment = dt.datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=dt.timezone.utc)
            if moment < decision:
                raise ValueError(f"forecast precedes decision: {day}")
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
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(latest[expected[0]][1]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(latest[moment][1] for moment in expected)
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
