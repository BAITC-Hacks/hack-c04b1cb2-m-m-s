"""Evaluate January archived forecasts by the lead time of each issue.

This supplementary check fits only the fixed source curve through 2025-12-31
and applies the already-selected 1.10 wind scale recorded in the project's
2026-01-31 model artifact. It never tunes parameters. January cache files are
required before retrieval so this tool cannot silently make network requests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast_weather import LOCATIONS, retrieve
from power_model import predict, train, validate_model
from validate_january import actual_hours

FIRST_ISSUE = dt.date(2026, 1, 15)
LAST_ISSUE = dt.date(2026, 1, 31)
CSV_LOCAL_MINUS_UTC_HOURS = 5
SOURCE_CUTOFF = dt.date(2025, 12, 31)
CALIBRATION_FIRST = dt.date(2026, 1, 1)
CALIBRATION_LAST = dt.date(2026, 1, 14)
CALIBRATED_ARTIFACT = ROOT / "models" / "power-curve-2026-01-31.json"


def fixed_calibrated_model(source_model: dict) -> dict:
    """Apply the existing model artifact's verified Jan 1–14 scale, without fitting."""
    saved = json.loads(CALIBRATED_ARTIFACT.read_text(encoding="utf-8"))
    calibrated = json.loads(json.dumps(source_model))
    for turbine in LOCATIONS:
        entry = saved["turbines"][turbine]
        metadata = entry.get("forecast_wind_calibration")
        if (entry.get("forecast_wind_scale") != 1.10 or not isinstance(metadata, dict)
                or metadata.get("period") != [str(CALIBRATION_FIRST), str(CALIBRATION_LAST)]
                or metadata.get("csv_local_minus_utc_hours_assumed") != CSV_LOCAL_MINUS_UTC_HOURS
                or metadata.get("calibration_source_model_cutoff") != str(SOURCE_CUTOFF)):
            raise ValueError(f"No expected fixed calibration provenance for {turbine}")
        calibrated["turbines"][turbine]["forecast_wind_scale"] = 1.10
        calibrated["turbines"][turbine]["forecast_wind_calibration"] = metadata
    return calibrated


def metrics(rows: list[tuple[float, float]]) -> dict:
    """Summarize (prediction, observed) pairs; never serialize row-level data."""
    if not rows:
        raise ValueError("no observed issue/target pairs in a lead-time bucket")
    errors = [predicted - observed for predicted, observed in rows]
    return {
        "mae_normalized_power": round(statistics.mean(abs(error) for error in errors), 6),
        "rmse_normalized_power": round(math.sqrt(statistics.mean(error * error for error in errors)), 6),
        "bias_prediction_minus_actual": round(statistics.mean(errors), 6),
    }


def evaluate(input_dir: pathlib.Path, cache_dir: pathlib.Path) -> dict:
    required = [cache_dir / f"{day}-{turbine}.json"
                for day in (FIRST_ISSUE + dt.timedelta(days=n)
                            for n in range((LAST_ISSUE - FIRST_ISSUE).days + 1))
                for turbine in LOCATIONS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required archived weather cache files; no network fetch attempted.")

    source_model = train(input_dir, SOURCE_CUTOFF)
    validate_model(source_model, SOURCE_CUTOFF)
    calibrated_model = fixed_calibrated_model(source_model)
    actual = {turbine: actual_hours(input_dir / f"{turbine}.csv", FIRST_ISSUE, LAST_ISSUE)
              for turbine in LOCATIONS}
    buckets: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {
        turbine: {"lead_0_23": {"baseline": [], "calibrated": []},
                 "lead_24_47": {"baseline": [], "calibrated": []}}
        for turbine in LOCATIONS
    }
    issue_counts: dict[str, dict[str, set[dt.datetime]]] = {
        turbine: {"lead_0_23": set(), "lead_24_47": set()} for turbine in LOCATIONS
    }

    day = FIRST_ISSUE
    while day <= LAST_ISSUE:
        for turbine in LOCATIONS:
            # All files were checked above. This validates provenance and reads
            # from cache only; a missing file cannot trigger a network request.
            weather = retrieve(day, turbine, cache_dir)
            baseline_rows = predict(source_model, weather)["forecast"]
            calibrated_rows = predict(calibrated_model, weather)["forecast"]
            decision = dt.datetime.fromisoformat(weather["decision_time_utc"].replace("Z", "+00:00"))
            if decision.tzinfo is None or decision.utcoffset() != dt.timedelta(0):
                raise ValueError(f"weather decision is not UTC: {day} {turbine}")
            if len(baseline_rows) != len(calibrated_rows) or len(baseline_rows) != 48:
                raise ValueError(f"expected 48 paired forecast hours: {day} {turbine}")
            for baseline, calibrated in zip(baseline_rows, calibrated_rows):
                if baseline["time_utc"] != calibrated["time_utc"]:
                    raise ValueError("baseline and calibrated target timestamps differ")
                target = dt.datetime.fromisoformat(baseline["time_utc"].replace("Z", "+00:00"))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=dt.timezone.utc)
                lead = (target - decision).total_seconds() / 3600
                if lead < 0 or lead >= 48 or lead % 1:
                    raise ValueError(f"invalid integer lead time: {day} {turbine}")
                observed_local = (target.replace(tzinfo=None)
                                  + dt.timedelta(hours=CSV_LOCAL_MINUS_UTC_HOURS))
                observed = actual[turbine].get(observed_local)
                if observed is None:
                    continue
                bucket = "lead_0_23" if lead < 24 else "lead_24_47"
                buckets[turbine][bucket]["baseline"].append(
                    (baseline["normalized_power"], observed))
                buckets[turbine][bucket]["calibrated"].append(
                    (calibrated["normalized_power"], observed))
                issue_counts[turbine][bucket].add(decision)
        day += dt.timedelta(days=1)

    result = {
        "validation_issues": [str(FIRST_ISSUE), str(LAST_ISSUE)],
        "source_model_cutoff_inclusive": str(SOURCE_CUTOFF),
        "calibration_period": [str(CALIBRATION_FIRST), str(CALIBRATION_LAST)],
        "calibration_scale_by_turbine": {name: 1.10 for name in LOCATIONS},
        "csv_local_minus_utc_assumption_hours": CSV_LOCAL_MINUS_UTC_HOURS,
        "weather_source": "cached Open-Meteo archived ECMWF IFS runs; default competition station profile",
        "sampling": "Each issue is scored at its own lead time; later issues do not replace earlier issue hours.",
        "turbines": {},
    }
    for turbine in LOCATIONS:
        result["turbines"][turbine] = {}
        for bucket in ("lead_0_23", "lead_24_47"):
            baseline = buckets[turbine][bucket]["baseline"]
            calibrated = buckets[turbine][bucket]["calibrated"]
            if len(baseline) != len(calibrated):
                raise AssertionError("paired variants have different sample counts")
            result["turbines"][turbine][bucket] = {
                "issue_count": len(issue_counts[turbine][bucket]),
                "paired_issue_target_hours": len(baseline),
                "baseline": metrics(baseline),
                "calibrated": metrics(calibrated),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True,
                        help="directory containing turbine-1.csv and turbine-2.csv")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"),
                        help="directory with all 2026-01-15 through 2026-01-31 weather cache files")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("predictions/lead-time-validation.json"),
                        help="output path for aggregate metrics only")
    args = parser.parse_args()
    try:
        report = evaluate(args.input_dir, args.cache_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Lead-time validation failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved aggregate metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
