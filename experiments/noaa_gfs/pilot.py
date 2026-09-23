#!/usr/bin/env python3
"""Fetch just three archived GFS 100 m wind forecasts and compare with ECMWF."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
import time
import statistics
import urllib.request
import sys

from eccodes import codes_grib_find_nearest, codes_new_from_message, codes_get, codes_release

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from power_model import curve_value, train
from validate_january import actual_hours

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
LOC = {"turbine-1": (43.645150, 78.535604), "turbine-2": (43.643198, 78.538828)}
DAYS = [dt.date(2026, 1, d) for d in (15, 16, 17)]
UTC = dt.timezone.utc


def fetch(url: str, headers: dict[str, str] | None = None,
          expected_range: tuple[int, int] | None = None) -> tuple[bytes, dict]:
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=70) as r:
                status = r.status
                content_range = r.headers.get("Content-Range")
                content_length = r.headers.get("Content-Length")
                if expected_range is not None:
                    start, end = expected_range
                    if status != 206 or not (content_range or "").startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"server ignored requested HTTP range: status={status}, range={content_range}")
                    body = r.read(end - start + 2)
                    if len(body) != end - start + 1:
                        raise RuntimeError("HTTP range returned an unexpected payload length")
                else:
                    body = r.read()
                return body, {"status": status, "content_range": content_range,
                              "content_length": content_length}
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(1 + attempt)
    raise last


def field_wind(cache: pathlib.Path, day: dt.date, lead: int, lat: float, lon: float,
               stats: dict) -> tuple[float, dict]:
    cycle_date = day - dt.timedelta(days=1)
    tag = cycle_date.strftime("%Y%m%d")
    root = f"{BASE}/gfs.{tag}/18/atmos/gfs.t18z.pgrb2.0p25.f{lead:03d}"
    idx_path = cache / f"{tag}-f{lead:03}.idx"
    if idx_path.exists():
        idx = idx_path.read_bytes()
    else:
        idx, meta = fetch(root + ".idx")
        stats["downloaded_bytes"] += len(idx)
        idx_path.write_bytes(idx)
    lines = idx.decode("ascii").splitlines()
    fields = {}
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) >= 6 and parts[3] in ("UGRD", "VGRD") and parts[4] == "100 m above ground":
            fields[parts[3]] = (int(parts[1]), int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None,
                                line)
    if set(fields) != {"UGRD", "VGRD"}:
        raise RuntimeError(f"100 m U/V fields missing for {tag} lead {lead}")
    uv = []
    selected = {}
    for name in ("UGRD", "VGRD"):
        start, end, desc = fields[name]
        # Last index field extends to the end of the GRIB object. Expected leads have successors.
        if end is None:
            raise RuntimeError("selected field has no following index offset")
        cache_file = cache / f"{tag}-f{lead:03}-{name}.grib"
        if cache_file.exists():
            raw = cache_file.read_bytes()
            if len(raw) != end - start:
                raise RuntimeError(f"cached GRIB range has wrong length: {cache_file}")
            response = {"status": 206, "content_range": f"bytes {start}-{end-1}/cached"}
        else:
            if stats["downloaded_bytes"] + end - start > 500_000_000:
                raise RuntimeError("500 MB pilot download ceiling reached")
            raw, response = fetch(root, {"Range": f"bytes={start}-{end-1}"}, (start, end-1))
        expected = f"bytes {start}-{end-1}/"
        if (response["status"] != 206 or not (response["content_range"] or "").startswith(expected)
                or len(raw) != end - start):
            raise RuntimeError(f"HTTP Range not honored for {name}: {response}")
        if not cache_file.exists():
            stats["downloaded_bytes"] += len(raw)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(raw)
        handle = codes_new_from_message(raw)
        if handle is None:
            raise RuntimeError(f"cannot decode {name} lead {lead}")
        try:
            expected_short = name.lower()[0]
            if (codes_get(handle, "shortName") != expected_short or codes_get(handle, "level") != 100
                    or codes_get(handle, "typeOfLevel") != "heightAboveGround"
                    or codes_get(handle, "dataDate") != int(tag) or codes_get(handle, "dataTime") != 1800
                    or codes_get(handle, "step") != lead or codes_get(handle, "units") not in ("m s**-1", "m/s")):
                raise RuntimeError(f"unexpected GRIB message for {name}: {desc}")
            valid = dt.datetime.combine(cycle_date, dt.time(18), tzinfo=UTC) + dt.timedelta(hours=lead)
            if (codes_get(handle, "validityDate") != int(valid.strftime("%Y%m%d"))
                    or codes_get(handle, "validityTime") != int(valid.strftime("%H%M"))):
                raise RuntimeError(f"GRIB validity time mismatch for {name}: {desc}")
            nearest = codes_grib_find_nearest(handle, lat, lon)[0]
            uv.append(float(nearest.value))
            selected[name] = {"idx_line": desc, "range_start": start, "range_end_inclusive": end-1,
                              "range_bytes": end-start, "nearest_latitude": nearest.lat,
                              "nearest_longitude": nearest.lon, "grid_value_ms": float(nearest.value),
                              "units": codes_get(handle, "units"), "dataDate": codes_get(handle, "dataDate"),
                              "dataTime": codes_get(handle, "dataTime"), "step_hours": codes_get(handle, "step"),
                              "validityDate": codes_get(handle, "validityDate"),
                              "validityTime": codes_get(handle, "validityTime")}
        finally:
            codes_release(handle)
    return math.hypot(*uv), selected


def error_metrics(errors: list[float]) -> dict:
    if not errors:
        return {"hours": 0}
    return {"hours": len(errors), "mae": round(statistics.mean(map(abs, errors)), 6),
            "rmse": round(math.sqrt(statistics.mean(e*e for e in errors)), 6),
            "bias_prediction_minus_actual": round(statistics.mean(errors), 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=pathlib.Path, required=True)
    ap.add_argument("--ecmwf-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"))
    ap.add_argument("--cache-dir", type=pathlib.Path, required=True)
    ap.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path(__file__).parent)
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = train(args.input_dir, dt.date(2025, 12, 31))
    rows, selected_fields = [], {}
    stats = {"downloaded_bytes": sum(p.stat().st_size for p in args.cache_dir.glob("*.grib"))
             + sum(p.stat().st_size for p in args.cache_dir.glob("*.idx"))}
    actual = {t: actual_hours(args.input_dir / f"{t}.csv", dt.date(2026, 1, 15), dt.date(2026, 1, 19))
              for t in LOC}
    offset = 5  # Fixed from the already documented hypothesis; not selected on these days.
    for day in DAYS:
        decision = dt.datetime.combine(day, dt.time(6), tzinfo=UTC)
        e_day = day.isoformat()
        for turbine, (lat, lon) in LOC.items():
            curve = model["turbines"][turbine]["curve"]
            gfs_winds = {}
            for lead in range(12, 36):
                wind, fieldmeta = field_wind(args.cache_dir, day, lead, lat, lon, stats)
                hour = decision + dt.timedelta(hours=lead-12)
                gfs_winds[hour] = wind
                selected_fields[f"{e_day}/{turbine}/f{lead:03}"] = fieldmeta
            ecmwf_path = args.ecmwf_dir / f"{e_day}-{turbine}.json"
            ecmwf = json.loads(ecmwf_path.read_text())
            e_by_time = {}
            for r in ecmwf["forecast"]:
                stamp = dt.datetime.fromisoformat(r["time_utc"].replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=UTC)
                e_by_time[stamp] = float(r["wind_speed_100m"])
            for timestamp, gwind in gfs_winds.items():
                local = timestamp.replace(tzinfo=None) + dt.timedelta(hours=offset)
                measured = actual[turbine].get(local)
                if measured is None:
                    continue
                ewind = e_by_time.get(timestamp)
                if ewind is None:
                    continue
                e10 = curve_value(curve, ewind * 1.10)
                rows.append({"decision_date_utc": e_day, "valid_time_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "turbine": turbine, "gfs_wind_100m_ms": round(gwind, 6),
                             "ecmwf_wind_100m_ms": round(ewind, 6),
                             "gfs_power": curve_value(curve, gwind), "ecmwf_power": curve_value(curve, ewind),
                             "ecmwf_x1_10_power": e10,
                             "_gfs_error": curve_value(curve, gwind) - measured,
                             "_ecmwf_error": curve_value(curve, ewind) - measured,
                             "_ecmwf_x1_10_error": e10 - measured})
    # The committed comparison CSV contains forecasts only. Errors and targets are
    # represented only by aggregate metrics so the actual time series cannot be recovered.
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if not k.startswith("_")], lineterminator="\n")
        w.writeheader()
        w.writerows([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    report = {"dates": [d.isoformat() for d in DAYS], "decision_time_utc": "06:00 UTC",
              "target_window_utc": ["2026-01-15T06:00:00Z", "2026-01-18T05:00:00Z"],
              "requested_target_hours": 72,
              "run_rule": "fixed previous-day GFS 18 UTC cycle; forecast leads 12-35",
              "utc_local_assumption_hours": offset, "training_cutoff": model["trained_through_inclusive"],
              "source_csv_sha256": {t: hashlib.sha256((args.input_dir / f"{t}.csv").read_bytes()).hexdigest() for t in LOC},
              "turbines": {}, "downloaded_bytes": stats["downloaded_bytes"]}
    for t in LOC:
        tr = [r for r in rows if r["turbine"] == t]
        report["turbines"][t] = {label: error_metrics([r["_" + err] for r in tr]) for label, err in
                                 (("gfs", "gfs_error"), ("ecmwf", "ecmwf_error"),
                                  ("ecmwf_x1_10", "ecmwf_x1_10_error"))}
    provenance = {**report, "source": "NOAA GFS 0p25 GRIB2; https://registry.opendata.aws/noaa-gfs-bdp-pds/",
                  "run_url_template": f"{BASE}/gfs.YYYYMMDD/18/atmos/gfs.t18z.pgrb2.0p25.fFFF",
                  "field_url_template": f"{BASE}/gfs.YYYYMMDD/18/atmos/gfs.t18z.pgrb2.0p25.fFFF.idx",
                  "field_filter": ["UGRD:100 m above ground", "VGRD:100 m above ground"],
                  "coordinates": LOC, "spatial_sampling": "nearest grid point independently for each turbine; no interpolation",
                  "wind_speed": "hypot(UGRD,VGRD)", "range_method": "GET each .idx and byte-range only each selected GRIB message; verify HTTP 206 Content-Range",
                  "selected_fields": selected_fields, "measurements": "excluded from committed outputs; comparison CSV contains forecasts only",
                  "tool_versions": {name: importlib.metadata.version(name) for name in ("numpy", "eccodes", "eccodeslib")},
                  "tool_licenses": {name: {"license": importlib.metadata.metadata(name).get("License") or importlib.metadata.metadata(name).get("License-Expression") or "unspecified in package metadata",
                                            "license_classifiers": [v for v in importlib.metadata.metadata(name).get_all("Classifier", []) if "License" in v],
                                            "license_files": importlib.metadata.metadata(name).get_all("License-File", [])}
                                    for name in ("numpy", "eccodes", "eccodeslib")},
                  "platform": platform.platform(),
                  "python_version": sys.version,
                  "limitations": ["CSV timezone is unspecified; local-UTC +5 is a hypothesis", "72 h pilot only; not full January validation", "no coefficients fit on pilot", "run cycle time is not proof of historical public availability"]}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
