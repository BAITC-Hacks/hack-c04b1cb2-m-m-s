"""Offline checks of repeated archived-input checks and recovery."""
from __future__ import annotations

import datetime as dt
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import watch_forecast
import forecast_weather
from forecast_weather import LOCATIONS, run_for
from run_forecast import combine


DAY = dt.date(2026, 2, 1)
MODEL = pathlib.Path(__file__).resolve().parents[1] / "models/power-curve-2026-01-31.json"


def weather(turbine: str, wind: float = 8.0) -> dict:
    decision, run = run_for(DAY)
    return {
        "source": "offline test archive",
        "request_url": f"https://example.invalid/archive/{turbine}",
        "model_run_utc": run.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision_time_utc": decision.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "turbine": turbine,
        "coordinates": dict(zip(("latitude", "longitude"), LOCATIONS[turbine])),
        "units": {"wind_speed_10m": "m/s", "wind_speed_100m": "m/s",
                  "temperature_2m": "°C"},
        "forecast": [{"time_utc": (decision + dt.timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M"),
                      "wind_speed_10m": wind * 0.8, "wind_speed_100m": wind,
                      "temperature_2m": 10.0} for hour in range(48)],
    }


class WatchForecastTests(unittest.TestCase):
    def test_model_update_and_invalid_model_keep_last_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model_path = root / "model.json"
            model = json.loads(MODEL.read_text())
            model_path.write_text(json.dumps(model))
            output = root / "output"
            path = output / "2026-02-01-forecast.json"
            with mock.patch.object(watch_forecast, "retrieve", side_effect=lambda day, turbine, cache,
                                                                            refresh: weather(turbine)):
                self.assertTrue(watch_forecast.run_cycle(DAY, model_path, root / "cache", output))
                before = path.read_bytes()
                for turbine in model["turbines"].values():
                    turbine["forecast_wind_scale"] = 1.2
                model_path.write_text(json.dumps(model))
                self.assertTrue(watch_forecast.run_cycle(DAY, model_path, root / "cache", output))
                good = path.read_bytes()
                self.assertNotEqual(before, good)
                for invalid in ([], [-0.1, 0.5], [float("nan")]):
                    model["turbines"]["turbine-1"]["curve"]["values"] = invalid
                    model_path.write_text(json.dumps(model))
                    self.assertEqual(watch_forecast.watch(DAY, model_path, root / "cache", output, 1, 1), 1)
                    self.assertEqual(good, path.read_bytes())
                model["trained_through_inclusive"] = "2026-02-01"
                model_path.write_text(json.dumps(model))
                self.assertEqual(watch_forecast.watch(DAY, model_path, root / "cache", output, 1, 1), 1)
                self.assertEqual(good, path.read_bytes())

    def test_refresh_keeps_identical_weather_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            decision, _ = run_for(DAY)
            times = [(decision - dt.timedelta(hours=6) + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
                     for i in range(60)]
            body = {
                "hourly_units": {"wind_speed_10m": "m/s", "wind_speed_100m": "m/s",
                                 "temperature_2m": "°C"},
                "hourly": {"time": times, "wind_speed_10m": [6.0] * 60,
                           "wind_speed_100m": [8.0] * 60, "temperature_2m": [10.0] * 60},
            }

            def response(_url, timeout):
                self.assertEqual(timeout, 30)
                return io.BytesIO(json.dumps(body).encode())

            path = cache / "2026-02-01-turbine-1.json"
            with mock.patch.object(forecast_weather.urllib.request, "urlopen", side_effect=response):
                forecast_weather.retrieve(DAY, "turbine-1", cache, refresh=True)
                initial = (path.read_bytes(), path.stat().st_mtime_ns)
                forecast_weather.retrieve(DAY, "turbine-1", cache, refresh=True)
                self.assertEqual(initial, (path.read_bytes(), path.stat().st_mtime_ns))
                body["hourly"]["wind_speed_100m"][6] = 9.0
                forecast_weather.retrieve(DAY, "turbine-1", cache, refresh=True)
                self.assertNotEqual(initial[0], path.read_bytes())

    def test_change_failure_recovery_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model_path = root / "model.json"
            model_path.write_bytes(MODEL.read_bytes())
            output_dir = root / "output"
            cache_dir = root / "cache"
            output_path = output_dir / "2026-02-01-forecast.json"
            state_path = output_dir / "2026-02-01-watch-state.json"
            cycle = 0
            snapshots = []

            def source(day, turbine, cache, refresh=False):
                nonlocal cycle
                self.assertEqual(day, DAY)
                self.assertEqual(cache, cache_dir)
                self.assertTrue(refresh)
                if turbine == "turbine-1":
                    cycle += 1
                if cycle == 4 and turbine == "turbine-2":
                    raise OSError("archive unavailable")
                result = weather(turbine, 12.0 if cycle >= 3 else 8.0)
                if cycle == 5 and turbine == "turbine-2":
                    result["forecast"][3]["time_utc"] = result["forecast"][2]["time_utc"]
                return result

            def observe(_interval):
                snapshots.append((output_path.read_bytes(), output_path.stat().st_mtime_ns,
                                  state_path.read_bytes(), state_path.stat().st_mtime_ns))
                if cycle == 1:
                    # Formatting alone is not a model update.
                    model_path.write_text(json.dumps(json.loads(model_path.read_text())))

            with mock.patch.object(watch_forecast, "retrieve", side_effect=source), \
                    mock.patch.object(watch_forecast.time, "sleep", side_effect=observe):
                self.assertEqual(watch_forecast.watch(DAY, model_path, cache_dir,
                                                       output_dir, 0.01, 6), 0)
            snapshots.append((output_path.read_bytes(), output_path.stat().st_mtime_ns,
                              state_path.read_bytes(), state_path.stat().st_mtime_ns))

            self.assertEqual(len(snapshots), 6)
            self.assertEqual(snapshots[0], snapshots[1])  # unchanged
            self.assertNotEqual(snapshots[1][0], snapshots[2][0])  # updated inputs
            self.assertEqual(snapshots[2], snapshots[3])  # source failure
            self.assertEqual(snapshots[3], snapshots[4])  # invalid chronology
            self.assertEqual(snapshots[4], snapshots[5])  # recovered, still unchanged
            output = json.loads(output_path.read_text())
            self.assertEqual(output["forecast_date"], DAY.isoformat())
            self.assertEqual(len(output["forecast"]), 48)
            expected = combine(DAY, json.loads(model_path.read_text()),
                               {name: weather(name, 12.0) for name in LOCATIONS})
            self.assertEqual(output_path.read_text(), json.dumps(expected, ensure_ascii=False, indent=2) + "\n")

            with mock.patch.object(watch_forecast, "retrieve", side_effect=lambda day, turbine, cache,
                                                                            refresh: weather(turbine, 12.0)):
                self.assertEqual(watch_forecast.watch(DAY, model_path, cache_dir,
                                                       output_dir, 0.01, 1), 0)
            self.assertEqual(snapshots[-1], (output_path.read_bytes(), output_path.stat().st_mtime_ns,
                                             state_path.read_bytes(), state_path.stat().st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
