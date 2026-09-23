"""Reject future calibration in every entry point; keep evaluation causal."""
from __future__ import annotations

import datetime as dt
import io
import json
import pathlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr
from unittest import mock

import power_model
import run_forecast
import validate_january
import web

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/power-curve-2026-01-31.json"


class ModelChronologyTests(unittest.TestCase):
    def future_model(self):
        model = json.loads(MODEL.read_text())
        model["turbines"]["turbine-1"]["forecast_wind_calibration"]["period"] = ["2026-02-01", "2026-02-14"]
        return model

    def test_predict_and_validation_reject_future_calibration_before_inputs(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            power_model.predict(self.future_model(), {"decision_time_utc": "2026-02-01T06:00:00Z"})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "calibration"):
                validate_january.predicted_hours(self.future_model(), pathlib.Path(temporary),
                                                "turbine-1", dt.date(2026, 2, 1), dt.date(2026, 2, 2))

    def test_cli_rejects_future_calibration_before_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            model_path = root / "model.json"
            model_path.write_text(json.dumps(self.future_model()))
            with mock.patch("sys.argv", ["run_forecast.py", "2026-02-01", "--model", str(model_path),
                                         "--output-dir", str(root / "output")]), \
                    mock.patch.object(run_forecast, "retrieve") as fetch, redirect_stderr(io.StringIO()):
                self.assertEqual(run_forecast.main(), 1)
                fetch.assert_not_called()
            self.assertFalse((root / "output/2026-02-01-forecast.json").exists())

    def test_web_rejects_future_calibration_and_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / MODEL.name).write_text(json.dumps(self.future_model()))
            previous = root / "2026-02-01-forecast.json"
            previous.write_bytes(b"previous good forecast")
            server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with mock.patch.object(web, "MODELS", root), mock.patch.object(web, "RECALCULATED", root), \
                        mock.patch.object(web, "retrieve") as fetch, \
                        mock.patch.object(web.Handler, "log_message"):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/forecast?date=2026-02-01&refresh=1")
                    self.assertEqual(caught.exception.code, 502)
                    caught.exception.close()
                    fetch.assert_not_called()
                self.assertEqual(previous.read_bytes(), b"previous good forecast")
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_holdout_uses_only_decisions_after_calibration(self):
        model = json.loads(MODEL.read_text())
        model["trained_through_inclusive"] = "2025-12-31"
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            # A preceding file exists but must never contribute to this holdout.
            (root / "2026-01-14-turbine-1.json").write_text("invalid preceding input")
            (root / "2026-01-15-turbine-1.json").write_text(json.dumps({"decision_time_utc": "2026-01-15T06:00:00Z"}))
            def predict(_model, weather):
                self.assertEqual(weather["decision_time_utc"], "2026-01-15T06:00:00Z")
                return {"decision_time_utc": weather["decision_time_utc"], "forecast": [
                    {"time_utc": "2026-01-15T06:00", "normalized_power": 0.5}]}
            with mock.patch.object(validate_january, "predict", side_effect=predict) as forecast:
                hours = validate_january.predicted_hours(model, root, "turbine-1", dt.date(2026, 1, 15), dt.date(2026, 1, 15))
                self.assertEqual(forecast.call_count, 1)
                self.assertEqual(list(hours), [dt.datetime(2026, 1, 15, 6, tzinfo=dt.timezone.utc)])


if __name__ == "__main__":
    unittest.main()
