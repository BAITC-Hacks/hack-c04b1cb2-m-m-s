"""Offline HTTP checks for the server-configured autonomous campaign."""
from __future__ import annotations

import collections
import http.client
import json
import os
import pathlib
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

import web
import agent_settings
from station_profile import DEFAULT_STATION

REPO = pathlib.Path(__file__).resolve().parents[1]


class CampaignWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        (self.root / "stations").mkdir()
        (self.root / "stations/example.json").write_text(json.dumps(DEFAULT_STATION))
        (self.root / "models").mkdir()
        for path in (REPO / "models").glob("power-curve-*.json"):
            (self.root / "models" / path.name).write_bytes(path.read_bytes())
        self.csv_dir = self.root / "measurements"
        self.csv_dir.mkdir()
        for name in ("turbine-1", "turbine-2"):
            (self.csv_dir / f"{name}.csv").write_text("test fixture\n")
        self.forecast = json.loads((REPO / "results/2026-02-01-forecast.json").read_text())
        patches = [
            mock.patch.object(web, "ROOT", self.root),
            mock.patch.object(web, "AGENT_JOBS", {}),
            mock.patch.object(web, "AGENT_STARTS", collections.deque()),
            mock.patch.object(web, "load_agent_environment", return_value=None),
            mock.patch.object(web, "agent_available", return_value=True),
            mock.patch.object(web.Handler, "log_message", return_value=None),
            mock.patch.dict(os.environ, {"WIND_TRAINING_DIR": str(self.csv_dir),
                                      "WIND_STATION_FILE": ""}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            if body is not None and not isinstance(body, (str, bytes)):
                body = json.dumps(body)
            connection.request(method, path, body=body,
                               headers={"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def wait_status(self, job_id, status, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code, state = self.request("GET", f"/api/agent/status?id={job_id}")
            self.assertEqual(code, 200)
            if state["status"] == status:
                return state
            time.sleep(0.01)
        self.fail(f"job did not reach {status}")

    def report(self, **kwargs):
        return {"status": "completed", "summary": "29 дней обработаны", "events": [],
                "station": DEFAULT_STATION, "completed_days": 29, "total_days": 29,
                "forecast": self.forecast}

    def test_config_missing_key_csv_and_bad_profile_disables_campaign(self):
        with mock.patch.object(web, "agent_available", return_value=False):
            code, config = self.request("GET", "/api/agent/config")
            self.assertEqual(code, 200)
            self.assertFalse(config["campaign_available"])
            self.assertTrue(config["campaign_reason"])
            self.assertEqual(self.request("POST", "/api/campaign", {})[0], 400)
        with mock.patch.dict(os.environ, {"WIND_TRAINING_DIR": "missing-csv-directory"}):
            self.assertFalse(self.request("GET", "/api/agent/config")[1]["campaign_available"])
        (self.root / "stations/example.json").write_text('{"id":"bad"}')
        self.assertFalse(self.request("GET", "/api/agent/config")[1]["campaign_available"])
        self.assertEqual(web.AGENT_JOBS, {})
        self.assertEqual(len(web.AGENT_STARTS), 0)

    def test_campaign_starts_on_prepared_models_without_training_directory(self):
        for configured in (None, "", "  "):
            with self.subTest(configured=configured), mock.patch.dict(os.environ):
                if configured is None:
                    os.environ.pop("WIND_TRAINING_DIR", None)
                else:
                    os.environ["WIND_TRAINING_DIR"] = configured
                code, config = self.request("GET", "/api/agent/config")
                self.assertEqual(code, 200)
                self.assertTrue(config["campaign_available"], config["campaign_reason"])
                self.assertEqual(config["campaign_reason"], "")
                self.assertEqual(config["campaign_mode"], "prepared_models")
                with mock.patch.object(web, "run_campaign_job", side_effect=self.report) as run:
                    code, started = self.request("POST", "/api/campaign", {})
                    self.assertEqual(code, 202)
                    self.wait_status(started["job_id"], "completed")
                self.assertIsNone(run.call_args.kwargs["input_dir"])
                self.assertEqual(run.call_args.kwargs["station"], DEFAULT_STATION)

    def test_prepared_mode_rejects_missing_or_corrupt_models_and_other_station(self):
        path = self.root / "models/power-curve-2026-01-31.json"
        original = path.read_bytes()
        with mock.patch.dict(os.environ, {"WIND_TRAINING_DIR": ""}):
            for contents in (None, b"{bad json}"):
                if contents is None:
                    path.unlink()
                else:
                    path.write_bytes(contents)
                config = self.request("GET", "/api/agent/config")[1]
                self.assertFalse(config["campaign_available"])
                self.assertIsNone(config["campaign_mode"])
                self.assertEqual(self.request("POST", "/api/campaign", {})[0], 400)
                self.assertNotIn(str(self.root), config["campaign_reason"])
            path.write_bytes(original)
            station = json.loads(json.dumps(DEFAULT_STATION))
            station["locations"]["turbine-1"][0] = 44.125
            (self.root / "stations/example.json").write_text(json.dumps(station))
            self.assertFalse(self.request("GET", "/api/agent/config")[1]["campaign_available"])
            self.assertEqual(self.request("POST", "/api/campaign", {})[0], 400)
        self.assertEqual(web.AGENT_JOBS, {})
        self.assertEqual(len(web.AGENT_STARTS), 0)

    def test_start_poll_progress_and_finished_forecast(self):
        self.assertEqual(self.request("GET", "/api/agent/config")[1]["campaign_mode"], "training")
        started = threading.Event()
        release = threading.Event()

        def campaign(**kwargs):
            self.assertEqual(kwargs["root"], self.root)
            self.assertEqual(kwargs["input_dir"], self.csv_dir)
            self.assertEqual(kwargs["station"], DEFAULT_STATION)
            kwargs["on_event"]({"step": 1, "tool": "campaign_day", "status": "ok",
                                "detail": "День завершён", "forecast_date": "2026-01-31"})
            kwargs["on_event"]({"step": 2, "tool": "campaign_retry", "status": "start",
                                "detail": "День 1/29, попытка 2/2.", "forecast_date": "2026-01-31"})
            kwargs["on_event"]({"step": 3, "tool": "inspect_station", "status": "ok",
                                "detail": "Инструмент выполнен.", "forecast_date": "2026-01-31",
                                "result": {"station": DEFAULT_STATION, "training_cutoff": "2026-01-30",
                                           "model_available": False, "training_configured": True,
                                           "private_path": "/private/measurements"}})
            kwargs["on_event"]({"step": 4, "tool": "get_weather", "status": "ok",
                                "detail": "Инструмент выполнен.", "forecast_date": "2026-01-31",
                                "arguments": {"turbine": "turbine-1", "refresh": False,
                                              "path": "/private/cache"},
                                "result": {"turbine": "turbine-1", "hours": 48,
                                           "model_run_utc": "2026-01-30T18:00:00Z",
                                           "refresh": False, "path": "/private/cache"}})
            started.set()
            self.assertTrue(release.wait(3))
            return self.report()

        with mock.patch.object(web, "run_campaign_job", side_effect=campaign) as run:
            code, response = self.request("POST", "/api/campaign", {})
            self.assertEqual(code, 202)
            self.assertTrue(started.wait(2))
            code, state = self.request("GET", f"/api/agent/status?id={response['job_id']}")
            self.assertEqual(code, 200)
            self.assertEqual(state["kind"], "campaign")
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["station"]["name"], DEFAULT_STATION["name"])
            self.assertEqual(state["completed_days"], 1)
            self.assertEqual(state["events"][0]["forecast_date"], "2026-01-31")
            self.assertEqual(state["events"][1]["detail"], "День 1/29, попытка 2/2.")
            self.assertEqual(state["events"][2]["result"]["station_name"], DEFAULT_STATION["name"])
            self.assertEqual(state["events"][3]["result"]["hours"], 48)
            self.assertEqual(state["events"][3]["arguments"], {"turbine": "turbine-1", "refresh": False})
            self.assertNotIn("/private", json.dumps(state))
            release.set()
            done = self.wait_status(response["job_id"], "completed")
            self.assertEqual(done["completed_days"], 29)
            self.assertEqual(done["forecast"], self.forecast)
            self.assertEqual(done["summary"], "29 дней обработаны")
            run.assert_called_once()

    def test_failed_report_does_not_expose_partial_forecast_as_complete(self):
        partial = {"status": "failed", "summary": "private/path/detail", "events": [],
                   "station": DEFAULT_STATION, "completed_days": 7, "total_days": 29,
                   "forecast": self.forecast}
        with mock.patch.object(web, "run_campaign_job", return_value=partial):
            code, started = self.request("POST", "/api/campaign", {})
            self.assertEqual(code, 202)
            state = self.wait_status(started["job_id"], "failed")
        self.assertEqual(state["completed_days"], 7)
        self.assertIsNone(state["forecast"])
        self.assertNotIn("private/path", state["summary"])

    def test_shared_busy_and_hourly_limit_with_daily_agent(self):
        entered = threading.Event()
        release = threading.Event()

        def campaign(**kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return self.report()

        daily = {"status": "completed", "summary": "День готов", "events": [],
                 "forecast": self.forecast}
        with mock.patch.object(web, "run_campaign_job", side_effect=campaign), \
                mock.patch.object(web, "run_agent", return_value=daily) as run_daily:
            code, first = self.request("POST", "/api/campaign", {})
            self.assertEqual(code, 202)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.request("POST", "/api/agent", {"date": "2026-02-01"})[0], 429)
            run_daily.assert_not_called()
            release.set()
            self.wait_status(first["job_id"], "completed")
            # The first campaign consumes one of six starts, regardless of its 29 days.
            for _ in range(5):
                code, response = self.request("POST", "/api/agent", {"date": "2026-02-01"})
                self.assertEqual(code, 202)
                self.wait_status(response["job_id"], "completed")
            self.assertEqual(self.request("POST", "/api/campaign", {})[0], 429)
            self.assertEqual(run_daily.call_count, 5)
            self.assertEqual(len(web.AGENT_STARTS), 6)

    def test_campaign_rejects_client_paths_and_existing_daily_flow_survives(self):
        with mock.patch.object(web, "run_campaign_job") as run_campaign, \
                mock.patch.object(web, "run_agent", return_value={
                    "status": "completed", "summary": "День готов", "events": [],
                    "forecast": self.forecast}) as run_daily:
            for body in ({"date": "2026-02-01"}, {"station": "elsewhere"},
                         {"input_dir": "/tmp"}, []):
                self.assertEqual(self.request("POST", "/api/campaign", body)[0], 400)
            self.assertEqual(self.request("POST", "/api/campaign", {},
                                          headers={"Origin": "http://evil.invalid"})[0], 403)
            run_campaign.assert_not_called()
            code, response = self.request("POST", "/api/agent", {"date": "2026-02-01"})
            self.assertEqual(code, 202)
            self.wait_status(response["job_id"], "completed")
            run_daily.assert_called_once()

    def test_settings_load_only_allowlisted_campaign_paths(self):
        (self.root / ".env").write_text(
            'WIND_TRAINING_DIR="measurements"\n'
            'WIND_STATION_FILE=stations/example.json\n'
            'UNRELATED_SETTING=unexpected\n')
        with mock.patch.dict(os.environ, {"WIND_TRAINING_DIR": "", "WIND_STATION_FILE": ""}):
            os.environ.pop("UNRELATED_SETTING", None)
            agent_settings.load_agent_environment(self.root)
            self.assertEqual(os.environ["WIND_TRAINING_DIR"], "measurements")
            self.assertEqual(os.environ["WIND_STATION_FILE"], "stations/example.json")
            self.assertNotIn("UNRELATED_SETTING", os.environ)


if __name__ == "__main__":
    unittest.main()
