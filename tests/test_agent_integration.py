"""Offline integration checks for the watcher and local agent HTTP API."""
from __future__ import annotations

import collections
import datetime as dt
import http.client
import io
import json
import pathlib
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import ThreadingHTTPServer
from unittest import mock

import forecast_agent
import watch_forecast
import web
from forecast_weather import LOCATIONS
from run_forecast import combine
from tests.test_watch_forecast import weather

DAY = dt.date(2026, 2, 1)
MODEL = pathlib.Path(__file__).resolve().parents[1] / "models/power-curve-2026-01-31.json"


class WatchAgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.model_path = self.root / "model.json"
        self.model_path.write_bytes(MODEL.read_bytes())
        self.model = json.loads(self.model_path.read_text())
        self.cache = self.root / "cache"
        self.output = self.root / "output"
        self.forecast_path = self.output / "2026-02-01-forecast.json"
        self.state_path = self.output / "2026-02-01-watch-state.json"
        self.wind = 8.0
        self.calls = []

        def retrieve(day, turbine, cache, refresh=False):
            self.assertEqual(day, DAY)
            self.assertEqual(cache, self.cache)
            self.assertTrue(refresh)
            return weather(turbine, self.wind)

        def agent_run(day, *, root, model_path, cache_dir, output_dir):
            self.assertEqual(day, DAY)
            self.assertEqual(model_path, self.model_path)
            self.assertEqual(cache_dir, self.cache)
            self.assertEqual(output_dir, self.output)
            inputs = {name: weather(name, self.wind) for name in LOCATIONS}
            digest = watch_forecast.canonical_digest({"model": self.model, "weather": inputs})
            self.calls.append(self.wind)
            return {"status": "completed", "summary": "Готово", "forecast": combine(DAY, self.model, inputs),
                    "input_sha256": digest}

        self.fetch_patch = mock.patch.object(watch_forecast, "retrieve", side_effect=retrieve)
        self.agent_patch = mock.patch.object(forecast_agent, "run_agent", side_effect=agent_run)
        self.fetch_patch.start()
        self.agent_patch.start()
        self.addCleanup(self.fetch_patch.stop)
        self.addCleanup(self.agent_patch.stop)

    def cycle(self, agent_mode=True):
        return watch_forecast.run_cycle(DAY, self.model_path, self.cache, self.output,
                                        agent_mode=agent_mode)

    def test_agent_only_runs_for_first_and_changed_inputs_including_restart(self):
        self.assertTrue(self.cycle())
        self.assertEqual(self.calls, [8.0])
        first = (self.forecast_path.read_bytes(), self.state_path.read_bytes(),
                 self.forecast_path.stat().st_mtime_ns, self.state_path.stat().st_mtime_ns)
        self.assertFalse(self.cycle())
        self.assertEqual(self.calls, [8.0])
        self.assertEqual(first, (self.forecast_path.read_bytes(), self.state_path.read_bytes(),
                                 self.forecast_path.stat().st_mtime_ns, self.state_path.stat().st_mtime_ns))
        # A new invocation reads persisted state and still skips the model.
        with redirect_stdout(io.StringIO()):
            self.assertEqual(watch_forecast.watch(DAY, self.model_path, self.cache,
                                                  self.output, 0.01, 1, agent_mode=True), 0)
        self.assertEqual(self.calls, [8.0])
        self.wind = 12.0
        self.assertTrue(self.cycle())
        self.assertEqual(self.calls, [8.0, 12.0])
        self.assertNotEqual(first[0], self.forecast_path.read_bytes())
        self.assertEqual(json.loads(self.state_path.read_text())["controller"], "openai")

    def test_deterministic_to_agent_switch_runs_once_on_identical_inputs(self):
        self.assertTrue(self.cycle(agent_mode=False))
        deterministic = self.forecast_path.read_bytes()
        self.assertEqual(self.calls, [])
        self.assertTrue(self.cycle(agent_mode=True))
        self.assertEqual(self.calls, [8.0])
        self.assertEqual(self.forecast_path.read_bytes(), deterministic)
        self.assertFalse(self.cycle(agent_mode=True))
        self.assertEqual(self.calls, [8.0])
        self.assertEqual(json.loads(self.state_path.read_text())["controller"], "openai")

    def test_failed_agent_preserves_published_files_then_recovers(self):
        self.assertTrue(self.cycle())
        previous = self.forecast_path.read_bytes(), self.state_path.read_bytes()
        self.wind = 12.0
        failing = {"status": "failed", "summary": "Внешний сервис недоступен", "forecast": None,
                   "input_sha256": None}
        with mock.patch.object(forecast_agent, "run_agent", return_value=failing), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(watch_forecast.watch(DAY, self.model_path, self.cache,
                                                  self.output, 0.01, 1, agent_mode=True), 1)
        self.assertEqual((self.forecast_path.read_bytes(), self.state_path.read_bytes()), previous)
        self.assertTrue(self.cycle())
        self.assertNotEqual(self.forecast_path.read_bytes(), previous[0])
        self.assertEqual(self.calls, [8.0, 12.0])


class WebAgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.jobs_patch = mock.patch.object(web, "AGENT_JOBS", {})
        self.starts_patch = mock.patch.object(web, "AGENT_STARTS", collections.deque())
        self.env_patch = mock.patch.object(web, "load_agent_environment", return_value=None)
        self.available_patch = mock.patch.object(web, "agent_available", return_value=True)
        self.log_patch = mock.patch.object(web.Handler, "log_message", return_value=None)
        for patch in (self.jobs_patch, self.starts_patch, self.env_patch,
                      self.available_patch, self.log_patch):
            patch.start()
            self.addCleanup(patch.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.forecast = json.loads((web.ROOT / "results/2026-02-01-forecast.json").read_text())

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            if body is not None and not isinstance(body, (str, bytes)):
                body = json.dumps(body)
            hdr = {"Content-Type": "application/json", **(headers or {})}
            connection.request(method, path, body=body, headers=hdr)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def post(self, date="2026-02-01", **kwargs):
        return self.request("POST", "/api/agent", {"date": date}, **kwargs)

    def wait_status(self, job_id, wanted, timeout=3):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            code, state = self.request("GET", f"/api/agent/status?id={job_id}")
            self.assertEqual(code, 200)
            if state["status"] == wanted:
                return state
            time.sleep(0.01)
        self.fail(f"job {job_id} did not reach {wanted}")

    def test_missing_key_and_invalid_requests_never_call_model(self):
        with mock.patch.object(web, "run_agent") as run:
            with mock.patch.object(web, "agent_available", return_value=False):
                code, config = self.request("GET", "/api/agent/config")
                self.assertEqual(code, 200)
                self.assertFalse(config["available"])
                self.assertTrue(config["reason"])
                self.assertEqual(self.post()[0], 400)
            for date in ("2026-02-29", "../../etc/passwd", "2026-01-30", "bad"):
                self.assertEqual(self.post(date)[0], 400)
            self.assertEqual(self.request("POST", "/api/agent", {"date": "2026-02-01", "path": "/tmp"})[0], 400)
            self.assertEqual(self.post(headers={"Origin": "http://evil.test"})[0], 403)
            self.assertEqual(self.request("POST", "/api/agent", b"not json")[0], 400)
            self.assertEqual(self.request("GET", "/api/agent/status?id=unknown")[0], 404)
            run.assert_not_called()
        self.assertEqual(len(web.AGENT_JOBS), 0)
        self.assertEqual(len(web.AGENT_STARTS), 0)

    def test_post_poll_completed_trace_and_concurrent_rejection(self):
        entered = threading.Event()
        release = threading.Event()

        def run(day, *, root, on_event):
            self.assertEqual(day, DAY)
            on_event({"step": 1, "tool": "inspect_inputs", "status": "ok", "detail": "Входы проверены"})
            entered.set()
            self.assertTrue(release.wait(3))
            return {"status": "completed", "summary": "Прогноз сохранён", "events": [
                {"step": 1, "tool": "inspect_inputs", "status": "ok", "detail": "Входы проверены"}],
                "forecast": self.forecast}

        with mock.patch.object(web, "run_agent", side_effect=run) as called:
            code, started = self.post()
            self.assertEqual(code, 202)
            self.assertTrue(entered.wait(2))
            code, running = self.request("GET", f"/api/agent/status?id={started['job_id']}")
            self.assertEqual(code, 200)
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["events"][0]["tool"], "inspect_inputs")
            self.assertEqual(self.post()[0], 429)
            release.set()
            finished = self.wait_status(started["job_id"], "completed")
            self.assertEqual(finished["summary"], "Прогноз сохранён")
            self.assertEqual(finished["forecast"], self.forecast)
            self.assertEqual(finished["events"][0]["step"], 1)
            called.assert_called_once()

    def test_seven_starts_in_one_hour_are_rejected(self):
        def run(day, *, root, on_event):
            return {"status": "completed", "summary": "Готово", "events": [], "forecast": self.forecast}

        with mock.patch.object(web, "run_agent", side_effect=run) as called:
            for _ in range(6):
                code, started = self.post()
                self.assertEqual(code, 202)
                self.wait_status(started["job_id"], "completed")
            self.assertEqual(self.post()[0], 429)
            self.assertEqual(called.call_count, 6)
            self.assertEqual(len(web.AGENT_STARTS), 6)


if __name__ == "__main__":
    unittest.main()
