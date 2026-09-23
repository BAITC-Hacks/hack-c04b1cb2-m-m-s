"""Offline integration tests for the bounded tool-calling forecast controller."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import forecast_agent as agent
from run_forecast import combine

DAY = dt.date(2026, 2, 1)
REPO = pathlib.Path(__file__).resolve().parents[1]


def weather(turbine: str) -> dict:
    start = dt.datetime(2026, 2, 1, 6)
    return {"turbine": turbine, "decision_time_utc": "2026-02-01T06:00:00Z",
            "model_run_utc": "2026-01-31T18:00:00Z", "request_url": "https://example.invalid/fixed",
            "units": {"wind_speed_100m": "m/s"},
            "forecast": [{"time_utc": (start + dt.timedelta(hours=i)).isoformat(),
                          "wind_speed_100m": 5.0 + i / 20} for i in range(48)]}


def call(name: str, args: dict | None = None, index: int = 1) -> dict:
    return {"type": "function_call", "name": name,
            "arguments": json.dumps(args or {}), "call_id": f"call_{index}"}


def answer(*items: dict, text: str = "") -> dict:
    output = list(items)
    if text:
        output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
    return {"output": output, "usage": {"input_tokens": 10, "output_tokens": 2,
                                         "total_tokens": 12}}


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        (self.root / "models").mkdir()
        self.model = json.loads((REPO / "models/power-curve-2026-01-31.json").read_text())
        (self.root / "models/power-curve-2026-01-31.json").write_text(json.dumps(self.model))
        self.key = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret", "OPENAI_MODEL": "gpt-5.4-mini"})
        self.key.start()
        self.addCleanup(self.key.stop)
        self.fetch = mock.patch.object(agent, "retrieve", side_effect=lambda day, turbine, cache, refresh=False: weather(turbine))
        self.fetch.start()
        self.addCleanup(self.fetch.stop)

    def run_script(self, sequence: list[dict]) -> dict:
        with mock.patch.object(agent, "_response", side_effect=sequence) as transport:
            result = agent.run_agent(DAY, root=self.root)
            self.assertLessEqual(transport.call_count, 8)
            if transport.call_args:
                payload = transport.call_args.args[0]
                self.assertFalse(payload["store"])
                self.assertFalse(payload["parallel_tool_calls"])
                self.assertEqual(payload["max_output_tokens"], 1800)
        return result

    def test_success_recovers_from_missing_weather_then_matches_core(self):
        sequence = [
            answer(call("calculate_forecast")),
            answer(call("inspect_inputs")),
            answer(call("get_weather", {"turbine": "turbine-1", "refresh": False})),
            answer(call("get_weather", {"turbine": "turbine-2", "refresh": False})),
            answer(call("calculate_forecast")),
            answer(call("publish_forecast")),
            answer(text="Прогноз на 48 часов опубликован; значения нормализованы, точность февраля неизвестна."),
        ]
        result = self.run_script(sequence)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["events"][0]["status"], "error")
        self.assertEqual(result["usage"]["total_tokens"], 84)
        expected = combine(DAY, self.model, {name: weather(name) for name in ("turbine-1", "turbine-2")})
        self.assertEqual(result["forecast"], expected)
        canonical = json.dumps({"model": self.model,
                                "weather": {name: weather(name) for name in ("turbine-1", "turbine-2")}},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        self.assertEqual(result["input_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
        saved = json.loads((self.root / "predictions/2026-02-01-agent-forecast.json").read_text())
        self.assertEqual(saved, expected)
        self.assertTrue((self.root / "predictions/2026-02-01-agent-report.json").exists())

    def test_early_final_fails(self):
        result = self.run_script([answer(text="Готово.")])
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["forecast"])

    def test_refresh_after_publish_requires_recalculation_and_republish(self):
        cycle = agent._Cycle(DAY, self.root,
                             self.root / "models/power-curve-2026-01-31.json",
                             self.root / "weather-cache", self.root / "predictions")
        cycle.invoke("inspect_inputs", {})
        for turbine in ("turbine-1", "turbine-2"):
            cycle.invoke("get_weather", {"turbine": turbine, "refresh": False})
        cycle.invoke("calculate_forecast", {})
        cycle.invoke("publish_forecast", {})
        self.assertTrue(cycle.is_published())
        cycle.invoke("get_weather", {"turbine": "turbine-1", "refresh": True})
        self.assertFalse(cycle.is_published())
        with self.assertRaises(agent.AgentFailure):
            cycle.invoke("publish_forecast", {})

    def test_model_reinspection_invalidates_published_forecast(self):
        model_path = self.root / "models/power-curve-2026-01-31.json"
        cycle = agent._Cycle(DAY, self.root, model_path,
                             self.root / "weather-cache", self.root / "predictions")
        cycle.invoke("inspect_inputs", {})
        for turbine in ("turbine-1", "turbine-2"):
            cycle.invoke("get_weather", {"turbine": turbine, "refresh": False})
        cycle.invoke("calculate_forecast", {})
        cycle.invoke("publish_forecast", {})
        self.assertTrue(cycle.is_published())
        revised = json.loads(model_path.read_text())
        revised["turbines"]["turbine-1"]["curve"]["values"][0] = 0.123456
        model_path.write_text(json.dumps(revised))
        cycle.invoke("inspect_inputs", {})
        self.assertFalse(cycle.is_published())
        with self.assertRaises(agent.AgentFailure):
            cycle.invoke("publish_forecast", {})

    def test_unknown_tool_path_and_bad_hours_rejected(self):
        for name, raw in [("arbitrary_exec", "{}"), ("get_weather", '{"turbine":"../secret","refresh":false}'),
                          ("get_weather", '{"turbine":"turbine-1","refresh":false,"path":"/tmp/x"}'),
                          ("inspect_forecast", '{"start_hour":NaN,"end_hour":5}')]:
            with self.subTest(name=name, raw=raw), self.assertRaises(agent.AgentFailure):
                agent._arguments(name, raw)
        result = self.run_script([answer(call("arbitrary_exec")), answer(text="Done")])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["events"][0]["status"], "error")

    def test_api_failure_sanitized_and_no_key_leak(self):
        with mock.patch.object(agent.urllib.request, "urlopen", side_effect=agent.urllib.error.HTTPError(
                agent.API_URL, 401, "test-secret", {}, None)):
            result = agent.run_agent(DAY, root=self.root)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("test-secret", json.dumps(result))
        self.assertIn("HTTP 401", result["summary"])

    def test_malformed_response_and_incomplete_http_are_safe(self):
        with mock.patch.object(agent, "_response", return_value={"output": ["bad item"]}):
            result = agent.run_agent(DAY, root=self.root)
        self.assertEqual(result["status"], "failed")
        with mock.patch.object(agent.urllib.request, "urlopen",
                               side_effect=agent.http.client.IncompleteRead(b"bad")):
            result = agent.run_agent(DAY, root=self.root)
        self.assertEqual(result["status"], "failed")

    def test_turn_bound(self):
        result = self.run_script([answer(call("inspect_inputs", index=i)) for i in range(8)])
        self.assertEqual(result["status"], "failed")
        self.assertIn("лимит 8", result["summary"])
        self.assertEqual(len(result["events"]), 8)

    def test_caller_supplied_paths(self):
        chosen_model = self.root / "chosen-model.json"
        chosen_model.write_text(json.dumps(self.model))
        chosen_cache = self.root / "chosen-cache"
        chosen_output = self.root / "chosen-output"
        sequence = [answer(call("inspect_inputs")),
                    answer(call("get_weather", {"turbine": "turbine-1", "refresh": False})),
                    answer(call("get_weather", {"turbine": "turbine-2", "refresh": False})),
                    answer(call("calculate_forecast")), answer(call("publish_forecast")),
                    answer(text="Прогноз опубликован.")]
        with mock.patch.object(agent, "_response", side_effect=sequence):
            result = agent.run_agent(DAY, root=self.root, model_path=chosen_model,
                                     cache_dir=chosen_cache, output_dir=chosen_output)
        self.assertEqual(result["status"], "completed")
        self.assertTrue((chosen_output / "2026-02-01-agent-forecast.json").exists())
        self.assertTrue((chosen_output / "2026-02-01-agent-report.json").exists())


if __name__ == "__main__":
    unittest.main()
