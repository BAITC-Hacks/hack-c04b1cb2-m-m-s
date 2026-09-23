"""Offline end-to-end checks for resumable station forecast campaigns."""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import forecast_agent
import run_campaign as campaign
from power_model import FIELDS, train
from station_profile import DEFAULT_STATION, station_digest

REPO = pathlib.Path(__file__).resolve().parents[1]
FIRST = dt.date(2026, 1, 31)
STATION = {"id": "campaign-test", "name": "Campaign Test Farm",
           "locations": {"turbine-1": [44.125, 79.875],
                         "turbine-2": [44.126, 79.876]}}


def tool(name: str, args: dict | None = None, n: int = 1) -> dict:
    return {"type": "function_call", "name": name,
            "arguments": json.dumps(args or {}), "call_id": f"call_{n}"}


def reply(*items: dict, text: str = "") -> dict:
    output = list(items)
    if text:
        output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
    return {"output": output, "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = pathlib.Path(self.temp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.csv_dir = self.base / "history"
        self.csv_dir.mkdir()
        self.output = self.base / "campaign-output"
        self._make_csvs()
        self.calls: list[tuple[str, str]] = []
        self.attempts: dict[str, int] = {}
        self.key_patch = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key"})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.weather_patch1 = mock.patch.object(forecast_agent, "retrieve", side_effect=self._weather)
        self.weather_patch2 = mock.patch.object(campaign, "retrieve", side_effect=self._weather)
        self.weather_patch1.start()
        self.weather_patch2.start()
        self.addCleanup(self.weather_patch1.stop)
        self.addCleanup(self.weather_patch2.stop)
        self.llm_patch = mock.patch.object(forecast_agent, "_response", side_effect=self._response)
        self.llm_patch.start()
        self.addCleanup(self.llm_patch.stop)

    def _make_csvs(self):
        for turbine_index, name in enumerate(("turbine-1", "turbine-2")):
            with (self.csv_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(FIELDS.values()))
                writer.writeheader()
                # Complete hours well before the first decision; non-constant wind
                # ensures the trainer builds a real empirical curve.
                for hour_index in range(48):
                    stamp = dt.datetime(2025, 12, 20) + dt.timedelta(hours=hour_index)
                    wind = 4.0 + (hour_index % 12) * 0.5 + turbine_index * 0.2
                    power = min(0.95, 0.05 + wind * 0.07)
                    for minute in (0, 10, 20, 30, 40, 50):
                        writer.writerow({FIELDS["time"]: (stamp + dt.timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S"),
                                         FIELDS["wind"]: wind,
                                         FIELDS["power"]: power,
                                         FIELDS["temperature"]: 8.0})

    def _weather(self, day, turbine, cache_dir, refresh=False, locations=None):
        selected = locations or {
            "turbine-1": [43.645150, 78.535604],
            "turbine-2": [43.643198, 78.538828],
        }
        lat, lon = selected[turbine]
        decision = dt.datetime.combine(day, dt.time(6), tzinfo=dt.timezone.utc)
        run = decision - dt.timedelta(hours=12)
        result = {"source": "offline fixture", "request_url": f"https://offline.invalid/{lat}/{lon}/{day}",
                  "model_run_utc": run.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "decision_time_utc": decision.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "assumed_publication_delay_hours": 6, "turbine": turbine,
                  "coordinates": {"latitude": lat, "longitude": lon},
                  "units": {"wind_speed_10m": "m/s", "wind_speed_100m": "m/s", "temperature_2m": "°C"},
                  "forecast": [{"time_utc": (decision.replace(tzinfo=None) + dt.timedelta(hours=i)).isoformat(),
                                "wind_speed_10m": 6.0, "wind_speed_100m": 7.0 + i / 100,
                                "temperature_2m": 8.0} for i in range(48)]}
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{day}-{turbine}.json").write_text(json.dumps(result))
        return result

    def _response(self, payload, _key):
        prompt = payload["input"][0]["content"]
        day = prompt.rsplit(" ", 1)[-1].rstrip(".")
        cutoff = min(dt.date.fromisoformat(day) - dt.timedelta(days=1), campaign.MODEL_LAST_DAY)
        active_output = getattr(self, "active_output", self.output)
        model_path = active_output / "models" / f"power-curve-{cutoff}.json"
        # A new run_agent call begins with only the original user message.
        if len(payload["input"]) == 1:
            self.attempts[day] = self.attempts.get(day, 0) + 1
            if day in getattr(self, "fail_all_attempts", set()):
                raise forecast_agent.AgentFailure("offline simulated API failure")
        attempt = self.attempts[day]
        key = (day, attempt)
        cursor = getattr(self, "cursor", {})
        index = cursor.get(key, 0)
        plans = getattr(self, "plans", {})
        if key not in plans:
            plan = self._plan(day, model_path)
            if getattr(self, "retry_training", False) and not model_path.exists() and attempt == 1:
                plan.insert(2, ("train_model", {}))
            plans[key] = plan
            self.plans = plans
        plan = plans[key]
        if len(payload["input"]) > 1 and payload["input"][-1].get("type") == "function_call_output":
            last_result = json.loads(payload["input"][-1]["output"])
            previous_tool = next((item.get("name") for item in reversed(payload["input"][:-1])
                                  if item.get("type") == "function_call"), None)
            if previous_tool == "inspect_inputs" and not last_result.get("ok"):
                plan.insert(index, ("train_model", {}))
            elif previous_tool == "train_model" and not last_result.get("ok") and index < len(plan):
                if plan[index][0] != "train_model":
                    plan.insert(index, ("train_model", {}))
        if index >= len(plan):
            return reply(text="Прогноз сохранён; вычисления выполнены численной моделью.")
        name, args = plan[index]
        cursor[key] = index + 1
        self.cursor = cursor
        self.calls.append((day, name))
        return reply(tool(name, args, index + 1))

    def _plan(self, day, model_path):
        plan = [("inspect_station", {})]
        if model_path.exists():
            plan.append(("inspect_inputs", {}))
        else:
            plan.append(("train_model", {}))
        plan.extend(("get_weather", {"turbine": name, "refresh": False})
                    for name in ("turbine-1", "turbine-2"))
        plan.extend([("calculate_forecast", {}), ("publish_forecast", {})])
        return plan

    def launch_campaign(self, days=1, *, output_dir=None):
        self.active_output = pathlib.Path(output_dir) if output_dir is not None else self.output
        return campaign.run_campaign(root=self.root, station=copy.deepcopy(STATION),
                                      input_dir=self.csv_dir, first_day=FIRST, days=days,
                                      output_dir=self.active_output)

    def test_agent_trains_station_model_and_uses_configured_coordinates(self):
        result = self.launch_campaign()
        self.assertEqual(result["status"], "completed", result["summary"])
        self.assertEqual(result["completed_days"], 1)
        self.assertIn(("2026-01-31", "train_model"), self.calls)
        model_path = self.output / "models/power-curve-2026-01-30.json"
        model = json.loads(model_path.read_text())
        self.assertEqual(model["station"], STATION)
        self.assertEqual(model["station_sha256"], station_digest(STATION))
        self.assertEqual(model["trained_through_inclusive"], "2026-01-30")
        self.assertEqual(model["turbines"]["turbine-1"]["source_sha256"],
                         hashlib.sha256((self.csv_dir / "turbine-1.csv").read_bytes()).hexdigest())
        weather_doc = json.loads((self.output / "weather-cache/2026-01-31-turbine-1.json").read_text())
        self.assertEqual(weather_doc["coordinates"], {"latitude": 44.125, "longitude": 79.875})
        self.assertEqual(result["forecast"]["station"], STATION)

    def test_multi_day_campaign_advances_then_resume_skips_without_llm(self):
        result = self.launch_campaign(days=2)
        self.assertEqual(result["status"], "completed", result["summary"])
        self.assertEqual(result["completed_days"], 2)
        self.assertTrue((self.output / "2026-01-31-agent-forecast.json").is_file())
        self.assertTrue((self.output / "2026-02-01-agent-forecast.json").is_file())
        first_calls = list(self.calls)
        self.assertIn(("2026-02-01", "inspect_station"), first_calls)
        self.assertIn(("2026-02-01", "train_model"), first_calls)
        again = self.launch_campaign(days=2)
        self.assertEqual(again["status"], "completed")
        self.assertEqual(again["completed_days"], 2)
        self.assertEqual(self.calls, first_calls)

    def test_failure_on_second_day_preserves_first_and_restart_resumes_there(self):
        self.fail_all_attempts = {"2026-02-01"}
        failed = self.launch_campaign(days=2)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["completed_days"], 1)
        first_path = self.output / "2026-01-31-agent-forecast.json"
        first_bytes = first_path.read_bytes()
        self.assertTrue(first_path.exists())
        self.assertEqual(self.attempts["2026-02-01"], 2)
        del self.fail_all_attempts
        self.calls.clear()
        self.cursor = {}
        resumed = self.launch_campaign(days=2)
        self.assertEqual(resumed["status"], "completed", resumed["summary"])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertFalse(any(day == "2026-01-31" for day, _ in self.calls))
        self.assertTrue(any(day == "2026-02-01" for day, _ in self.calls))

    def test_corrupt_saved_forecast_is_recomputed_not_skipped(self):
        self.assertEqual(self.launch_campaign()["status"], "completed")
        path = self.output / "2026-01-31-agent-forecast.json"
        path.write_text('{"forecast": []}')
        self.calls.clear()
        self.cursor = {}
        result = self.launch_campaign()
        self.assertEqual(result["status"], "completed", result["summary"])
        self.assertTrue(self.calls)
        self.assertEqual(len(json.loads(path.read_text())["forecast"]), 48)
        model_path = self.output / "models/power-curve-2026-01-30.json"
        model_path.write_text("not-json")
        self.calls.clear()
        self.cursor = {}
        repaired = self.launch_campaign()
        self.assertEqual(repaired["status"], "completed", repaired["summary"])
        self.assertIn(("2026-01-31", "train_model"), self.calls)
        self.assertEqual(json.loads(model_path.read_text())["station"], STATION)

    def test_training_tool_recovers_after_transient_local_training_error(self):
        self.retry_training = True
        real_train = forecast_agent.train
        training_calls = 0
        def flaky_train(directory, cutoff):
            nonlocal training_calls
            training_calls += 1
            if training_calls == 1:
                raise ValueError("temporary fixture fault")
            return real_train(directory, cutoff)
        with mock.patch.object(forecast_agent, "train",
                               side_effect=flaky_train) as trainer:
            result = self.launch_campaign()
        self.assertEqual(result["status"], "completed", result["summary"])
        self.assertEqual(trainer.call_count, 2)
        self.assertTrue((self.output / "models/power-curve-2026-01-30.json").is_file())
        self.assertIn(("2026-01-31", "train_model"), self.calls)

    def test_explicit_output_dir_rejects_changed_csv(self):
        self.assertEqual(self.launch_campaign()["status"], "completed")
        with (self.csv_dir / "turbine-1.csv").open("a", encoding="utf-8") as stream:
            stream.write("changed\n")
        self.calls.clear()
        result = self.launch_campaign()
        self.assertEqual(result["status"], "failed")
        self.assertIn("относится к другой станции или истории", result["summary"])
        self.assertEqual(self.calls, [])

    def test_csv_campaign_keeps_legacy_default_path_and_state_format(self):
        profile_hash = station_digest(STATION)
        dataset_hash = campaign.canonical_digest(campaign.source_hashes(self.csv_dir))
        legacy_dir = self.root / "predictions/campaigns" / (
            f"{STATION['id']}-{profile_hash[:10]}-{dataset_hash[:10]}")
        first = campaign.run_campaign(root=self.root, station=copy.deepcopy(STATION),
            input_dir=self.csv_dir, first_day=FIRST, days=1)
        self.assertEqual(first["status"], "completed", first["summary"])
        self.assertTrue((legacy_dir / "campaign-state.json").is_file())
        state_path = legacy_dir / "campaign-state.json"
        state = json.loads(state_path.read_text())
        state.pop("mode")
        state_path.write_text(json.dumps(state))
        self.calls.clear()
        resumed = campaign.run_campaign(root=self.root, station=copy.deepcopy(STATION),
            input_dir=self.csv_dir, first_day=FIRST, days=1)
        self.assertEqual(resumed["status"], "completed", resumed["summary"])
        self.assertEqual(self.calls, [])

    def test_protected_output_and_models_symlink_are_rejected_without_touching_baseline(self):
        (self.root / "models").mkdir()
        (self.root / "results").mkdir()
        protected_model = self.root / "models/power-curve-2026-01-30.json"
        protected_result = self.root / "results/2026-01-31-forecast.json"
        protected_model.write_text("model sentinel")
        protected_result.write_text("result sentinel")
        baseline = protected_model.read_bytes(), protected_result.read_bytes()

        direct = campaign.run_campaign(root=self.root, station=copy.deepcopy(STATION),
            input_dir=self.csv_dir, first_day=FIRST, days=1, output_dir=self.root)
        self.assertEqual(direct["status"], "failed")
        self.assertIn("не должна совпадать", direct["summary"])

        unsafe = self.base / "unsafe-output"
        unsafe.mkdir()
        (unsafe / "models").symlink_to(self.root / "models", target_is_directory=True)
        linked = campaign.run_campaign(root=self.root, station=copy.deepcopy(STATION),
            input_dir=self.csv_dir, first_day=FIRST, days=1, output_dir=unsafe)
        self.assertEqual(linked["status"], "failed")
        self.assertIn("не должна совпадать", linked["summary"])
        self.assertEqual(baseline, (protected_model.read_bytes(), protected_result.read_bytes()))

    def test_station_agent_default_workspace_isolated_from_legacy_outputs(self):
        profile_dir = self.root / "predictions/stations" / f"{STATION['id']}-{station_digest(STATION)[:12]}"
        self.active_output = profile_dir
        result = forecast_agent.run_agent(FIRST, root=self.root, station=copy.deepcopy(STATION),
                                         input_dir=self.csv_dir)
        self.assertEqual(result["status"], "completed", result["summary"])
        self.assertTrue((profile_dir / "2026-01-31-agent-forecast.json").is_file())
        self.assertFalse((self.root / "models/power-curve-2026-01-30.json").exists())
        self.assertFalse((self.root / "predictions/2026-01-31-agent-forecast.json").exists())

    def test_future_cutoff_or_foreign_station_model_is_rejected_then_retrained(self):
        for variant in ("future", "foreign"):
            with self.subTest(variant=variant):
                # Isolate each scenario with a fresh campaign output directory.
                output = self.base / f"campaign-{variant}"
                self.active_output = output
                model_path = output / "models/power-curve-2026-01-30.json"
                model_path.parent.mkdir(parents=True)
                cutoff = dt.date(2026, 1, 31) if variant == "future" else dt.date(2026, 1, 30)
                model = train(self.csv_dir, cutoff)
                model_station = copy.deepcopy(STATION)
                if variant == "foreign":
                    model_station["id"] = "different-station"
                model["station"] = model_station
                model["station_sha256"] = station_digest(model_station)
                model_path.write_text(json.dumps(model))
                self.calls.clear()
                self.cursor = {}
                result = campaign.run_campaign(root=self.root, station=copy.deepcopy(STATION),
                    input_dir=self.csv_dir, first_day=FIRST, days=1, output_dir=output)
                self.assertEqual(result["status"], "completed", result["summary"])
                self.assertIn(("2026-01-31", "train_model"), self.calls)
                repaired = json.loads(model_path.read_text())
                self.assertEqual(repaired["trained_through_inclusive"], "2026-01-30")
                self.assertEqual(repaired["station"], STATION)

    def test_prepared_models_run_two_days_without_training_and_resume_offline(self):
        ready_root = self.base / "ready-project"
        (ready_root / "models").mkdir(parents=True)
        for name in ("power-curve-2026-01-30.json", "power-curve-2026-01-31.json"):
            (ready_root / "models" / name).write_bytes((REPO / "models" / name).read_bytes())
        before = {name: (ready_root / "models" / name).read_bytes()
                  for name in ("power-curve-2026-01-30.json", "power-curve-2026-01-31.json")}
        self.root = ready_root
        self.active_output = self.output
        # Reset the mock's plan state, then use the real agent with the existing
        # offline weather/Responses fixtures. No training tool should be needed.
        self.cursor = {}
        self.plans = {}
        self.attempts = {}
        result = campaign.run_campaign(root=ready_root, station=copy.deepcopy(DEFAULT_STATION),
            first_day=FIRST, days=2, output_dir=self.output)
        self.assertEqual(result["status"], "completed", result["summary"])
        self.assertEqual(result["completed_days"], 2)
        self.assertNotIn("train_model", [name for _day, name in self.calls])
        for name, contents in before.items():
            self.assertEqual((ready_root / "models" / name).read_bytes(), contents)
        campaign_models = list((self.output / "models").glob("*.json"))
        self.assertEqual(len(campaign_models), 2)
        for model_path in campaign_models:
            model = json.loads(model_path.read_text())
            self.assertEqual(model["station"], DEFAULT_STATION)
            self.assertEqual(model["station_sha256"], station_digest(DEFAULT_STATION))
        calls = list(self.calls)
        with mock.patch.object(forecast_agent, "_response", side_effect=AssertionError("LLM called")):
            resumed = campaign.run_campaign(root=ready_root, station=copy.deepcopy(DEFAULT_STATION),
                first_day=FIRST, days=2, output_dir=self.output)
        self.assertEqual(resumed["status"], "completed", resumed["summary"])
        self.assertEqual(resumed["completed_days"], 2)
        self.assertEqual(self.calls, calls)

    def test_prepared_source_change_invalidates_explicit_campaign_identity(self):
        ready_root = self.base / "changed-source-project"
        (ready_root / "models").mkdir(parents=True)
        for name in ("power-curve-2026-01-30.json", "power-curve-2026-01-31.json"):
            (ready_root / "models" / name).write_bytes((REPO / "models" / name).read_bytes())
        first = campaign.run_campaign(root=ready_root, station=copy.deepcopy(DEFAULT_STATION),
            first_day=FIRST, days=1, output_dir=self.output)
        self.assertEqual(first["status"], "completed", first["summary"])
        source_path = ready_root / "models/power-curve-2026-01-30.json"
        source_path.write_bytes(source_path.read_bytes() + b"\n")
        self.calls.clear()
        changed = campaign.run_campaign(root=ready_root, station=copy.deepcopy(DEFAULT_STATION),
            first_day=FIRST, days=1, output_dir=self.output)
        self.assertEqual(changed["status"], "failed")
        self.assertIn("относится к другой станции или истории", changed["summary"])
        self.assertEqual(self.calls, [])

    def test_prepared_corrupt_campaign_copy_is_restored_before_recomputation(self):
        ready_root = self.base / "copy-repair-project"
        (ready_root / "models").mkdir(parents=True)
        for name in ("power-curve-2026-01-30.json", "power-curve-2026-01-31.json"):
            (ready_root / "models" / name).write_bytes((REPO / "models" / name).read_bytes())
        first = campaign.run_campaign(root=ready_root, station=copy.deepcopy(DEFAULT_STATION),
            first_day=FIRST, days=1, output_dir=self.output)
        self.assertEqual(first["status"], "completed", first["summary"])
        local_copy = self.output / "models/power-curve-2026-01-30.json"
        local_copy.write_text("damaged copy")
        self.calls.clear()
        self.cursor = {}
        self.plans = {}
        self.attempts = {}
        repaired = campaign.run_campaign(root=ready_root, station=copy.deepcopy(DEFAULT_STATION),
            first_day=FIRST, days=1, output_dir=self.output)
        self.assertEqual(repaired["status"], "completed", repaired["summary"])
        self.assertTrue(self.calls)
        model = json.loads(local_copy.read_text())
        self.assertEqual(model["trained_through_inclusive"], "2026-01-30")
        self.assertEqual(model["station"], DEFAULT_STATION)
        self.assertNotIn("train_model", [name for _day, name in self.calls])

    def test_prepared_campaign_completes_full_historical_period_offline(self):
        ready_root = self.base / "full-ready-project"
        (ready_root / "models").mkdir(parents=True)
        for name in ("power-curve-2026-01-30.json", "power-curve-2026-01-31.json"):
            (ready_root / "models" / name).write_bytes((REPO / "models" / name).read_bytes())
        self.calls.clear()
        self.cursor = {}
        self.plans = {}
        self.attempts = {}
        result = campaign.run_campaign(root=ready_root, station=copy.deepcopy(DEFAULT_STATION),
            first_day=FIRST, days=29, output_dir=self.output)
        self.assertEqual(result["status"], "completed", result["summary"])
        self.assertEqual(result["completed_days"], 29)
        self.assertEqual(result["forecast"]["forecast_date"], "2026-02-28")
        self.assertNotIn("train_model", [name for _day, name in self.calls])

    def test_prepared_mode_rejects_other_station_and_missing_or_invalid_model(self):
        with self.assertRaises(ValueError):
            campaign.load_prepared_models(REPO, STATION)
        broken = self.base / "broken-project"
        (broken / "models").mkdir(parents=True)
        with self.assertRaises(OSError):
            campaign.load_prepared_models(broken, DEFAULT_STATION)
        (broken / "models/power-curve-2026-01-30.json").write_text("not-json")
        with self.assertRaises(ValueError):
            campaign.load_prepared_models(broken, DEFAULT_STATION)

        for metadata in ("station", "station_sha256"):
            with self.subTest(metadata=metadata):
                marked = self.base / f"marked-{metadata}"
                (marked / "models").mkdir(parents=True)
                for name in ("power-curve-2026-01-30.json", "power-curve-2026-01-31.json"):
                    model = json.loads((REPO / "models" / name).read_text())
                    if metadata == "station":
                        model[metadata] = copy.deepcopy(STATION)
                    else:
                        model[metadata] = station_digest(STATION)
                    (marked / "models" / name).write_text(json.dumps(model))
                with self.assertRaises(ValueError):
                    campaign.load_prepared_models(marked, DEFAULT_STATION)


if __name__ == "__main__":
    unittest.main()
