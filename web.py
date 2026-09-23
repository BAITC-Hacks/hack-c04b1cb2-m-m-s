"""Local, dependency-free dashboard for the February wind-power forecast."""
from __future__ import annotations

import datetime as dt
import argparse
import collections
import copy
import json
import os
import pathlib
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from forecast_weather import LOCATIONS, retrieve
from run_forecast import combine
from power_model import validate_model
from storage import atomic_write
from agent_settings import load_agent_environment
from forecast_agent import DEFAULT_MODEL, agent_available, run_agent
from station_profile import load_station

ROOT = pathlib.Path(__file__).resolve().parent
HTML = ROOT / "static" / "index.html"
FEBRUARY_CSV = ROOT / "results" / "february-hourly.csv"
MODELS = ROOT / "models"
CACHE = ROOT / "weather-cache"
RESULTS = ROOT / "results"
RECALCULATED = ROOT / "predictions"
DATE_PATTERN = r"2026-(?:01-31|02-(?:0[1-9]|1[0-9]|2[0-8]))"
AGENT_LOCK = threading.Lock()
AGENT_JOBS: dict[str, dict] = {}
AGENT_STARTS: collections.deque[float] = collections.deque()


def _reserve_job(*, kind: str, date: str) -> str:
    """Share the single running slot and six-start hourly budget across both modes."""
    with AGENT_LOCK:
        if any(job["status"] == "running" for job in AGENT_JOBS.values()):
            raise RuntimeError("Агент уже выполняет расчёт. Дождитесь завершения и повторите запуск.")
        now = time.monotonic()
        while AGENT_STARTS and now - AGENT_STARTS[0] >= 3600:
            AGENT_STARTS.popleft()
        if len(AGENT_STARTS) >= 6:
            raise RuntimeError("Достигнут лимит демо: 6 запусков агента в час. Сохранённые прогнозы доступны.")
        job_id = secrets.token_urlsafe(18)
        if len(AGENT_JOBS) >= 20:
            del AGENT_JOBS[next(iter(AGENT_JOBS))]
        AGENT_JOBS[job_id] = {"job_id": job_id, "date": date, "kind": kind,
                              "status": "running", "events": [], "summary": "", "forecast": None}
        AGENT_STARTS.append(now)
    return job_id


def _campaign_settings() -> tuple[dict | None, pathlib.Path | None, str]:
    """Resolve server-owned inputs; never return local paths in error text."""
    try:
        load_agent_environment(ROOT)
        if not agent_available():
            return None, None, "Для автономного прохода нужен серверный ключ OpenAI."
        configured = os.environ.get("WIND_TRAINING_DIR", "").strip()
        if not configured:
            return None, None, "Для автономного прохода задайте WIND_TRAINING_DIR на сервере."
        station_setting = os.environ.get("WIND_STATION_FILE", "").strip()
        station_path = pathlib.Path(station_setting) if station_setting else pathlib.Path("stations/example.json")
        if not station_path.is_absolute():
            station_path = ROOT / station_path
        station = load_station(station_path)
        input_dir = pathlib.Path(configured)
        if not input_dir.is_absolute():
            input_dir = ROOT / input_dir
        if not input_dir.is_dir() or any(not (input_dir / f"{name}.csv").is_file() for name in LOCATIONS):
            return None, None, "Не найдены CSV обеих турбин в WIND_TRAINING_DIR."
        return station, input_dir, ""
    except (OSError, ValueError, TypeError):
        return None, None, "Профиль станции или каталог CSV недоступен либо некорректен."


def run_campaign_job(**kwargs) -> dict:
    """Late import keeps the published dashboard usable before campaign setup."""
    from run_campaign import run_campaign
    return run_campaign(**kwargs)


def start_agent_job(day: dt.date) -> str:
    """Run one daily agent job under the shared public demo budget."""
    load_agent_environment(ROOT)
    if not agent_available():
        raise ValueError("AI-агент не настроен на сервере; опубликованные прогнозы доступны.")
    job_id = _reserve_job(kind="agent", date=day.isoformat())

    def event_received(event: dict) -> None:
        with AGENT_LOCK:
            AGENT_JOBS[job_id]["events"].append(copy.deepcopy(event))

    def work() -> None:
        try:
            report = run_agent(day, root=ROOT, on_event=event_received)
            with AGENT_LOCK:
                AGENT_JOBS[job_id].update(report)
        except Exception:
            # Do not expose credentials, upstream response bodies, or local paths.
            with AGENT_LOCK:
                AGENT_JOBS[job_id].update(status="failed", summary="Расчёт агента прерван. Предыдущие прогнозы сохранены; повторите запуск.")

    threading.Thread(target=work, daemon=True, name=f"forecast-agent-{day}").start()
    return job_id


def start_campaign_job() -> str:
    station, input_dir, reason = _campaign_settings()
    if reason:
        raise ValueError(reason)
    job_id = _reserve_job(kind="campaign", date="2026-01-31")
    with AGENT_LOCK:
        AGENT_JOBS[job_id].update(completed_days=0, total_days=29, station=station)
    finished_days: set[str] = set()

    def event_received(event: dict) -> None:
        safe = {key: copy.deepcopy(event[key]) for key in
                ("step", "tool", "status", "forecast_date") if key in event}
        detail = event.get("detail", "")
        if event.get("status") == "error":
            safe["detail"] = "Шаг завершился ошибкой; проверьте серверный журнал."
        else:
            safe["detail"] = str(detail)[:220]
        result = event.get("result")
        if isinstance(result, dict):
            public_fields = {
                "inspect_inputs": ("model_cutoff", "weather_cache", "loaded_turbines"),
                "get_weather": ("turbine", "hours", "model_run_utc", "refresh"),
                "calculate_forecast": ("hours", "mean_normalized_power", "largest_hourly_ramp"),
                "inspect_forecast": ("start_hour", "end_hour", "hours_inspected"),
                "compare_published": ("baseline", "mean_delta", "max_absolute_hourly_delta"),
                "publish_forecast": ("saved", "artifact", "hours", "mean_normalized_power"),
                "inspect_station": ("training_cutoff", "model_available", "training_configured"),
                "train_model": ("trained_through_inclusive", "data_quality", "calibration"),
                "campaign_day": ("completed_days", "total_days"),
            }.get(event.get("tool"), ())
            safe["result"] = {key: copy.deepcopy(result[key]) for key in public_fields if key in result}
            if event.get("tool") == "inspect_station" and isinstance(result.get("station"), dict):
                safe["result"]["station_name"] = str(result["station"].get("name", ""))[:120]
        arguments = event.get("arguments")
        if isinstance(arguments, dict):
            public_args = {"get_weather": ("turbine", "refresh"),
                           "inspect_forecast": ("start_hour", "end_hour")}.get(event.get("tool"), ())
            safe["arguments"] = {key: copy.deepcopy(arguments[key]) for key in public_args if key in arguments}
        with AGENT_LOCK:
            AGENT_JOBS[job_id]["events"].append(safe)
            if (safe.get("tool") == "campaign_day" and safe.get("status") in ("ok", "skipped")
                    and isinstance(safe.get("forecast_date"), str)):
                finished_days.add(safe["forecast_date"])
            AGENT_JOBS[job_id]["completed_days"] = len(finished_days)

    def work() -> None:
        try:
            report = run_campaign_job(root=ROOT, station=station, input_dir=input_dir,
                                      on_event=event_received)
            forecast = report.get("forecast")
            completed = (report.get("status") == "completed" and isinstance(forecast, dict)
                         and isinstance(forecast.get("forecast_date"), str)
                         and re.fullmatch(DATE_PATTERN, forecast["forecast_date"]) is not None
                         and isinstance(forecast.get("forecast"), list)
                         and len(forecast["forecast"]) == 48)
            if not completed:
                print(f"Autonomous campaign failed: {report.get('summary', 'no valid result')}",
                      file=sys.stderr, flush=True)
            with AGENT_LOCK:
                # Keep the sanitized live journal; runner reports may contain paths.
                AGENT_JOBS[job_id].update(status="completed" if completed else "failed",
                    summary=(str(report.get("summary", ""))[:500] if completed
                             else "Автономный проход не завершён. Проверьте серверный журнал и повторите запуск."),
                    station=report.get("station", station),
                    completed_days=report.get("completed_days", len(finished_days)),
                    total_days=report.get("total_days", 29),
                    forecast=forecast if completed else None)
        except Exception:
            with AGENT_LOCK:
                AGENT_JOBS[job_id].update(status="failed", summary=(
                    "Автономный проход прерван. Сохранённый прогресс не удалён; повторите запуск."))

    threading.Thread(target=work, daemon=True, name="forecast-campaign").start()
    return job_id


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, status: int, payload: dict) -> None:
        self.respond(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                     "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/api/agent/config":
            try:
                load_agent_environment(ROOT)
                available = agent_available()
            except OSError:
                available = False
            _station, _input_dir, campaign_reason = _campaign_settings()
            example = ROOT / "agent-examples" / "2026-02-01-report.json"
            self.json_response(200, {"available": available, "provider": "openai",
                "model": os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
                "reason": "" if available else "AI-агент не настроен на сервере; опубликованные прогнозы доступны.",
                "example_date": "2026-02-01" if example.is_file() else None,
                "campaign_available": not campaign_reason,
                "campaign_reason": campaign_reason})
            return
        if parsed.path == "/api/agent/example":
            try:
                report = json.loads((ROOT / "agent-examples" / "2026-02-01-report.json").read_text())
                report["recorded"] = True
                self.json_response(200, report)
            except (OSError, ValueError):
                self.json_response(404, {"error": "Запись реального запуска пока не сохранена."})
            return
        if parsed.path == "/api/agent/status":
            params = urllib.parse.parse_qs(parsed.query)
            ids = params.get("id", [])
            with AGENT_LOCK:
                job = copy.deepcopy(AGENT_JOBS.get(ids[0])) if len(ids) == 1 else None
            self.json_response(200 if job else 404, job or {"error": "Запуск не найден. Возможно, сервер был перезапущен."})
            return
        if parsed.path == "/":
            try:
                self.respond(200, HTML.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self.json_response(500, {"error": f"Не удалось открыть страницу: {exc}"})
            return
        if parsed.path == "/download/february-hourly.csv":
            try:
                self.respond(200, FEBRUARY_CSV.read_bytes(), "text/csv; charset=utf-8")
            except OSError as exc:
                self.json_response(500, {"error": f"Не удалось открыть CSV: {exc}"})
            return
        if parsed.path != "/api/forecast":
            self.json_response(404, {"error": "Маршрут не найден"})
            return
        params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        dates = params.get("date", [])
        refresh = params.get("refresh", ["0"])
        if len(dates) != 1 or not re.fullmatch(DATE_PATTERN, dates[0]) or len(refresh) != 1 or refresh[0] not in ("0", "1"):
            self.json_response(400, {"error": "Укажите дату 31 января — 28 февраля 2026"})
            return
        try:
            day = dt.date.fromisoformat(dates[0])
            if refresh[0] == "0":
                saved = json.loads((RESULTS / f"{day}-forecast.json").read_text())
                if saved["forecast_date"] != day.isoformat() or len(saved["forecast"]) != 48:
                    raise ValueError("сохранённый прогноз повреждён")
                saved["delivery_mode"] = "published_result"
                self.json_response(200, saved)
                return
            cutoff = min(day - dt.timedelta(days=1), dt.date(2026, 1, 31))
            model_path = MODELS / f"power-curve-{cutoff}.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            validate_model(model, cutoff)
            weather = {name: retrieve(day, name, CACHE, refresh=True) for name in LOCATIONS}
            output = combine(day, model, weather)
            RECALCULATED.mkdir(exist_ok=True)
            atomic_write(RECALCULATED / f"{day}-forecast.json",
                json.dumps(output, ensure_ascii=False, indent=2) + "\n")
            output["delivery_mode"] = "recalculated"
            self.json_response(200, output)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError,
                json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            self.json_response(502, {"error": f"Не удалось рассчитать прогноз: {exc}"})

    def do_POST(self) -> None:
        if self.path not in ("/api/agent", "/api/campaign"):
            self.json_response(404, {"error": "Маршрут не найден"})
            return
        origin = self.headers.get("Origin")
        if origin and urllib.parse.urlsplit(origin).netloc != self.headers.get("Host"):
            self.json_response(403, {"error": "Запускайте агента со страницы приложения."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.json_response(415, {"error": "Ожидается application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("Некорректный размер запроса.")
            body = json.loads(self.rfile.read(length))
            if self.path == "/api/campaign":
                if body != {}:
                    raise ValueError("Месячный запуск не принимает параметры клиента.")
                job_id = start_campaign_job()
            else:
                if (not isinstance(body, dict) or set(body) != {"date"}
                        or not isinstance(body["date"], str) or not re.fullmatch(DATE_PATTERN, body["date"])):
                    raise ValueError("Укажите дату 31 января — 28 февраля 2026.")
                job_id = start_agent_job(dt.date.fromisoformat(body["date"]))
            self.json_response(202, {"job_id": job_id})
        except (ValueError, OSError):
            self.json_response(400, {"error": ("Автономный проход недоступен: проверьте серверные настройки."
                                               if self.path == "/api/campaign" else
                                               "Проверьте дату и доступность AI-агента на сервере.")})
        except RuntimeError as exc:
            self.json_response(429, {"error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000,
                        help="local port (default: 8000; 0 selects a free port)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Не удалось запустить сервер: {exc}. "
                       "Выберите свободный порт: python3 web.py --port 8001\n")
    print(f"Wind forecast dashboard: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
