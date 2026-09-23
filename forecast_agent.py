"""Bounded OpenAI tool-calling controller for the existing wind forecast pipeline."""
from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import os
import pathlib
import urllib.error
import urllib.request
from typing import Callable

from forecast_weather import LOCATIONS, retrieve
from power_model import train, validate_model
from run_forecast import combine
from station_profile import validate_station, station_digest
from storage import atomic_write

API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.4-mini"
MAX_TURNS = 8
MAX_CALLS = 12
FIRST_DAY = dt.date(2026, 1, 31)
LAST_DAY = dt.date(2026, 2, 28)
MODEL_LAST_DAY = dt.date(2026, 1, 31)


def agent_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _tool(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "name": name, "description": description,
            "strict": True, "parameters": _schema(properties)}


TOOLS = [
    _tool("inspect_inputs", "Inspect validated model and available weather caches.", {}),
    _tool("get_weather", "Load and validate archived weather for one turbine; refresh if needed.", {
        "turbine": {"type": "string", "enum": list(LOCATIONS)},
        "refresh": {"type": "boolean"},
    }),
    _tool("calculate_forecast", "Calculate the 48-hour forecast using both loaded weather inputs.", {}),
    _tool("inspect_forecast", "Inspect a slice of calculated forecast hours, indexed 0 to 47 inclusive.", {
        "start_hour": {"type": "integer"}, "end_hour": {"type": "integer"},
    }),
    _tool("compare_published", "Compare calculated forecast with published baseline if available.", {}),
    _tool("publish_forecast", "Save the calculated, current forecast in predictions.", {}),
]
STATION_TOOLS = [
    _tool("inspect_station", "Read the configured turbine coordinates, training cutoff and available local history.", {}),
    _tool("train_model", "Train and validate the station's own power curves from configured local CSV history, without future measurements.", {}),
]
TOOL_PROPERTIES = {item["name"]: item["parameters"]["properties"] for item in TOOLS + STATION_TOOLS}
SYSTEM = (
    "Ты управляешь прогнозом ветропарка через предоставленные инструменты. "
    "Дата и пути уже зафиксированы сервером. Сначала проверь входы, загрузи погоду "
    "обеих турбин, вычисли прогноз, при необходимости исследуй часы и сравни с "
    "опубликованным базовым прогнозом, затем опубликуй. После публикации дай "
    "краткий русский отчёт только по фактам из инструментов. Не придумывай числа, "
    "точность февраля, МВт или доказательства качества. Выход — нормализованная "
    "мощность 0..1; прогнозные метки времени UTC. UTC+5 для исходного CSV — лишь "
    "гипотеза, не подтверждённый факт. Точность на февральских "
    "измерениях неизвестна. Если инструмент вернул ошибку, попробуй восстановиться. "
    "Если анализ выявил часовой скачок >=0.2, изучи его окрестность через inspect_forecast, "
    "если остаётся достаточно вызовов. Сравнение с опубликованным снимком проверяет "
    "воспроизводимость, а не точность относительно измерений. Итог: 3–5 коротких "
    "предложений без Markdown и файловых путей; дата, основные числа, риск и ограничение. "
    "Не раскрывай скрытые рассуждения; используй лишь краткие действия и выводы."
)


class AgentFailure(Exception):
    """Safe, user-facing failure without upstream response bodies or secrets."""


def _response(payload: dict, key: str) -> dict:
    request = urllib.request.Request(
        API_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=35) as stream:
            result = json.load(stream)
    except urllib.error.HTTPError as exc:
        raise AgentFailure(f"OpenAI API вернул HTTP {exc.code}; проверьте ключ, модель и квоту.") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        raise AgentFailure("Не удалось связаться с OpenAI API; проверьте сеть и повторите запуск.") from None
    except (ValueError, TypeError):
        raise AgentFailure("OpenAI API вернул некорректный JSON; повторите запуск.") from None
    if not isinstance(result, dict) or not isinstance(result.get("output"), list):
        raise AgentFailure("OpenAI API вернул неполный ответ; повторите запуск.")
    if any(not isinstance(item, dict) or not isinstance(item.get("type"), str)
           for item in result["output"]):
        raise AgentFailure("OpenAI API вернул некорректные элементы ответа; повторите запуск.")
    return result


def _arguments(name: str, raw: object) -> dict:
    if name not in TOOL_PROPERTIES:
        raise AgentFailure("Неизвестный инструмент; выберите инструмент из списка.")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        raise AgentFailure("Аргументы инструмента должны быть JSON-объектом.") from None
    props = TOOL_PROPERTIES[name]
    if not isinstance(args, dict) or set(args) != set(props):
        raise AgentFailure("Аргументы инструмента не соответствуют разрешённой схеме.")
    for key, specification in props.items():
        value = args[key]
        if specification["type"] == "string" and value not in specification["enum"]:
            raise AgentFailure("Указана неизвестная турбина.")
        if specification["type"] == "boolean" and type(value) is not bool:
            raise AgentFailure("Параметр refresh должен быть логическим.")
        if specification["type"] == "integer" and type(value) is not int:
            raise AgentFailure("Номер часа должен быть целым числом.")
    return args


class _Cycle:
    def __init__(self, day: dt.date, root: pathlib.Path, model_path: pathlib.Path,
                 cache_dir: pathlib.Path, output_dir: pathlib.Path, *,
                 station: dict | None = None, input_dir: pathlib.Path | None = None,
                 training_cutoff: dt.date | None = None):
        self.day = day
        self.root = root
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.cutoff = min(day - dt.timedelta(days=1), training_cutoff or MODEL_LAST_DAY)
        self.station = validate_station(station) if station is not None else None
        self.input_dir = input_dir
        self.model: dict | None = None
        self.weather: dict[str, dict] = {}
        self.calculated: dict | None = None
        self.revision = 0
        self.calculated_revision = -1
        self.published_revision = -1
        self.path = output_dir / f"{day}-agent-forecast.json"

    def invoke(self, name: str, args: dict) -> dict:
        if name == "inspect_station":
            if self.station is None:
                raise AgentFailure("Профиль станции не задан для этого запуска.")
            return {"station": self.station, "training_cutoff": self.cutoff.isoformat(),
                    "model_available": self.model_path.is_file(),
                    "training_configured": self.input_dir is not None,
                    "model_family": "empirical wind power curve; trained separately for each turbine"}
        if name == "train_model":
            if self.station is None or self.input_dir is None:
                raise AgentFailure("Для обучения нужны профиль станции и настроенная локальная история CSV.")
            try:
                model = train(self.input_dir, self.cutoff)
                model["station"] = self.station
                model["station_sha256"] = station_digest(self.station)
                validate_model(model, self.cutoff)
                atomic_write(self.model_path, json.dumps(model, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            except (OSError, ValueError, KeyError, TypeError):
                raise AgentFailure("Обучение не выполнено: проверьте два CSV, их колонки и полные часы до даты отсечения.") from None
            self.model = model
            self.revision += 1
            self.calculated = None
            self.calculated_revision = self.published_revision = -1
            return {"trained_through_inclusive": self.cutoff.isoformat(),
                    "station_sha256": model["station_sha256"],
                    "data_quality": {name: item["data_quality"] for name, item in model["turbines"].items()},
                    "source_sha256": {name: item["source_sha256"] for name, item in model["turbines"].items()},
                    "calibration": "none; station-specific curves fitted to local sensor history"}
        if name == "inspect_inputs":
            model_path = self.model_path
            if not model_path.is_file():
                raise AgentFailure(f"Нет модели на дату отсечения {self.cutoff}; вызовите train_model, если обучение настроено.")
            try:
                model = json.loads(model_path.read_text(encoding="utf-8"))
                validate_model(model, self.cutoff)
                if self.station is not None and (model.get("station_sha256") != station_digest(self.station)
                        or model.get("station") != self.station):
                    raise ValueError("model belongs to another station")
                if self.input_dir is not None and any(
                        model["turbines"][name].get("source_sha256") != hashlib.sha256(
                            (self.input_dir / f"{name}.csv").read_bytes()).hexdigest()
                        for name in LOCATIONS):
                    raise ValueError("model belongs to different training history")
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                raise AgentFailure("Модель повреждена или не подходит станции/дате; вызовите train_model, если обучение настроено.") from None
            if self.model is not None and self.model != model:
                self.revision += 1
                self.calculated = None
                self.calculated_revision = -1
                self.published_revision = -1
            self.model = model
            cache = {}
            for turbine in LOCATIONS:
                path = self.cache_dir / f"{self.day}-{turbine}.json"
                if not path.is_file():
                    cache[turbine] = "absent"
                    continue
                try:
                    retrieve(self.day, turbine, self.cache_dir, **self._weather_options())
                    cache[turbine] = "valid"
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, urllib.error.URLError):
                    cache[turbine] = "invalid"
            return {"model_cutoff": model["trained_through_inclusive"], "weather_cache": cache,
                    "loaded_turbines": sorted(self.weather)}
        if name == "get_weather":
            turbine = args["turbine"]
            try:
                weather = retrieve(self.day, turbine, self.cache_dir,
                                   refresh=args["refresh"], **self._weather_options())
            except (OSError, ValueError, RuntimeError, KeyError, TypeError,
                    urllib.error.URLError, TimeoutError):
                raise AgentFailure(f"Погода {turbine} недоступна или кэш повреждён; попробуйте refresh=true либо проверьте сеть.") from None
            self.weather[turbine] = weather
            self.revision += 1
            self.calculated = None
            self.calculated_revision = -1
            self.published_revision = -1
            return {"turbine": turbine, "hours": len(weather["forecast"]),
                    "model_run_utc": weather["model_run_utc"], "refresh": args["refresh"]}
        if name == "calculate_forecast":
            if self.model is None:
                raise AgentFailure("Сначала выполните inspect_inputs для проверки модели.")
            if set(self.weather) != set(LOCATIONS):
                raise AgentFailure("Сначала загрузите погоду обеих турбин.")
            try:
                self.calculated = combine(self.day, self.model, self.weather)
                if self.station is not None:
                    self.calculated["station"] = self.station
            except (ValueError, KeyError, TypeError, ZeroDivisionError):
                raise AgentFailure("Не удалось вычислить прогноз: проверьте погодные входы и модель.") from None
            self.calculated_revision = self.revision
            self.published_revision = -1
            return {"hours": len(self.calculated["forecast"]),
                    "analysis": self.calculated["analysis"],
                    "largest_ramp_hour_index": next(i for i, row in enumerate(self.calculated["forecast"])
                        if row["time_utc"] == self.calculated["analysis"]["largest_hourly_ramp_ending_utc"]),
                    "training_cutoff_inclusive": self.calculated["training_cutoff_inclusive"]}
        if name == "inspect_forecast":
            self._require_current()
            start, end = args["start_hour"], args["end_hour"]
            if not 0 <= start <= end < 48 or end - start > 11:
                raise AgentFailure("Выберите от 1 до 12 часов в пределах индексов 0..47.")
            return {"start_hour": start, "end_hour": end,
                    "hours": self.calculated["forecast"][start:end + 1],
                    "analysis": self.calculated["analysis"]}
        if name == "compare_published":
            self._require_current()
            if self.station is not None:
                return {"baseline": "absent", "reason": "No published baseline is configured for this station campaign."}
            path = self.root / "results" / f"{self.day}-forecast.json"
            if not path.is_file():
                return {"baseline": "absent"}
            try:
                baseline = json.loads(path.read_text(encoding="utf-8"))
                ours = self.calculated["forecast"]
                theirs = baseline["forecast"]
                if len(theirs) != 48 or any(a["time_utc"] != b["time_utc"]
                                            for a, b in zip(ours, theirs)):
                    raise ValueError("incompatible baseline")
                key = "farm_equal_capacity_mean_normalized_power"
                differences = [round(a[key] - b[key], 6) for a, b in zip(ours, theirs)]
            except (OSError, ValueError, KeyError, TypeError):
                raise AgentFailure("Опубликованный базовый прогноз повреждён или несовместим.") from None
            return {"baseline": "available", "mean_delta": round(sum(differences) / 48, 6),
                    "max_absolute_hourly_delta": max(abs(value) for value in differences)}
        if name == "publish_forecast":
            self._require_current()
            try:
                atomic_write(self.path, json.dumps(self.calculated, ensure_ascii=False,
                                                   allow_nan=False, indent=2) + "\n")
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if saved != self.calculated or len(saved["forecast"]) != 48:
                    raise ValueError("write verification failed")
            except (OSError, ValueError, TypeError, KeyError):
                raise AgentFailure("Не удалось сохранить и проверить прогноз в predictions.") from None
            self.published_revision = self.revision
            return {"saved": True, "artifact": self.path.name,
                    "hours": 48, "mean_normalized_power": self.calculated["analysis"]["mean_normalized_power"]}
        raise AgentFailure("Неизвестный инструмент; выберите инструмент из списка.")

    def _weather_options(self) -> dict:
        return {"locations": self.station["locations"]} if self.station is not None else {}

    def _require_current(self) -> None:
        if self.calculated is None or self.calculated_revision != self.revision:
            raise AgentFailure("Прогноз отсутствует или устарел после изменения входов; запустите calculate_forecast.")

    def is_published(self) -> bool:
        return self.calculated is not None and self.published_revision == self.revision


def _message_text(output: list) -> str:
    parts = []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return " ".join(parts).strip()


def _event_result(name: str, result: dict) -> dict:
    """Keep the public journal useful without copying bulky hourly slices."""
    if name == "inspect_forecast":
        return {"start_hour": result["start_hour"], "end_hour": result["end_hour"],
                "hours_inspected": len(result["hours"])}
    if name == "calculate_forecast":
        return {"hours": result["hours"],
                "mean_normalized_power": result["analysis"]["mean_normalized_power"],
                "largest_hourly_ramp": result["analysis"]["largest_hourly_ramp"]}
    return result


def run_agent(day: dt.date, *, root: pathlib.Path,
              on_event: Callable[[dict], None] | None = None,
              model_path: pathlib.Path | None = None,
              cache_dir: pathlib.Path | None = None,
              output_dir: pathlib.Path | None = None,
              station: dict | None = None,
              input_dir: pathlib.Path | None = None,
              training_cutoff: dt.date | None = None) -> dict:
    """Run a fixed-date bounded forecast cycle; expected failures are returned as data."""
    model_name = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    report = {"status": "failed", "provider": "openai", "model": model_name,
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "resolved_models": [],
              "forecast_date": day.isoformat() if isinstance(day, dt.date) else str(day),
              "summary": "", "events": [], "forecast": None,
              "input_sha256": None,
              "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

    def emit(tool: str, status: str, detail: str, **extra: object) -> None:
        event = {"step": len(report["events"]) + 1, "tool": tool, "status": status,
                 "detail": detail[:220], **extra}
        report["events"].append(event)
        if on_event:
            try:
                on_event(dict(event))
            except Exception:
                pass  # Observability must not change forecast state.

    if not isinstance(day, dt.date) or isinstance(day, dt.datetime) or not FIRST_DAY <= day <= LAST_DAY:
        report["summary"] = "Дата должна быть в диапазоне 31.01–28.02.2026."
        return report
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        report["summary"] = "Не задан OPENAI_API_KEY; добавьте ключ в окружение и повторите запуск."
        return report
    if training_cutoff is not None and (not isinstance(training_cutoff, dt.date)
            or isinstance(training_cutoff, dt.datetime) or training_cutoff > MODEL_LAST_DAY):
        report["summary"] = "Дата обучения должна быть не позднее 31.01.2026."
        return report
    try:
        station = validate_station(station) if station is not None else None
    except (ValueError, TypeError, KeyError):
        report["summary"] = "Некорректный профиль станции."
        return report
    root = pathlib.Path(root)
    cutoff = min(day - dt.timedelta(days=1), training_cutoff or MODEL_LAST_DAY)
    workspace = (root / "predictions" / "stations" / f"{station['id']}-{station_digest(station)[:12]}"
                 if station is not None else root)
    cycle = _Cycle(day, root,
                   pathlib.Path(model_path) if model_path is not None else workspace / "models" / f"power-curve-{cutoff}.json",
                   pathlib.Path(cache_dir) if cache_dir is not None else workspace / "weather-cache",
                   pathlib.Path(output_dir) if output_dir is not None else (workspace if station is not None else root / "predictions"),
                   station=station, input_dir=pathlib.Path(input_dir) if input_dir is not None else None,
                   training_cutoff=training_cutoff)
    max_turns, max_calls = (12, 16) if station is not None else (MAX_TURNS, MAX_CALLS)
    instructions = SYSTEM
    selected_tools = TOOLS
    if station is not None:
        report["station"] = station
        selected_tools = TOOLS + STATION_TOOLS
        instructions += (" В этом запуске сначала вызови inspect_station. Координаты заданы оператором и проверены сервером. "
                         "Если модели нет, обучи её через train_model; если есть — проверь через inspect_inputs. "
                         "При несовместимой модели можно обучить новую, если локальная история настроена. "
                         "Исходные CSV не передаются тебе. Не утверждай, что новая модель откалибрована или её точность доказана.")
    conversation: list[dict] = [{"role": "user", "content": f"Подготовь и опубликуй прогноз на {day.isoformat()}."}]
    calls = 0
    try:
        for turn in range(max_turns):
            remaining = max_turns - turn
            payload = {"model": model_name,
                       "instructions": instructions + f" Осталось ответов модели: {remaining}. Необязательные проверки пропускай, если нужно успеть опубликовать.",
                       "input": conversation,
                       "tools": selected_tools, "parallel_tool_calls": False, "store": False,
                       "reasoning": {"effort": "low"},
                       "max_output_tokens": 1800}
            response = _response(payload, key)
            resolved = response.get("model")
            if isinstance(resolved, str) and resolved not in report["resolved_models"]:
                report["resolved_models"].append(resolved)
            if response.get("status", "completed") != "completed":
                raise AgentFailure("OpenAI API не завершил ответ; повторите запуск.")
            usage = response.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            for field in report["usage"]:
                value = usage.get(field, 0)
                if type(value) is int and value >= 0:
                    report["usage"][field] += value
            output = response["output"]
            if not isinstance(output, list) or any(not isinstance(item, dict)
                                                   or not isinstance(item.get("type"), str)
                                                   for item in output):
                raise AgentFailure("OpenAI API вернул некорректные элементы ответа; повторите запуск.")
            conversation.extend(output)
            tool_calls = [item for item in output if item.get("type") == "function_call"]
            if not tool_calls:
                if cycle.is_published():
                    summary = _message_text(output)
                    if not summary:
                        raise AgentFailure("Модель не сформировала итоговый отчёт; повторите запуск.")
                    # Verify the saved artifact once more before claiming success.
                    if json.loads(cycle.path.read_text(encoding="utf-8")) != cycle.calculated:
                        raise AgentFailure("Сохранённый прогноз изменился; повторите запуск.")
                    report["status"] = "completed"
                    report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    report["forecast"] = cycle.calculated
                    report["summary"] = summary[:800]
                    canonical = json.dumps({"model": cycle.model, "weather": cycle.weather},
                                           sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False, allow_nan=False)
                    report["input_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    atomic_write(cycle.output_dir / f"{day}-agent-report.json",
                                 json.dumps({key: value for key, value in report.items() if key != "forecast"},
                                            ensure_ascii=False, indent=2) + "\n")
                    return report
                raise AgentFailure("Модель завершила работу до публикации прогноза.")
            for call in tool_calls:
                calls += 1
                if calls > max_calls:
                    raise AgentFailure(f"Достигнут лимит {max_calls} вызовов инструментов; повторите запуск.")
                name = call.get("name", "unknown")
                try:
                    if not isinstance(call.get("call_id"), str):
                        raise AgentFailure("Вызов инструмента не содержит call_id.")
                    args = _arguments(name, call.get("arguments"))
                    result = cycle.invoke(name, args)
                    emit(name, "ok", "Инструмент выполнен.", arguments=args,
                         result=_event_result(name, result))
                    tool_result = {"ok": True, "result": result}
                except AgentFailure as exc:
                    emit(name, "error", str(exc))
                    tool_result = {"ok": False, "error": str(exc)}
                conversation.append({"type": "function_call_output", "call_id": call.get("call_id", ""),
                                     "output": json.dumps(tool_result, ensure_ascii=False, allow_nan=False)})
        raise AgentFailure(f"Достигнут лимит {max_turns} ответов модели; повторите запуск.")
    except AgentFailure as exc:
        report["summary"] = str(exc)
    except (OSError, ValueError, TypeError, KeyError):
        report["summary"] = "Цикл прогноза завершился ошибкой проверки данных или сохранения; повторите запуск."
    report["status"] = "failed"
    report["forecast"] = None
    report["input_sha256"] = None
    return report
