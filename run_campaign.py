"""Run a resumable historical agent campaign for a configured two-turbine station.

By default, use the competition station and its prepared models without CSVs.
For local training, supply a station profile and compatible CSVs via --input-dir.
The agent chooses tools, including training when configured. The scheduler
advances historical days; it never substitutes today's weather or uploads raw
training rows to OpenAI.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Callable

from agent_settings import load_agent_environment
from forecast_agent import FIRST_DAY, LAST_DAY, MODEL_LAST_DAY, agent_available, run_agent
from forecast_weather import retrieve
from power_model import validate_model
from station_profile import DEFAULT_STATION, load_station, station_digest, validate_station
from storage import atomic_write
from watch_forecast import canonical_digest


def source_hashes(input_dir: pathlib.Path) -> dict:
    return {name: hashlib.sha256((input_dir / f"{name}.csv").read_bytes()).hexdigest()
            for name in ("turbine-1", "turbine-2")}


def file_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prepared_models(root: pathlib.Path, station: dict) -> dict[str, dict]:
    """Load prepared models for the competition turbines, preserving profile labels."""
    root = pathlib.Path(root)
    station = validate_station(station)
    # The bundled example names this same farm differently. Labels identify the
    # campaign workspace; the turbine coordinates determine model compatibility.
    if station["locations"] != validate_station(DEFAULT_STATION)["locations"]:
        raise ValueError("Готовые модели доступны только для стандартной станции; для другой станции нужны CSV.")
    models = {}
    for cutoff in (dt.date(2026, 1, 30), dt.date(2026, 1, 31)):
        name = f"power-curve-{cutoff}.json"
        model = json.loads((root / "models" / name).read_text(encoding="utf-8"))
        validate_model(model, cutoff)
        if model.get("trained_through_inclusive") != str(cutoff):
            raise ValueError(f"Готовая модель {name} имеет неверную дату отсечения.")
        if (("station" in model and model["station"] != station)
                or ("station_sha256" in model
                    and model["station_sha256"] != station_digest(station))):
            raise ValueError(f"Готовая модель {name} размечена для другой станции.")
        model = dict(model)
        model["station"] = station
        model["station_sha256"] = station_digest(station)
        models[name] = model
    return models


@contextlib.contextmanager
def campaign_lock(directory: pathlib.Path):
    # OS-owned locks disappear after process exit, including a crash/reboot.
    try:
        import fcntl
    except ImportError:
        raise RuntimeError("Автономный проход поддерживается на Linux и macOS.") from None
    with (directory / ".campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Этот автономный проход уже выполняется другим процессом.") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def saved_day(day: dt.date, entry: dict, directory: pathlib.Path, station: dict,
              hashes: dict, *, prepared: bool = False) -> dict | None:
    """Resume only if artifacts and the actually used inputs still match."""
    try:
        cutoff = min(day - dt.timedelta(days=1), MODEL_LAST_DAY)
        model_path = directory / "models" / f"power-curve-{cutoff}.json"
        forecast_path = directory / f"{day}-agent-forecast.json"
        report_path = directory / f"{day}-agent-report.json"
        if any(entry.get(key) != file_hash(path) for key, path in (
                ("model_sha256", model_path), ("forecast_sha256", forecast_path),
                ("report_sha256", report_path))):
            return None
        model = json.loads(model_path.read_text())
        validate_model(model, cutoff)
        if (model.get("station") != station or model.get("station_sha256") != station_digest(station)
                or (not prepared and any(model["turbines"][name].get("source_sha256") != digest
                                         for name, digest in hashes.items()))):
            return None
        cache_dir = directory / "weather-cache"
        # Resume is an offline integrity check. Missing cache triggers a new run.
        if any(not (cache_dir / f"{day}-{name}.json").is_file() for name in station["locations"]):
            return None
        weather = {name: retrieve(day, name, cache_dir, locations=station["locations"])
                   for name in station["locations"]}
        digest = canonical_digest({"model": model, "weather": weather})
        report = json.loads(report_path.read_text())
        forecast = json.loads(forecast_path.read_text())
        if (entry.get("input_sha256") != digest or report.get("input_sha256") != digest
                or report.get("status") != "completed" or report.get("forecast_date") != str(day)
                or forecast.get("forecast_date") != str(day) or forecast.get("station") != station
                or len(forecast.get("forecast", [])) != 48):
            return None
        return forecast
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        return None


def run_campaign(*, root: pathlib.Path, station: dict, input_dir: pathlib.Path | None = None,
                 first_day: dt.date = FIRST_DAY, days: int = 29,
                 output_dir: pathlib.Path | None = None,
                 on_event: Callable[[dict], None] | None = None, attempts: int = 2) -> dict:
    report = {"status": "failed", "summary": "", "events": [], "station": None,
              "completed_days": 0, "total_days": days, "forecast": None,
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

    def emit(tool: str, status: str, detail: str, day: dt.date, **extra):
        event = {"step": len(report["events"]) + 1, "tool": tool, "status": status,
                 "detail": detail, "forecast_date": str(day), **extra}
        report["events"].append(event)
        if on_event:
            try:
                on_event(event.copy())
            except Exception:
                pass  # A disconnected observer must not abort the saved work.

    try:
        station = validate_station(station)
        if (type(days) is not int or not 1 <= days <= 29 or type(attempts) is not int
                or not 1 <= attempts <= 2 or type(first_day) is not dt.date
                or not FIRST_DAY <= first_day <= first_day + dt.timedelta(days=days - 1) <= LAST_DAY):
            raise ValueError("Допустим последовательный период 31.01–28.02.2026; до двух попыток на день.")
        report["station"] = station
        root = pathlib.Path(root)
        prepared = input_dir is None
        if prepared:
            prepared_models = load_prepared_models(root, station)
            hashes = {name: file_hash(root / "models" / name) for name in prepared_models}
        else:
            input_dir = pathlib.Path(input_dir)
            prepared_models = {}
            hashes = source_hashes(input_dir)
        profile_hash = station_digest(station)
        dataset_hash = canonical_digest({"prepared_models": hashes} if prepared else hashes)
        dataset_suffix = f"prepared-{dataset_hash[:10]}" if prepared else dataset_hash[:10]
        directory = pathlib.Path(output_dir) if output_dir is not None else (
            root / "predictions" / "campaigns" / f"{station['id']}-{profile_hash[:10]}-{dataset_suffix}")
        protected = [(root / name).resolve() for name in ("models", "results")]
        if (directory.resolve() == root.resolve()
                or any(candidate.resolve().is_relative_to(path)
                       for path in protected for candidate in (directory, directory / "models"))):
            raise ValueError("Папка прохода не должна совпадать с корнем проекта или опубликованными models/results.")
        directory.mkdir(parents=True, exist_ok=True)
        with campaign_lock(directory):
            state_path = directory / "campaign-state.json"
            mode = "prepared-models-v1" if prepared else "csv-training-v1"
            state = {"version": 1, "mode": mode, "station_sha256": profile_hash,
                     "source_sha256": hashes, "days": {}}
            if state_path.is_file():
                prior = json.loads(state_path.read_text())
                if (prior.get("station_sha256") != profile_hash or prior.get("source_sha256") != hashes
                        or prior.get("mode", "csv-training-v1" if not prepared else None) != mode
                        or prior.get("version") != 1 or not isinstance(prior.get("days"), dict)):
                    raise ValueError("Папка прохода относится к другой станции или истории; выберите новую папку результата.")
                state = prior
            for offset in range(days):
                day = first_day + dt.timedelta(days=offset)
                if (source_hashes(input_dir) if not prepared else
                        {name: file_hash(root / "models" / name) for name in prepared_models}) != hashes:
                    raise ValueError("Исходные CSV или готовые модели изменились во время прохода; остановка для согласованного повторного запуска.")
                previous = saved_day(day, state["days"].get(str(day), {}), directory, station, hashes,
                                     prepared=prepared)
                if previous is not None:
                    report["forecast"] = previous
                    report["completed_days"] += 1
                    emit("campaign_day", "skipped", "Проверенный результат уже сохранён; OpenAI не вызывается.", day,
                         result={"completed_days": report["completed_days"], "total_days": days})
                    continue
                if not agent_available():
                    raise RuntimeError("Для нового дневного расчёта требуется серверный OPENAI_API_KEY.")
                cutoff = min(day - dt.timedelta(days=1), MODEL_LAST_DAY)
                model_path = directory / "models" / f"power-curve-{cutoff}.json"
                if prepared:
                    model_name = model_path.name
                    if model_name not in prepared_models:
                        raise ValueError(f"Нет готовой модели на дату отсечения {cutoff}.")
                    atomic_write(model_path, json.dumps(prepared_models[model_name], ensure_ascii=False,
                                                        allow_nan=False, indent=2) + "\n")
                daily = None
                for attempt in range(1, attempts + 1):
                    emit("campaign_day" if attempt == 1 else "campaign_retry", "start",
                         f"День {offset + 1}/{days}, попытка {attempt}/{attempts}.", day)

                    def relay(event):
                        event = dict(event)
                        event.pop("step", None)
                        emit(day=day, **event)

                    daily = run_agent(day, root=root, station=station, input_dir=None if prepared else input_dir,
                                      model_path=model_path, cache_dir=directory / "weather-cache",
                                      output_dir=directory, on_event=relay)
                    for field in report["usage"]:
                        report["usage"][field] += daily.get("usage", {}).get(field, 0)
                    if daily["status"] == "completed":
                        break
                    emit("campaign_day", "error", daily["summary"], day)
                if daily is None or daily["status"] != "completed":
                    raise RuntimeError(f"Остановка на {day}: {daily['summary'] if daily else 'нет результата'} Повторный запуск продолжит с этого дня.")
                if (source_hashes(input_dir) if not prepared else
                        {name: file_hash(root / "models" / name) for name in prepared_models}) != hashes:
                    raise ValueError("Исходные CSV или готовые модели изменились во время расчёта; результат не отмечен завершённым.")
                entry = {"input_sha256": daily["input_sha256"], "model_sha256": file_hash(model_path),
                         "forecast_sha256": file_hash(directory / f"{day}-agent-forecast.json"),
                         "report_sha256": file_hash(directory / f"{day}-agent-report.json")}
                verified = saved_day(day, entry, directory, station, hashes, prepared=prepared)
                if verified is None or verified != daily["forecast"]:
                    raise RuntimeError("Не удалось подтвердить сохранённый дневной результат; прогресс не обновлён.")
                state["days"][str(day)] = entry
                atomic_write(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
                report["forecast"] = verified
                report["completed_days"] += 1
                emit("campaign_day", "ok", "48 часов сохранены; переход к следующей исторической дате.", day,
                     result={"completed_days": report["completed_days"], "total_days": days})
            report["status"] = "completed"
            report["summary"] = f"Автономный проход завершён: {days}/{days} дней. Модели и прогнозы сохранены отдельно для этой станции."
            report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_write(directory / "campaign-report.json",
                         json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (OSError, KeyError, TypeError, AttributeError):
        report["status"] = "failed"
        report["summary"] = "Не удалось прочитать историю или сохранить результат прохода; проверьте серверную настройку файлов."
    except (ValueError, RuntimeError) as exc:
        report["status"] = "failed"
        report["summary"] = str(exc)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", type=pathlib.Path,
                        help="Station JSON profile (default: stations/example.json in the project)")
    parser.add_argument("--input-dir", type=pathlib.Path,
                        help="Training CSV directory; omit to use the competition station's prepared models")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=FIRST_DAY)
    parser.add_argument("--days", type=int, default=29)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    try:
        load_agent_environment(root)
        station = load_station(args.station or root / "stations" / "example.json")
        def event(item):
            print(f"{item['forecast_date']} {item['tool']} [{item['status']}]: {item['detail']}", flush=True)
        result = run_campaign(root=root, station=station, input_dir=args.input_dir, first_day=args.start,
                              days=args.days, output_dir=args.output_dir, attempts=args.attempts, on_event=event)
        print(result["summary"], flush=True)
        return 0 if result["status"] == "completed" else 1
    except (OSError, ValueError) as exc:
        print(f"Не удалось настроить проход: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
