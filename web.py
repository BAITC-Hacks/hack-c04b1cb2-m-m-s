"""Local, dependency-free dashboard for the February wind-power forecast."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from forecast_weather import LOCATIONS, retrieve
from run_forecast import combine

ROOT = pathlib.Path(__file__).resolve().parent
HTML = ROOT / "static" / "index.html"
FEBRUARY_CSV = ROOT / "results" / "february-hourly.csv"
MODELS = ROOT / "models"
CACHE = ROOT / "weather-cache"
RESULTS = ROOT / "results"
RECALCULATED = ROOT / "predictions"


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
        if len(dates) != 1 or not re.fullmatch(
            r"2026-(?:01-31|02-(?:0[1-9]|1[0-9]|2[0-8]))", dates[0]
        ) or len(refresh) != 1 or refresh[0] not in ("0", "1"):
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
            if dt.date.fromisoformat(model["trained_through_inclusive"]) > cutoff:
                raise ValueError("модель содержит будущие измерения")
            weather = {name: retrieve(day, name, CACHE, refresh=True) for name in LOCATIONS}
            output = combine(day, model, weather)
            RECALCULATED.mkdir(exist_ok=True)
            (RECALCULATED / f"{day}-forecast.json").write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n")
            output["delivery_mode"] = "recalculated"
            self.json_response(200, output)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError,
                json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            self.json_response(502, {"error": f"Не удалось рассчитать прогноз: {exc}"})


def main() -> None:
    address = ("127.0.0.1", 8000)
    server = ThreadingHTTPServer(address, Handler)
    print(f"Wind forecast dashboard: http://{address[0]}:{address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
