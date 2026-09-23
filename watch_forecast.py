"""Recheck one historical forecast's archived inputs at a fixed interval.

Only a changed, fully validated set of inputs replaces the published forecast.
The watcher state lives beside the output, outside the forecast JSON schema.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import time
import urllib.error

from forecast_weather import LOCATIONS, retrieve, run_for, validate_result
from run_forecast import LATEST_TRAINING_DATE, combine
from power_model import validate_model
from storage import atomic_write


def canonical_digest(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def run_cycle(day: dt.date, model_path: pathlib.Path, cache_dir: pathlib.Path,
              output_dir: pathlib.Path, *, agent_mode: bool = False) -> bool:
    """Return True when a validated forecast was saved, False when unchanged."""
    cutoff = min(day - dt.timedelta(days=1), LATEST_TRAINING_DATE)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    validate_model(model, cutoff)

    decision, run = run_for(day)
    weather = {}
    for turbine in LOCATIONS:
        data = retrieve(day, turbine, cache_dir, refresh=True)
        # Repeat validation at the cycle boundary, before comparing or publishing.
        validate_result(data, day, turbine, data["request_url"], decision, run)
        weather[turbine] = data

    input_digest = canonical_digest({"model": model, "weather": weather})
    output_path = output_dir / f"{day.isoformat()}-forecast.json"
    state_path = output_dir / f"{day.isoformat()}-watch-state.json"
    if output_path.exists() and state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (state.get("input_sha256") == input_digest
                    and state.get("controller", "deterministic") == ("openai" if agent_mode else "deterministic")
                    and state.get("output_sha256") == hashlib.sha256(output_path.read_bytes()).hexdigest()):
                return False
        except (OSError, ValueError, AttributeError):
            pass  # Missing or damaged state requires a complete recalculation.

    if agent_mode:
        from forecast_agent import run_agent
        report = run_agent(day, root=pathlib.Path(__file__).resolve().parent,
                           model_path=model_path, cache_dir=cache_dir, output_dir=output_dir)
        if report["status"] != "completed":
            raise RuntimeError(report["summary"])
        output = report["forecast"]
        input_digest = report["input_sha256"]
    else:
        output = combine(day, model, weather)
    serialized = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    atomic_write(output_path, serialized)
    atomic_write(state_path, json.dumps({
        "input_sha256": input_digest,
        "output_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "controller": "openai" if agent_mode else "deterministic",
    }, indent=2) + "\n")
    return True


def watch(day: dt.date, model_path: pathlib.Path, cache_dir: pathlib.Path,
          output_dir: pathlib.Path, interval: float, max_cycles: int | None = None,
          *, agent_mode: bool = False) -> int:
    last_failed = False
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        try:
            changed = run_cycle(day, model_path, cache_dir, output_dir, agent_mode=agent_mode)
            print(f"{day} cycle {cycle}: {'saved updated forecast' if changed else 'inputs unchanged'}",
                  flush=True)
            last_failed = False
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError,
                urllib.error.URLError) as exc:
            previous = output_dir / f"{day.isoformat()}-forecast.json"
            result_status = "keeping last good result" if previous.exists() else "no forecast published"
            print(f"{day} cycle {cycle}: forecast update failed: {exc}; "
                  f"{result_status}", file=sys.stderr, flush=True)
            last_failed = True
        if max_cycles is None or cycle < max_cycles:
            time.sleep(interval)
    return int(last_failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", type=dt.date.fromisoformat, help="historical decision date, YYYY-MM-DD")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=pathlib.Path, help="pretrained power-curve JSON")
    source.add_argument("--model-dir", type=pathlib.Path,
                        help="directory of power-curve-YYYY-MM-DD.json artifacts")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("weather-cache"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("predictions"))
    parser.add_argument("--interval", type=float, default=300, help="seconds between checks (default: 300)")
    parser.add_argument("--max-cycles", type=int, help="stop after this many checks (for smoke runs)")
    parser.add_argument("--agent", action="store_true",
                        help="run the OpenAI tool-calling agent only when validated inputs change")
    args = parser.parse_args()
    if (not math.isfinite(args.interval) or args.interval <= 0
            or args.max_cycles is not None and args.max_cycles < 1):
        parser.error("--interval must be finite and positive and --max-cycles must be at least 1")
    cutoff = min(args.date - dt.timedelta(days=1), LATEST_TRAINING_DATE)
    model_path = (args.model_dir / f"power-curve-{cutoff}.json"
                  if args.model_dir else args.model)
    if args.agent:
        from agent_settings import load_agent_environment
        from forecast_agent import agent_available, FIRST_DAY, LAST_DAY
        load_agent_environment(pathlib.Path(__file__).resolve().parent)
        if not FIRST_DAY <= args.date <= LAST_DAY:
            parser.error("--agent supports 2026-01-31 through 2026-02-28")
        if not agent_available():
            parser.error("--agent requires server-side OPENAI_API_KEY in environment or .env")
    try:
        return watch(args.date, model_path, args.cache_dir, args.output_dir,
                     args.interval, args.max_cycles, agent_mode=args.agent)
    except KeyboardInterrupt:
        print("watch stopped", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
