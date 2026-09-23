"""Load only the optional agent's settings; never execute a dotenv file."""
from __future__ import annotations

import os
import pathlib


def load_agent_environment(root: pathlib.Path) -> None:
    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.strip().partition("=")
        name = name.strip()
        if not separator or name not in {"OPENAI_API_KEY", "OPENAI_MODEL",
                                         "WIND_STATION_FILE", "WIND_TRAINING_DIR"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value and not os.environ.get(name):
            os.environ[name] = value
