from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path, 'r') as fp:
        return json.load(fp)

def save_config(data, path) -> None:
    with open(path, 'w') as fp:
        return json.dump(data, fp)
