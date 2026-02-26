from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime import load_yaml, setup_logging, ensure_dir, save_with_metadata, save_json  # noqa: E402


def read_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    return cfg
