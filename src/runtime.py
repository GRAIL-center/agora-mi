from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScriptContext:
    script_name: str
    log_path: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(script_name: str, logs_dir: str | Path = "logs") -> ScriptContext:
    log_dir = ensure_dir(logs_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{script_name}_{ts}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logging.info("Logging initialized: %s", log_path)
    return ScriptContext(script_name=script_name, log_path=log_path)


def get_git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def metadata_snapshot(config: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": get_git_hash(),
    }
    if config is not None:
        meta["config"] = config
    return meta


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_with_metadata(
    *,
    output_path: str | Path,
    payload: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    out = {
        "metadata": metadata_snapshot(config),
        **payload,
    }
    save_json(output_path, out)
