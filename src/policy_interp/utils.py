"""General utilities."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

BOOLEAN_TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in BOOLEAN_TRUE_VALUES


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash_frame(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=0).sort_index(axis=1)
    payload = ordered.to_json(orient="split", index=True, date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_hash_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_year(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    if not match:
        return None
    return int(match.group(0))


def minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if abs(high - low) < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def chunked(iterable: Iterable[Any], chunk_size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
