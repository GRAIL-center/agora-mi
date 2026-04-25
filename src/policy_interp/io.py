"""IO helpers for parquet, jsonl, and safetensors artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None
    load_file = None
    save_file = None

from policy_interp.utils import ensure_dir


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    frame.to_parquet(target, index=False)
    return target


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_safetensors(tensors: dict[str, torch.Tensor], path: str | Path) -> Path:
    if save_file is None:
        raise ImportError("safetensors and torch are required to write safetensors artifacts.")
    target = Path(path)
    ensure_dir(target.parent)
    save_file(tensors, str(target))
    return target


def read_safetensors(path: str | Path) -> dict[str, Any]:
    if load_file is None:
        raise ImportError("safetensors and torch are required to read safetensors artifacts.")
    return load_file(str(path))
