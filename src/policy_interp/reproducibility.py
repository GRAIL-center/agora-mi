"""Artifact hashing for deterministic reproducibility checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from policy_interp.io import read_jsonl, read_parquet, write_jsonl
from policy_interp.utils import stable_hash_frame, stable_hash_json


def build_artifact_hash_report(run_root: str | Path) -> Path:
    root = Path(run_root)
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".parquet":
            digest = stable_hash_frame(read_parquet(path))
        elif path.suffix == ".jsonl":
            digest = stable_hash_json(read_jsonl(path))
        elif path.suffix in {".md", ".html", ".yaml", ".yml", ".txt"}:
            digest = stable_hash_json({"text": path.read_text(encoding="utf-8")})
        else:
            continue
        records.append({"path": str(path.resolve()), "sha256": digest})
    report_path = root / "artifact_hashes.jsonl"
    write_jsonl(records, report_path)
    return report_path
