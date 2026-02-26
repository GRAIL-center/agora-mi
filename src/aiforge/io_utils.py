from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_parent_dir(path: Path | str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
