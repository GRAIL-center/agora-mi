from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiforge.io_utils import read_jsonl, write_jsonl


@dataclass(frozen=True)
class SplitResult:
    train: list[str]
    dev: list[str]
    test: list[str]


def parse_tag_list(raw: str) -> list[str]:
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def _normalize_tags(value: Any) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, list):
        return {str(x).strip().lower() for x in value if str(x).strip()}

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return {str(x).strip().lower() for x in parsed if str(x).strip()}
            except json.JSONDecodeError:
                pass
        pieces = re.split(r"[|,;/]", text)
        return {p.strip().lower() for p in pieces if p.strip()}

    return {str(value).strip().lower()}


def _split_doc_ids(
    doc_ids: list[str],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> SplitResult:
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio}+{dev_ratio}+{test_ratio}={total_ratio}"
        )

    shuffled = doc_ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = shuffled[:n_train]
    dev = shuffled[n_train : n_train + n_dev]
    test = shuffled[n_train + n_dev :]
    return SplitResult(train=train, dev=dev, test=test)


def build_contrastive_corpora(
    *,
    input_jsonl: str,
    output_dir: str,
    safe_tags: list[str],
    innov_tags: list[str],
    id_field: str,
    text_field: str,
    tag_field: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    max_per_class: int | None,
    seed: int,
) -> dict[str, int]:
    rows = read_jsonl(input_jsonl)

    safe_set = {t.lower() for t in safe_tags}
    innov_set = {t.lower() for t in innov_tags}
    if not safe_set:
        raise ValueError("safe_tags is empty")
    if not innov_set:
        raise ValueError("innov_tags is empty")

    doc_labels: dict[str, set[str]] = defaultdict(set)
    rows_by_doc_and_label: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped_missing_text = 0

    for idx, row in enumerate(rows):
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            skipped_missing_text += 1
            continue

        doc_id = str(row.get(id_field) or f"row_{idx}")
        tags = _normalize_tags(row.get(tag_field))
        is_safe = bool(tags & safe_set)
        is_innov = bool(tags & innov_set)

        if is_safe and is_innov:
            # Keep contrastive sets clean by dropping mixed examples.
            continue
        if not is_safe and not is_innov:
            continue

        label = "safe" if is_safe else "innov"
        doc_labels[doc_id].add(label)
        payload = dict(row)
        payload["label"] = label
        payload["doc_id"] = doc_id
        rows_by_doc_and_label[(doc_id, label)].append(payload)

    valid_docs_by_label: dict[str, list[str]] = {"safe": [], "innov": []}
    for doc_id, labels in doc_labels.items():
        if len(labels) != 1:
            continue
        label = next(iter(labels))
        valid_docs_by_label[label].append(doc_id)

    rng = random.Random(seed)
    doc_splits = {
        label: _split_doc_ids(
            doc_ids=doc_ids,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )
        for label, doc_ids in valid_docs_by_label.items()
    }

    split_to_doc_ids: dict[str, dict[str, set[str]]] = {
        "train": {
            "safe": set(doc_splits["safe"].train),
            "innov": set(doc_splits["innov"].train),
        },
        "dev": {
            "safe": set(doc_splits["safe"].dev),
            "innov": set(doc_splits["innov"].dev),
        },
        "test": {
            "safe": set(doc_splits["safe"].test),
            "innov": set(doc_splits["innov"].test),
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {
        "input_rows": len(rows),
        "skipped_missing_text": skipped_missing_text,
        "safe_docs": len(valid_docs_by_label["safe"]),
        "innov_docs": len(valid_docs_by_label["innov"]),
    }

    for split in ("train", "dev", "test"):
        for label in ("safe", "innov"):
            selected_doc_ids = split_to_doc_ids[split][label]
            selected_rows: list[dict] = []
            for doc_id in sorted(selected_doc_ids):
                selected_rows.extend(rows_by_doc_and_label[(doc_id, label)])

            if max_per_class is not None:
                selected_rows = selected_rows[:max_per_class]

            out_path = out_dir / f"D{label}_{split}.jsonl"
            write_jsonl(out_path, selected_rows)
            counters[f"{label}_{split}_rows"] = len(selected_rows)

    return counters
