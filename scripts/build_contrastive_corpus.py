from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from _common import read_config, save_with_metadata, setup_logging
from data.io import load_agora_records, write_jsonl
from data.labeling import LabelMap, assign_label
from data.split import assert_no_overlap, split_doc_ids


def _text_length_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"count": 0, "mean_chars": 0.0, "min_chars": 0.0, "max_chars": 0.0}
    lengths = [len(str(r["text"])) for r in rows]
    return {
        "count": len(rows),
        "mean_chars": sum(lengths) / len(lengths),
        "min_chars": float(min(lengths)),
        "max_chars": float(max(lengths)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/label_map.yaml")
    parser.add_argument("--input_dir", default="data/raw/agora")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging("build_contrastive_corpus")
    cfg = read_config(args.config)

    label_map = LabelMap(
        dsafe_tags=list(cfg.get("dsafe_tags", [])),
        dinnov_tags=list(cfg.get("dinnov_tags", [])),
        overlap_policy=str(cfg.get("overlap_policy", "drop")),
    )
    min_chars = int(cfg.get("min_text_chars", 0))
    max_chars = int(cfg.get("max_text_chars", 10_000_000))
    dedupe = bool(cfg.get("dedupe", True))
    train_ratio = float(cfg.get("train_ratio", 0.7))
    dev_ratio = float(cfg.get("dev_ratio", 0.15))
    test_ratio = float(cfg.get("test_ratio", 0.15))
    max_per_class = cfg.get("max_per_class", None)
    max_per_class = int(max_per_class) if max_per_class is not None else None

    records = load_agora_records(args.input_dir)
    logging.info("Loaded AGORA records: %d", len(records))

    labeled_rows: list[dict] = []
    dropped_overlap = 0
    dropped_length = 0
    for r in records:
        text = str(r["text"])
        if len(text) < min_chars or len(text) > max_chars:
            dropped_length += 1
            continue
        label = assign_label(r.get("tags", []), label_map)
        if label is None:
            continue
        if label == "both":
            # keep both copies with explicit labels
            for lb in ("safe", "innov"):
                row = dict(r)
                row["label"] = lb
                row["id"] = f"{r['id']}_{lb}"
                labeled_rows.append(row)
            continue
        if label_map.overlap_policy == "drop":
            is_safe = label == "safe"
            is_innov = label == "innov"
            if is_safe and is_innov:
                dropped_overlap += 1
                continue
        row = dict(r)
        row["label"] = label
        labeled_rows.append(row)

    if dedupe:
        seen = set()
        deduped: list[dict] = []
        for r in labeled_rows:
            key = (r["label"], " ".join(str(r["text"]).split()).lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        labeled_rows = deduped

    by_label_doc: dict[str, dict[str, list[dict]]] = {
        "safe": defaultdict(list),
        "innov": defaultdict(list),
    }
    for r in labeled_rows:
        lb = r["label"]
        by_label_doc[lb][str(r["doc_id"])].append(r)

    splits: dict[str, dict[str, set[str]]] = {}
    for lb in ("safe", "innov"):
        doc_ids = list(by_label_doc[lb].keys())
        ds = split_doc_ids(
            doc_ids,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )
        assert_no_overlap(ds)
        splits[lb] = {"train": set(ds.train), "dev": set(ds.dev), "test": set(ds.test)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_rows: dict[str, list[dict]] = {
        "dsafe_train": [],
        "dsafe_dev": [],
        "dsafe_test": [],
        "dinnov_train": [],
        "dinnov_dev": [],
        "dinnov_test": [],
    }
    for lb in ("safe", "innov"):
        prefix = "dsafe" if lb == "safe" else "dinnov"
        for split in ("train", "dev", "test"):
            key = f"{prefix}_{split}"
            rows: list[dict] = []
            for doc_id in sorted(splits[lb][split]):
                rows.extend(by_label_doc[lb][doc_id])
            if max_per_class is not None:
                rows = rows[:max_per_class]
            out_rows[key] = rows
            write_jsonl(out_dir / f"{key}.jsonl", rows)

    summary = {
        "input_records": len(records),
        "kept_records": len(labeled_rows),
        "dropped_length": dropped_length,
        "dropped_overlap": dropped_overlap,
        "overlap_policy": label_map.overlap_policy,
        "splits": {
            k: _text_length_stats(v) for k, v in out_rows.items()
        },
    }
    for k, st in summary["splits"].items():
        logging.info(
            "%s | n=%d mean_chars=%.1f min=%.0f max=%.0f",
            k,
            st["count"],
            st["mean_chars"],
            st["min_chars"],
            st["max_chars"],
        )

    save_with_metadata(
        output_path=Path("logs") / "build_contrastive_summary.json",
        payload={"summary": summary},
        config={"label_map": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
