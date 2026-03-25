from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from _common import ensure_dir, read_config, save_json, setup_logging


def infer_fields(columns: list[str]) -> dict[str, str | None]:
    lower = {c.lower(): c for c in columns}

    def pick(cands: list[str]) -> str | None:
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    return {
        "doc_id_field": pick(["Document ID", "AGORA ID", "doc_id"]),
        "text_field": pick(["Text", "text", "full_text"]),
        "tag_field": pick(["Tags", "tags"]),
    }


def inspect_csv(path: Path) -> dict:
    df = pd.read_csv(path, nrows=200)
    fields = infer_fields(list(df.columns))
    info = {
        "file": path.name,
        "columns": list(df.columns),
        "shape_head": [int(df.shape[0]), int(df.shape[1])],
        "inferred": fields,
    }

    tag_col = fields["tag_field"]
    if tag_col and tag_col in df.columns:
        sample = df[tag_col].dropna().astype(str).head(8).tolist()
        info["tag_samples"] = sample
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw/agora")
    parser.add_argument("--out", default="logs/agora_schema_summary.json")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    setup_logging("inspect_agora_schema")
    if args.config:
        cfg = read_config(args.config)
        input_dir = cfg.get("input_dir", args.input_dir)
    else:
        input_dir = args.input_dir

    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base}")

    files = sorted([p for p in base.iterdir() if p.is_file()])
    logging.info("Found %d files in %s", len(files), base)
    for p in files:
        logging.info("file: %s", p.name)

    csv_infos = []
    for p in files:
        if p.suffix.lower() == ".csv":
            info = inspect_csv(p)
            csv_infos.append(info)
            logging.info("[%s] columns=%d", p.name, len(info["columns"]))
            if info.get("tag_samples"):
                logging.info("[%s] tag samples: %s", p.name, info["tag_samples"][:3])

    summary = {
        "input_dir": str(base),
        "file_list": [p.name for p in files],
        "csv_summaries": csv_infos,
    }
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    save_json(out_path, summary)
    logging.info("Wrote schema summary: %s", out_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
