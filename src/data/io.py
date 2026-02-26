from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_tag(tag: str) -> str:
    s = str(tag).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_tags_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [x.strip() for x in text.split(";") if x.strip()]
    return [str(value).strip()]


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_agora_tables(input_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(input_dir)
    documents_path = base / "documents.csv"
    segments_path = base / "segments.csv"
    if not documents_path.exists():
        raise FileNotFoundError(f"Missing documents.csv: {documents_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing segments.csv: {segments_path}")

    documents = pd.read_csv(documents_path)
    segments = pd.read_csv(segments_path)
    return documents, segments


def load_agora_records(input_dir: str | Path) -> list[dict[str, Any]]:
    documents, segments = load_agora_tables(input_dir)

    doc_id_col = _pick_column(list(documents.columns), ["AGORA ID", "Document ID", "doc_id"])
    seg_doc_id_col = _pick_column(list(segments.columns), ["Document ID", "AGORA ID", "doc_id"])
    seg_text_col = _pick_column(list(segments.columns), ["Text", "text", "full_text"])
    seg_tags_col = _pick_column(list(segments.columns), ["Tags", "tags"])
    seg_pos_col = _pick_column(list(segments.columns), ["Segment position", "position", "segment_index"])
    if doc_id_col is None or seg_doc_id_col is None or seg_text_col is None:
        raise ValueError(
            "Unable to resolve AGORA schema fields (doc_id, segment doc_id, text). "
            "Run scripts/inspect_agora_schema.py first."
        )

    source_col = _pick_column(
        list(documents.columns),
        ["Link to document", "Official plaintext source", "source", "url"],
    )
    doc_tags_col = _pick_column(list(documents.columns), ["Tags", "tags"])
    name_col = _pick_column(list(documents.columns), ["Official name", "title", "name"])

    doc_meta: dict[str, dict[str, Any]] = {}
    for _, row in documents.iterrows():
        doc_id = str(int(row[doc_id_col])) if pd.notna(row[doc_id_col]) else None
        if doc_id is None:
            continue
        doc_meta[doc_id] = {
            "source": row[source_col] if source_col else None,
            "official_name": row[name_col] if name_col else None,
            "document_tags": parse_tags_cell(row[doc_tags_col]) if doc_tags_col else [],
        }

    records: list[dict[str, Any]] = []
    for idx, row in segments.iterrows():
        text_val = row.get(seg_text_col)
        if not isinstance(text_val, str) or not text_val.strip():
            continue
        doc_id_raw = row.get(seg_doc_id_col)
        if pd.isna(doc_id_raw):
            continue
        doc_id = str(int(doc_id_raw))
        seg_pos = row.get(seg_pos_col, idx + 1) if seg_pos_col else idx + 1
        seg_pos_num = int(seg_pos) if pd.notna(seg_pos) else idx + 1

        seg_tags = parse_tags_cell(row.get(seg_tags_col)) if seg_tags_col else []
        meta = doc_meta.get(doc_id, {})
        all_tags = seg_tags if seg_tags else meta.get("document_tags", [])

        records.append(
            {
                "id": f"{doc_id}_{seg_pos_num}",
                "doc_id": doc_id,
                "text": text_val.strip(),
                "tags": all_tags,
                "source": meta.get("source"),
                "official_name": meta.get("official_name"),
            }
        )
    return records
