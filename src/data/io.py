from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


AGORA_BINARY_TAG_PREFIXES = (
    "Applications:",
    "Harms:",
    "Incentives:",
    "Risk factors:",
    "Strategies:",
)


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


def is_truthy_cell(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return False
        return bool(int(value))
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


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


def parse_semicolon_cell(value: Any) -> list[str]:
    return parse_tags_cell(value)


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _binary_tag_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith(AGORA_BINARY_TAG_PREFIXES)]


def _extract_true_binary_tags(row: Any, binary_columns: list[str]) -> list[str]:
    return [col for col in binary_columns if is_truthy_cell(row.get(col))]


def _merge_unique_values(*value_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in value_groups:
        for value in group:
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _extract_prefixed_tags(tags: list[str], prefix: str) -> list[str]:
    return [tag for tag in tags if tag.startswith(prefix)]


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_year(*values: Any) -> int | None:
    for value in values:
        text = _coerce_optional_str(value)
        if not text:
            continue
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            return int(match.group(0))
    return None


def _infer_jurisdiction(authority: str | None, collection_values: list[str]) -> str:
    authority_text = (authority or "").lower()
    collection_text = " ; ".join(collection_values).lower()
    if "u.s. federal" in collection_text or "united states congress" in authority_text:
        return "us_federal"
    if "state and local" in collection_text or authority_text in {"california", "new york"}:
        return "us_state_local"
    if "chinese" in collection_text or "china" in authority_text:
        return "china"
    if "european union" in collection_text or "european union" in authority_text:
        return "eu"
    if "multinational" in collection_text:
        return "multinational"
    if "private-sector companies" in authority_text or "private-sector" in authority_text or "company" in authority_text:
        return "private_sector"
    return "other"


def _infer_document_form(
    authority: str | None,
    collection_values: list[str],
    source: str | None,
    official_name: str | None,
) -> str:
    authority_text = (authority or "").lower()
    collection_text = " ; ".join(collection_values).lower()
    source_text = (source or "").lower()
    name_text = (official_name or "").lower()
    if "federal laws" in collection_text or "state and local documents" in collection_text or "/bill/" in source_text:
        return "law"
    if "regulations" in collection_text or "executive order" in collection_text or "president" in authority_text:
        return "regulation_or_order"
    if "private-sector companies" in authority_text or "company" in authority_text:
        return "corporate_policy"
    if "multinational" in collection_text:
        return "multinational_framework"
    if "plan" in name_text or "strategy" in name_text or "report" in name_text:
        return "plan_or_report"
    if "miscellaneous" in collection_text:
        return "miscellaneous"
    return "policy_document"


def _collection_domains(collection_values: list[str]) -> list[str]:
    return [normalize_tag(value) for value in collection_values if normalize_tag(value)]


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

    doc_columns = list(documents.columns)
    seg_columns = list(segments.columns)

    doc_id_col = _pick_column(doc_columns, ["AGORA ID", "Document ID", "doc_id"])
    seg_doc_id_col = _pick_column(seg_columns, ["Document ID", "AGORA ID", "doc_id"])
    seg_text_col = _pick_column(seg_columns, ["Text", "text", "full_text"])
    seg_tags_col = _pick_column(seg_columns, ["Tags", "tags"])
    seg_pos_col = _pick_column(seg_columns, ["Segment position", "position", "segment_index"])
    if doc_id_col is None or seg_doc_id_col is None or seg_text_col is None:
        raise ValueError(
            "Unable to resolve AGORA schema fields (doc_id, segment doc_id, text). "
            "Run scripts/inspect_agora_schema.py first."
        )

    source_col = _pick_column(
        doc_columns,
        ["Link to document", "Official plaintext source", "source", "url"],
    )
    plaintext_source_col = _pick_column(doc_columns, ["Official plaintext source", "official_plaintext_source"])
    pdf_source_col = _pick_column(doc_columns, ["Official pdf source", "official_pdf_source"])
    doc_tags_col = _pick_column(doc_columns, ["Tags", "tags"])
    name_col = _pick_column(doc_columns, ["Official name", "title", "name"])
    authority_col = _pick_column(doc_columns, ["Authority", "authority"])
    collections_col = _pick_column(doc_columns, ["Collections", "collections"])
    doc_activity_col = _pick_column(doc_columns, ["Most recent activity", "most_recent_activity"])
    doc_activity_date_col = _pick_column(doc_columns, ["Most recent activity date", "most_recent_activity_date"])
    doc_proposed_date_col = _pick_column(doc_columns, ["Proposed date", "proposed_date"])
    doc_annotated_col = _pick_column(doc_columns, ["Annotated?", "annotated"])
    doc_validated_col = _pick_column(doc_columns, ["Validated?", "validated"])
    doc_gov_col = _pick_column(
        doc_columns,
        ["Primarily applies to the government", "primarily_applies_to_the_government"],
    )
    doc_private_col = _pick_column(
        doc_columns,
        ["Primarily applies to the private sector", "primarily_applies_to_the_private_sector"],
    )
    seg_summary_col = _pick_column(seg_columns, ["Summary", "summary"])
    seg_non_operative_col = _pick_column(seg_columns, ["Non-operative", "non_operative"])
    seg_not_ai_related_col = _pick_column(seg_columns, ["Not AI-related", "not_ai_related"])
    seg_annotated_col = _pick_column(seg_columns, ["Segment annotated", "segment_annotated"])
    seg_validated_col = _pick_column(seg_columns, ["Segment validated", "segment_validated"])
    seg_unreviewed_machine_col = _pick_column(
        seg_columns,
        ["Summaries and tags may include unreviewed machine output", "summaries_and_tags_may_include_unreviewed_machine_output"],
    )

    doc_meta: dict[str, dict[str, Any]] = {}
    for _, row in documents.iterrows():
        doc_id = str(int(row[doc_id_col])) if pd.notna(row[doc_id_col]) else None
        if doc_id is None:
            continue
        collection_values = parse_semicolon_cell(row[collections_col]) if collections_col else []
        authority = _coerce_optional_str(row[authority_col]) if authority_col else None
        official_name = _coerce_optional_str(row[name_col]) if name_col else None
        primary_source = _coerce_optional_str(row[source_col]) if source_col else None
        plaintext_source = _coerce_optional_str(row[plaintext_source_col]) if plaintext_source_col else None
        pdf_source = _coerce_optional_str(row[pdf_source_col]) if pdf_source_col else None
        year = _parse_year(
            row[doc_proposed_date_col] if doc_proposed_date_col else None,
            row[doc_activity_date_col] if doc_activity_date_col else None,
        )
        doc_meta[doc_id] = {
            "source": primary_source,
            "official_plaintext_source": plaintext_source,
            "official_pdf_source": pdf_source,
            "official_name": official_name,
            "document_tags": parse_tags_cell(row[doc_tags_col]) if doc_tags_col else [],
            "authority": authority,
            "collections": _coerce_optional_str(row[collections_col]) if collections_col else None,
            "collection_values": collection_values,
            "collection_domains": _collection_domains(collection_values),
            "jurisdiction": _infer_jurisdiction(authority, collection_values),
            "document_form": _infer_document_form(
                authority,
                collection_values,
                primary_source or plaintext_source,
                official_name,
            ),
            "year": year,
            "most_recent_activity": _coerce_optional_str(row[doc_activity_col]) if doc_activity_col else None,
            "document_most_recent_activity_date": row[doc_activity_date_col] if doc_activity_date_col else None,
            "document_proposed_date": row[doc_proposed_date_col] if doc_proposed_date_col else None,
            "document_annotated": is_truthy_cell(row[doc_annotated_col]) if doc_annotated_col else None,
            "document_validated": is_truthy_cell(row[doc_validated_col]) if doc_validated_col else None,
            "document_applies_government": is_truthy_cell(row[doc_gov_col]) if doc_gov_col else None,
            "document_applies_private_sector": is_truthy_cell(row[doc_private_col]) if doc_private_col else None,
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
        effective_string_tags = seg_tags if seg_tags else meta.get("document_tags", [])
        all_tags = _merge_unique_values(effective_string_tags)
        application_tags = _extract_prefixed_tags(all_tags, "Applications:")
        risk_tags = _extract_prefixed_tags(all_tags, "Risk factors:")
        harm_tags = _extract_prefixed_tags(all_tags, "Harms:")
        incentive_tags = _extract_prefixed_tags(all_tags, "Incentives:")
        strategy_tags = _extract_prefixed_tags(all_tags, "Strategies:")

        records.append(
            {
                "id": f"{doc_id}_{seg_pos_num}",
                "segment_id": f"{doc_id}_{seg_pos_num}",
                "doc_id": doc_id,
                "document_id": doc_id,
                "text": text_val.strip(),
                "tags": effective_string_tags,
                "all_tags": all_tags,
                "segment_tags": seg_tags,
                "document_tags": meta.get("document_tags", []),
                "application_tags": application_tags,
                "risk_tags": risk_tags,
                "harm_tags": harm_tags,
                "incentive_tags": incentive_tags,
                "strategy_tags": strategy_tags,
                "summary": row[seg_summary_col] if seg_summary_col else None,
                "non_operative": is_truthy_cell(row[seg_non_operative_col]) if seg_non_operative_col else None,
                "not_ai_related": is_truthy_cell(row[seg_not_ai_related_col]) if seg_not_ai_related_col else None,
                "segment_annotated": is_truthy_cell(row[seg_annotated_col]) if seg_annotated_col else None,
                "segment_validated": is_truthy_cell(row[seg_validated_col]) if seg_validated_col else None,
                "segment_unreviewed_machine_output": (
                    is_truthy_cell(row[seg_unreviewed_machine_col]) if seg_unreviewed_machine_col else None
                ),
                "source": meta.get("source"),
                "official_plaintext_source": meta.get("official_plaintext_source"),
                "official_pdf_source": meta.get("official_pdf_source"),
                "official_name": meta.get("official_name"),
                "authority": meta.get("authority"),
                "collections": meta.get("collections"),
                "collection_values": meta.get("collection_values", []),
                "collection_domains": meta.get("collection_domains", []),
                "jurisdiction": meta.get("jurisdiction"),
                "document_form": meta.get("document_form"),
                "year": meta.get("year"),
                "most_recent_activity": meta.get("most_recent_activity"),
                "document_most_recent_activity_date": meta.get("document_most_recent_activity_date"),
                "document_proposed_date": meta.get("document_proposed_date"),
                "document_annotated": meta.get("document_annotated"),
                "document_validated": meta.get("document_validated"),
                "document_applies_government": meta.get("document_applies_government"),
                "document_applies_private_sector": meta.get("document_applies_private_sector"),
            }
        )
    return records
