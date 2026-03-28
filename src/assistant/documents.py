from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def normalize_document_input(
    payload: str | list[str] | list[dict[str, Any]] | dict[str, Any],
    *,
    document_id: str = "document_1",
    title: str = "Untitled document",
    source_type: str = "user_text",
) -> dict[str, Any]:
    if isinstance(payload, str):
        raw_text = payload
        paragraphs: list[str] = []
        segments_provided = False
        segment_payloads: list[dict[str, Any]] = []
    elif isinstance(payload, list):
        if not payload:
            raw_text = ""
            paragraphs = []
            segments_provided = False
            segment_payloads = []
        elif all(isinstance(item, str) for item in payload):
            paragraphs = [str(item).strip() for item in payload if str(item).strip()]
            raw_text = "\n\n".join(paragraphs)
            segments_provided = False
            segment_payloads = []
        elif all(isinstance(item, dict) for item in payload):
            segment_payloads = [dict(item) for item in payload]
            raw_text = "\n\n".join(str(item.get("text", "")).strip() for item in segment_payloads if str(item.get("text", "")).strip())
            paragraphs = []
            segments_provided = True
        else:
            raise TypeError("List payloads must be either all strings or all dictionaries.")
    elif isinstance(payload, dict):
        raw_text = str(payload.get("raw_text", "") or "")
        paragraphs = [str(item).strip() for item in payload.get("paragraphs", []) if str(item).strip()]
        segment_payloads = [dict(item) for item in payload.get("segments", [])]
        if not raw_text and paragraphs:
            raw_text = "\n\n".join(paragraphs)
        if not raw_text and segment_payloads:
            raw_text = "\n\n".join(str(item.get("text", "")).strip() for item in segment_payloads if str(item.get("text", "")).strip())
        segments_provided = bool(segment_payloads)
        document_id = str(payload.get("document_id") or document_id)
        title = str(payload.get("title") or title)
        source_type = str(payload.get("source_type") or source_type)
    else:
        raise TypeError("Unsupported payload type for document normalization.")

    return {
        "document_id": str(document_id),
        "title": str(title),
        "source_type": str(source_type),
        "raw_text": raw_text,
        "segments_provided": bool(segments_provided),
        "paragraphs": paragraphs,
        "provided_segments": segment_payloads if segments_provided else [],
    }


def _paragraph_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S[\s\S]*?(?:(?:\n\s*\n)|\Z)", text):
        start, end = match.start(), match.end()
        piece = text[start:end].strip()
        if piece:
            spans.append((start, end, piece))
    return spans


def _chunk_segments(text: str, *, chunk_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if not text.strip():
        return segments
    start = 0
    index = 1
    step = max(chunk_chars - overlap_chars, 1)
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk_text = text[start:end].strip()
        if chunk_text:
            segments.append(
                {
                    "segment_id": f"seg_{index}",
                    "char_start": start,
                    "char_end": end,
                    "section_hint": None,
                    "segment_text": chunk_text,
                }
            )
            index += 1
        if end >= len(text):
            break
        start += step
    return segments


def segment_document(
    normalized_document: dict[str, Any],
    *,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[dict[str, Any]]:
    if normalized_document.get("segments_provided"):
        output: list[dict[str, Any]] = []
        for index, row in enumerate(normalized_document.get("provided_segments", []), start=1):
            text = str(row.get("text", "") or row.get("segment_text", "")).strip()
            if not text:
                continue
            output.append(
                {
                    "segment_id": str(row.get("segment_id") or f"seg_{index}"),
                    "char_start": int(row.get("char_start", 0) or 0),
                    "char_end": int(row.get("char_end", row.get("char_stop", len(text))) or len(text)),
                    "section_hint": row.get("section_hint"),
                    "segment_text": text,
                }
            )
        return output

    raw_text = str(normalized_document.get("raw_text", "") or "")
    paragraph_spans = _paragraph_spans(raw_text)
    if len(paragraph_spans) >= 2:
        return [
            {
                "segment_id": f"seg_{index}",
                "char_start": start,
                "char_end": end,
                "section_hint": None,
                "segment_text": piece,
            }
            for index, (start, end, piece) in enumerate(paragraph_spans, start=1)
        ]
    return _chunk_segments(raw_text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)


def load_document_payload(path: str | Path) -> str | list[str] | list[dict[str, Any]] | dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        import json

        return json.loads(text)
    return text
