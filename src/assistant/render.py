from __future__ import annotations

from typing import Any


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "NA"


def build_segment_note(segment_card: dict[str, Any]) -> str:
    family = str(segment_card.get("family_display_name") or segment_card.get("family") or "policy concern")
    proxy_anchor = str(segment_card.get("proxy_anchor_display_name") or segment_card.get("proxy_anchor") or "anchor proxy")
    rank = segment_card.get("document_rank")
    total_segments = segment_card.get("document_segment_count")
    concern_score = _fmt(segment_card.get("concern_score"))
    related_support_score = _fmt(segment_card.get("related_support_score"))
    reliability_score = _fmt(segment_card.get("reliability_score"))
    priority_score = _fmt(segment_card.get("priority_score"))
    supporting_feature_count = segment_card.get("supporting_feature_count")

    rank_text = f"ranked {rank} of {total_segments} in this document" if rank and total_segments else "surfaced in this document"
    feature_text = (
        f"It is supported by {supporting_feature_count} sparse features."
        if supporting_feature_count is not None
        else "It is supported by the fitted internal scoring model."
    )
    return (
        f"This segment is surfaced for {family.lower()} with {proxy_anchor.lower()} as the strongest anchor "
        f"(priority {priority_score}; {rank_text}). "
        f"The evidence combines concern score {concern_score}, related support {related_support_score}, "
        f"and reliability {reliability_score}. {feature_text}"
    )


def build_document_summary_note(document_brief: dict[str, Any]) -> str:
    dominant = document_brief.get("dominant_families", [])
    review_priority = document_brief.get("review_priority_order", [])
    if dominant:
        dominant_text = ", ".join(
            f"{row['family_display_name']} ({_fmt(row['document_family_score'])})" for row in dominant[:2]
        )
    else:
        dominant_text = "no strong family signal"
    if review_priority:
        first = review_priority[0]
        review_text = (
            f"The top review priority is {first['family_display_name']}, led by segment "
            f"{first['top_segment_id']} with priority {_fmt(first['top_priority_score'])}."
        )
    else:
        review_text = "No review priority queue is available."
    return f"The document is dominated by {dominant_text}. {review_text}"
