from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from data.io import normalize_tag


SYNONYMS: dict[str, list[str]] = {
    "audit": ["external_auditing", "conformity_assessment", "post_market_monitoring"],
    "penalty": ["fines", "liability", "imprisonment"],
    "liability": ["civil_liability", "criminal_liability"],
    "funding_support": ["government_support", "government_support_for_r_d", "subsidies"],
    "r_and_d_support": ["government_support_for_r_d"],
    "standards_support": ["governance_development", "government_study_or_report"],
    "mandate": ["licensing_registration_and_certification", "performance_requirements", "input_controls"],
}


@dataclass(frozen=True)
class LabelMap:
    dsafe_tags: list[str]
    dinnov_tags: list[str]
    overlap_policy: str = "drop"


def _expand_queries(queries: Iterable[str]) -> list[str]:
    out: list[str] = []
    for q in queries:
        qn = normalize_tag(q)
        if not qn:
            continue
        out.append(qn)
        for syn in SYNONYMS.get(qn, []):
            out.append(normalize_tag(syn))
    # keep order, drop duplicates
    seen = set()
    uniq: list[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def normalize_tags(tags: Iterable[str]) -> set[str]:
    return {normalize_tag(t) for t in tags if normalize_tag(t)}


def _matches_any(tags_norm: set[str], query_norm: list[str]) -> bool:
    if not tags_norm or not query_norm:
        return False
    for q in query_norm:
        for t in tags_norm:
            if q == t:
                return True
            if q in t:
                return True
    return False


def assign_membership(tags: Iterable[str], label_map: LabelMap) -> tuple[bool, bool]:
    tags_norm = normalize_tags(tags)
    safe_queries = _expand_queries(label_map.dsafe_tags)
    innov_queries = _expand_queries(label_map.dinnov_tags)
    is_safe = _matches_any(tags_norm, safe_queries)
    is_innov = _matches_any(tags_norm, innov_queries)
    return is_safe, is_innov


def assign_label(tags: Iterable[str], label_map: LabelMap) -> str | None:
    is_safe, is_innov = assign_membership(tags, label_map)
    if is_safe and is_innov:
        if label_map.overlap_policy == "keep_both":
            return "both"
        return None
    if is_safe:
        return "safe"
    if is_innov:
        return "innov"
    return None
