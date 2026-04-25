"""Proxy free evidence packages for white box and black box audit evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

from policy_interp.io import read_jsonl, write_jsonl
from policy_interp.utils import ensure_dir, normalize_text, parse_bool, stable_hash_json


CONDITION_DEFINITIONS: dict[str, dict[str, str]] = {
    "C0_passage_only": {
        "name": "Passage only",
        "description": "The auditor receives only the policy passage and non diagnostic source metadata.",
    },
    "C1_blackbox_surface": {
        "name": "Black box evidence",
        "description": "The auditor receives text only observations produced without internal model state.",
    },
    "C2_sae_only": {
        "name": "SAE only",
        "description": "The auditor receives manually revised sparse feature evidence.",
    },
    "C3_logit_lens_only": {
        "name": "Logit lens only",
        "description": "The auditor receives intermediate token direction evidence.",
    },
    "C4_steering_only": {
        "name": "Steering vector only",
        "description": "The auditor receives sensitivity evidence from internal state interventions.",
    },
    "C5_activation_oracle_only": {
        "name": "Activation oracle only",
        "description": "The auditor receives natural language explanations of hidden activations.",
    },
    "C6_full_whitebox": {
        "name": "Full white box toolkit",
        "description": "The auditor receives all available white box tools in one package.",
    },
    "C7_hybrid_blackbox_whitebox": {
        "name": "Hybrid black box and white box",
        "description": "The auditor receives text only observations and all available white box tools.",
    },
    "C8_raw_autointerp_sae": {
        "name": "Raw AutoInterp SAE ablation",
        "description": "The auditor receives unrevised AutoInterp sparse feature labels.",
    },
    "C9_shuffled_whitebox_control": {
        "name": "Shuffled white box control",
        "description": "The auditor receives white box evidence from a different case.",
    },
}


WHITEBOX_CONDITIONS = {
    "C2_sae_only",
    "C3_logit_lens_only",
    "C4_steering_only",
    "C5_activation_oracle_only",
    "C6_full_whitebox",
    "C7_hybrid_blackbox_whitebox",
    "C8_raw_autointerp_sae",
    "C9_shuffled_whitebox_control",
}


PROXY_FIELD_MARKERS = (
    "proxy",
    "family_id",
    "family_label",
    "ranking_family",
    "retrieval",
)


SOURCE_METADATA_COLUMNS = (
    "document_id",
    "segment_id",
    "split",
    "Official name",
    "Casual name",
    "authority",
    "jurisdiction",
    "collection_list",
)


AUDIT_SURFACE_TERMS = (
    "shall",
    "must",
    "required",
    "requirement",
    "obligation",
    "prohibited",
    "prohibition",
    "risk",
    "safety",
    "security",
    "transparency",
    "monitoring",
    "assessment",
    "testing",
    "threshold",
    "compliance",
    "enforcement",
    "provider",
    "operator",
    "user",
    "deploy",
    "market",
    "harm",
    "vulnerable",
    "fundamental rights",
)


@dataclass(slots=True)
class AuditEvidenceInputs:
    """Input tables for the audit evidence suite."""

    case_manifest: pd.DataFrame
    case_scores: pd.DataFrame
    manual_review: pd.DataFrame
    blackbox_observations: dict[str, list[dict[str, Any]]]
    logit_lens_items: dict[str, list[dict[str, Any]]]
    steering_items: dict[str, list[dict[str, Any]]]
    activation_oracle_items: dict[str, list[dict[str, Any]]]


@dataclass(slots=True)
class AuditEvidenceBuildResult:
    """Paths and counts from building an audit evidence evaluation bundle."""

    output_root: Path
    package_paths: dict[str, Path]
    case_manifest_path: Path
    pilot_case_manifest_path: Path
    gold_template_path: Path
    package_manifest_path: Path
    prompt_paths: dict[str, Path]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _as_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return normalize_text(str(value))


def _safe_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _row_value(row: pd.Series, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return default


def _blocked_key(key: str) -> bool:
    lower = key.lower()
    return any(marker in lower for marker in PROXY_FIELD_MARKERS)


def _drop_blocked_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_blocked_keys(v) for k, v in value.items() if not _blocked_key(str(k))}
    if isinstance(value, list):
        return [_drop_blocked_keys(item) for item in value]
    return value


def _record_hash(record: dict[str, Any]) -> str:
    return stable_hash_json(_drop_blocked_keys(record))[:16]


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target)


def _read_optional_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    target = Path(path)
    if not target.exists():
        return []
    return read_jsonl(target)


def _load_tool_items(path: str | Path | None, default_tool: str) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    if target.suffix.lower() == ".jsonl":
        records = read_jsonl(target)
    elif target.suffix.lower() == ".json":
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload if isinstance(payload, list) else payload.get("records", [])
    elif target.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if target.suffix.lower() == ".tsv" else ","
        records = pd.read_csv(target, sep=sep).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported tool evidence format: {target}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in records:
        record = _drop_blocked_keys(dict(raw))
        case_id = _as_text(record.get("case_id"))
        if not case_id:
            continue
        if isinstance(record.get("evidence_items"), str):
            try:
                parsed_items = json.loads(record["evidence_items"])
            except json.JSONDecodeError:
                parsed_items = []
        else:
            parsed_items = record.get("evidence_items")
        if isinstance(parsed_items, list):
            for item in parsed_items:
                if isinstance(item, dict):
                    grouped[case_id].append(_normalize_external_item(item, default_tool))
        else:
            grouped[case_id].append(_normalize_external_item(record, default_tool))
    return dict(grouped)


def _normalize_external_item(item: dict[str, Any], default_tool: str) -> dict[str, Any]:
    clean = _drop_blocked_keys(dict(item))
    evidence_id = _as_text(clean.get("evidence_id"))
    if not evidence_id:
        evidence_id = f"{default_tool}_{_record_hash(clean)}"
    clean["evidence_id"] = evidence_id
    clean.setdefault("tool", default_tool)
    clean.setdefault("evidence_type", default_tool)
    clean.setdefault("label_source", default_tool)
    return clean


def load_audit_evidence_inputs(
    source_root: str | Path,
    *,
    case_manifest_path: str | Path | None = None,
    case_scores_path: str | Path | None = None,
    manual_review_path: str | Path | None = None,
    blackbox_observations_path: str | Path | None = None,
    logit_lens_path: str | Path | None = None,
    steering_path: str | Path | None = None,
    activation_oracle_path: str | Path | None = None,
) -> AuditEvidenceInputs:
    """Load all tables needed to build evidence packages.

    The loader accepts optional tool evidence files so that expensive local
    model runs can be generated once and then reused by the package builder.
    """

    root = Path(source_root)
    case_manifest_target = Path(case_manifest_path) if case_manifest_path else root / "taigr_revision_eval" / "case_manifest.csv"
    case_scores_target = Path(case_scores_path) if case_scores_path else root / "taigr_revision_eval" / "case_scores.csv"
    manual_target = Path(manual_review_path) if manual_review_path else root / "paper_exports" / "manual_review_features.csv"

    if not case_manifest_target.exists():
        raise FileNotFoundError(f"Missing case manifest: {case_manifest_target}")
    if not case_scores_target.exists():
        raise FileNotFoundError(f"Missing case scores: {case_scores_target}")

    case_manifest = pd.read_csv(case_manifest_target)
    case_scores = pd.read_csv(case_scores_target)
    manual_review = _read_optional_csv(manual_target)

    return AuditEvidenceInputs(
        case_manifest=case_manifest,
        case_scores=case_scores,
        manual_review=manual_review,
        blackbox_observations=_load_tool_items(blackbox_observations_path, "blackbox"),
        logit_lens_items=_load_tool_items(logit_lens_path, "logit_lens"),
        steering_items=_load_tool_items(steering_path, "steering_vector"),
        activation_oracle_items=_load_tool_items(activation_oracle_path, "activation_oracle"),
    )


def select_cases(case_manifest: pd.DataFrame, limit: int | None = None, seed: int = 13) -> pd.DataFrame:
    """Select a deterministic case subset without using proxy labels."""

    cases = case_manifest.copy()
    if "case_id" not in cases:
        raise ValueError("case_manifest must contain case_id.")
    if "text" not in cases:
        raise ValueError("case_manifest must contain text.")
    cases = cases.drop_duplicates("case_id").sort_values("case_id").reset_index(drop=True)
    if limit is None or limit <= 0 or limit >= len(cases):
        return cases
    return cases.sample(n=limit, random_state=seed).sort_values("case_id").reset_index(drop=True)


def make_case_record(row: pd.Series) -> dict[str, Any]:
    """Create the non diagnostic case metadata visible to auditors."""

    metadata: dict[str, Any] = {}
    for column in SOURCE_METADATA_COLUMNS:
        if column in row and not _blocked_key(column):
            value = row[column]
            if not _is_missing(value):
                metadata[column] = value
    return {
        "case_id": _as_text(row["case_id"]),
        "passage": _as_text(row["text"]),
        "visible_metadata": _drop_blocked_keys(metadata),
    }


def manual_review_index(manual_review: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    """Index manual feature review rows by layer and feature id."""

    if manual_review.empty or "layer" not in manual_review or "feature_id" not in manual_review:
        return {}
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for _, row in manual_review.iterrows():
        layer = _safe_int(row.get("layer"))
        feature_id = _safe_int(row.get("feature_id"))
        if layer is None or feature_id is None:
            continue
        index[(layer, feature_id)] = _drop_blocked_keys(row.to_dict())
    return index


def build_sae_evidence_items(
    case_id: str,
    case_scores: pd.DataFrame,
    manual_index: dict[tuple[int, int], dict[str, Any]],
    *,
    max_items: int = 5,
    min_activation: float = 1e-9,
    raw_autointerp: bool = False,
) -> list[dict[str, Any]]:
    """Build SAE evidence for one case without proxy based fields."""

    if case_scores.empty:
        return []
    rows = case_scores[case_scores["case_id"].astype(str) == str(case_id)].copy()
    if rows.empty:
        return []
    sort_col = "score" if "score" in rows else "pooled_activation"
    rows = rows.sort_values(sort_col, ascending=False)
    evidence_items: list[dict[str, Any]] = []
    seen_features: set[tuple[int, int]] = set()
    for _, row in rows.iterrows():
        layer = _safe_int(row.get("layer"))
        feature_id = _safe_int(row.get("feature_id"))
        if layer is None or feature_id is None:
            continue
        feature_key = (layer, feature_id)
        if feature_key in seen_features:
            continue
        activation = _safe_float(row.get("pooled_activation")) or 0.0
        if activation <= min_activation:
            continue
        review = manual_index.get((layer, feature_id), {})
        if raw_autointerp:
            label = _as_text(_row_value(row, "generated_name", "semantic_tag", default="unknown sparse feature"))
            label_source = "autointerp_unrevised"
            decision = "unreviewed"
            paper_usable = None
            support = ""
            specificity = ""
            notes = ""
        else:
            paper_usable = parse_bool(review.get("paper_usable", review.get("use_in_paper")))
            decision = _as_text(review.get("decision")).lower()
            if not paper_usable and decision not in {"accept", "revise", "preserve"}:
                continue
            label = _as_text(
                review.get("human_label")
                or review.get("normalized_manual_taxonomy_label")
                or review.get("autointerp_label")
                or row.get("generated_name")
            )
            label_source = "human_revised_autointerp"
            support = _as_text(review.get("span_support"))
            specificity = _as_text(review.get("policy_specificity"))
            notes = _as_text(review.get("notes"))
        if not label:
            continue
        evidence_items.append(
            {
                "evidence_id": f"SAE_L{layer}_F{feature_id}",
                "tool": "sae",
                "evidence_type": "sparse_feature_activation",
                "label": label,
                "label_source": label_source,
                "manual_decision": decision,
                "paper_usable": paper_usable,
                "span_support": support,
                "policy_specificity": specificity,
                "layer": layer,
                "feature_id": feature_id,
                "activation": activation,
                "peak_token_position": _safe_int(row.get("peak_token_position")),
                "activated_span": _as_text(row.get("top_token_span_text")),
                "model_id": _as_text(_row_value(row, "model_id_x", "model_id_y", default="")),
                "sae_release": _as_text(_row_value(row, "sae_release_x", "sae_release_y", default="")),
                "caveat": "SAE labels describe activated feature concepts and should not be treated as final audit findings.",
                "review_notes": notes,
            }
        )
        seen_features.add(feature_key)
        if len(evidence_items) >= max_items:
            break
    return evidence_items


def split_sentences(text: str) -> list[str]:
    """Split policy text into coarse sentences suitable for extractive scaffolds."""

    compact = normalize_text(text)
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+(?=[A-Z0-9(])", compact)
    return [part.strip() for part in parts if part.strip()]


def build_blackbox_surface_items(case_record: dict[str, Any], max_items: int = 5) -> list[dict[str, Any]]:
    """Build a text only audit scaffold for the black box condition."""

    passage = _as_text(case_record.get("passage"))
    sentences = split_sentences(passage)
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = sum(1 for term in AUDIT_SURFACE_TERMS if term in lower)
        if score > 0:
            scored.append((score, index, sentence))
    if not scored and sentences:
        scored.append((0, 0, sentences[0]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    items: list[dict[str, Any]] = []
    for rank, (score, index, sentence) in enumerate(scored[:max_items], start=1):
        items.append(
            {
                "evidence_id": f"BB_SURFACE_{rank}",
                "tool": "blackbox",
                "evidence_type": "surface_text_observation",
                "label": "Surface obligation or risk cue",
                "label_source": "extractive_policy_text_scaffold",
                "sentence_index": index,
                "term_hit_count": score,
                "supporting_span": sentence,
                "caveat": "This item is text only evidence and contains no hidden state information.",
            }
        )
    return items


def _status(items: list[dict[str, Any]], available_label: str = "available") -> str:
    return available_label if items else "not_run_or_not_available"


def make_package(
    case_record: dict[str, Any],
    condition_id: str,
    evidence_items: list[dict[str, Any]],
    tool_status: dict[str, str] | None = None,
    *,
    shuffled_from_case_id: str | None = None,
) -> dict[str, Any]:
    """Create a complete evidence package."""

    if condition_id not in CONDITION_DEFINITIONS:
        raise ValueError(f"Unknown condition: {condition_id}")
    clean_items = [_drop_blocked_keys(item) for item in evidence_items]
    package = {
        "case_id": case_record["case_id"],
        "condition_id": condition_id,
        "condition_name": CONDITION_DEFINITIONS[condition_id]["name"],
        "condition_description": CONDITION_DEFINITIONS[condition_id]["description"],
        "passage": case_record["passage"],
        "visible_metadata": case_record["visible_metadata"],
        "evidence_items": clean_items,
        "tool_status": tool_status or {},
        "package_hash": _record_hash(
            {
                "case_id": case_record["case_id"],
                "condition_id": condition_id,
                "items": clean_items,
            }
        ),
        "audit_caveat": (
            "Use the evidence to form audit hypotheses, but cite policy spans and evidence ids separately. "
            "Do not infer policy obligations from tool labels alone."
        ),
    }
    if shuffled_from_case_id:
        package["shuffled_from_case_id"] = shuffled_from_case_id
    return _drop_blocked_keys(package)


def build_evidence_packages(
    inputs: AuditEvidenceInputs,
    *,
    case_limit: int | None = None,
    max_sae_items: int = 5,
    max_surface_items: int = 5,
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    """Build all condition packages for the selected cases."""

    cases = select_cases(inputs.case_manifest, limit=case_limit, seed=seed)
    review_index = manual_review_index(inputs.manual_review)
    packages: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITION_DEFINITIONS}
    full_whitebox_by_case: dict[str, list[dict[str, Any]]] = {}

    for _, row in cases.iterrows():
        case = make_case_record(row)
        case_id = case["case_id"]
        surface_items = inputs.blackbox_observations.get(case_id) or build_blackbox_surface_items(
            case, max_items=max_surface_items
        )
        sae_items = build_sae_evidence_items(
            case_id,
            inputs.case_scores,
            review_index,
            max_items=max_sae_items,
            raw_autointerp=False,
        )
        raw_sae_items = build_sae_evidence_items(
            case_id,
            inputs.case_scores,
            review_index,
            max_items=max_sae_items,
            raw_autointerp=True,
        )
        logit_items = inputs.logit_lens_items.get(case_id, [])
        steering_items = inputs.steering_items.get(case_id, [])
        oracle_items = inputs.activation_oracle_items.get(case_id, [])
        full_whitebox = sae_items + logit_items + steering_items + oracle_items
        full_whitebox_by_case[case_id] = full_whitebox

        whitebox_status = {
            "sae": _status(sae_items),
            "logit_lens": _status(logit_items),
            "steering_vector": _status(steering_items),
            "activation_oracle": _status(oracle_items),
        }
        packages["C0_passage_only"].append(make_package(case, "C0_passage_only", [], {}))
        packages["C1_blackbox_surface"].append(
            make_package(case, "C1_blackbox_surface", surface_items, {"blackbox": _status(surface_items)})
        )
        packages["C2_sae_only"].append(make_package(case, "C2_sae_only", sae_items, {"sae": _status(sae_items)}))
        packages["C3_logit_lens_only"].append(
            make_package(case, "C3_logit_lens_only", logit_items, {"logit_lens": _status(logit_items)})
        )
        packages["C4_steering_only"].append(
            make_package(case, "C4_steering_only", steering_items, {"steering_vector": _status(steering_items)})
        )
        packages["C5_activation_oracle_only"].append(
            make_package(
                case,
                "C5_activation_oracle_only",
                oracle_items,
                {"activation_oracle": _status(oracle_items)},
            )
        )
        packages["C6_full_whitebox"].append(make_package(case, "C6_full_whitebox", full_whitebox, whitebox_status))
        packages["C7_hybrid_blackbox_whitebox"].append(
            make_package(
                case,
                "C7_hybrid_blackbox_whitebox",
                surface_items + full_whitebox,
                {"blackbox": _status(surface_items), **whitebox_status},
            )
        )
        packages["C8_raw_autointerp_sae"].append(
            make_package(
                case,
                "C8_raw_autointerp_sae",
                raw_sae_items,
                {"sae_raw_autointerp": _status(raw_sae_items)},
            )
        )

    case_ids = [package["case_id"] for package in packages["C0_passage_only"]]
    if case_ids:
        shifted = case_ids[1:] + case_ids[:1]
        shuffled_lookup = dict(zip(case_ids, shifted, strict=False))
        for base_package in packages["C0_passage_only"]:
            case_id = base_package["case_id"]
            shuffled_case_id = shuffled_lookup[case_id]
            shuffled_items = full_whitebox_by_case.get(shuffled_case_id, [])
            status = {
                "shuffled_whitebox": _status(shuffled_items),
                "source_case_hidden_from_auditor": "true",
            }
            packages["C9_shuffled_whitebox_control"].append(
                make_package(
                    {
                        "case_id": case_id,
                        "passage": base_package["passage"],
                        "visible_metadata": base_package["visible_metadata"],
                    },
                    "C9_shuffled_whitebox_control",
                    shuffled_items,
                    status,
                    shuffled_from_case_id=shuffled_case_id,
                )
            )
    return packages


def render_auditor_prompt(package: dict[str, Any]) -> str:
    """Render the downstream auditor prompt for one evidence package."""

    compact_items = [_compact_for_prompt(item) for item in package.get("evidence_items", [])]
    visible_package = {
        "case_id": package["case_id"],
        "condition_name": package["condition_name"],
        "passage": package["passage"],
        "visible_metadata": package.get("visible_metadata", {}),
        "tool_status": package.get("tool_status", {}),
        "evidence_items": compact_items,
        "audit_caveat": package.get("audit_caveat", ""),
    }
    package_json = json.dumps(visible_package, ensure_ascii=False, indent=2)
    return (
        "You are an independent AI policy audit analyst.\n"
        "Use the policy passage as the primary source of truth. Use evidence items only as hypotheses or pointers.\n"
        "Be concise. Use at most 3 issue findings. Use at most 2 supporting spans per finding. "
        "Use at most 5 evidence ids per finding and at most 8 internal_evidence_used ids overall. "
        "Return raw JSON only, with no markdown fence and no prose before or after the JSON.\n"
        "Return a single valid JSON object with this schema:\n"
        "{\n"
        '  "summary": "one concise paragraph",\n'
        '  "issue_findings": [\n'
        "    {\n"
        '      "issue_tag": "short audit issue label",\n'
        '      "finding": "what the auditor should notice",\n'
        '      "actor": "responsible actor if present",\n'
        '      "obligation_or_risk": "obligation, risk, or gap",\n'
        '      "supporting_policy_spans": ["exact short spans from the passage"],\n'
        '      "evidence_ids": ["evidence ids used, if any"],\n'
        '      "confidence": 0.0,\n'
        '      "uncertainty": "main limitation or missing information"\n'
        "    }\n"
        "  ],\n"
        '  "confounds_or_alternatives": ["plausible alternative readings"],\n'
        '  "internal_evidence_used": ["evidence ids that materially changed the audit"],\n'
        '  "unsupported_or_low_confidence_claims": ["claims that should not be over stated"],\n'
        '  "overall_confidence": 0.0\n'
        "}\n"
        "Do not include hidden reasoning. Do not invent evidence ids. Do not claim causal proof from one tool alone.\n\n"
        f"Evidence package:\n{package_json}\n"
    )


def _compact_for_prompt(value: Any, *, max_string_chars: int = 360, max_list_items: int = 8) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"raw_output"}:
                continue
            compact[key] = _compact_for_prompt(item, max_string_chars=max_string_chars, max_list_items=max_list_items)
        return compact
    if isinstance(value, list):
        return [
            _compact_for_prompt(item, max_string_chars=max_string_chars, max_list_items=max_list_items)
            for item in value[:max_list_items]
        ]
    if isinstance(value, str) and len(value) > max_string_chars:
        return value[: max_string_chars - 15].rstrip() + " [truncated]"
    return value


def render_blackbox_generator_prompt(package: dict[str, Any]) -> str:
    """Render a text only prompt for generating black box observations."""

    payload = {
        "case_id": package["case_id"],
        "passage": package["passage"],
        "visible_metadata": package.get("visible_metadata", {}),
    }
    return (
        "You are preparing text only observations for a later independent audit.\n"
        "Read the passage and return JSON with an evidence_items list. Each item must cite an exact passage span.\n"
        "Do not use internal model state, sparse features, activations, logits, steering, or retrieval labels.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def render_judge_prompt(report: dict[str, Any], gold: dict[str, Any], package: dict[str, Any]) -> str:
    """Render a local LLM judge prompt for calibrated secondary scoring."""

    payload = {
        "report": report,
        "gold_brief": gold,
        "case_id": package.get("case_id"),
        "condition_id": package.get("condition_id"),
    }
    return (
        "You are a conservative audit report grader. The human gold brief is authoritative.\n"
        "Return JSON with component scores and short rationales. Penalize unsupported internal mechanism claims.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model output.")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : index + 1])
    raise ValueError("Unclosed JSON object in model output.")


def dry_run_audit_report(package: dict[str, Any]) -> dict[str, Any]:
    """Create a deterministic audit report for pipeline smoke tests."""

    passage = _as_text(package.get("passage"))
    spans = split_sentences(passage)[:3]
    items = package.get("evidence_items", [])
    issue_findings: list[dict[str, Any]] = []
    for index, item in enumerate(items[:3], start=1):
        span = _as_text(item.get("supporting_span") or item.get("activated_span") or (spans[0] if spans else ""))
        label = _as_text(item.get("label") or item.get("evidence_type") or "audit issue")
        issue_findings.append(
            {
                "issue_tag": label[:80],
                "finding": f"The evidence suggests checking whether the passage contains {label}.",
                "actor": _infer_actor(passage),
                "obligation_or_risk": label,
                "supporting_policy_spans": [span] if span else [],
                "evidence_ids": [_as_text(item.get("evidence_id"))] if item.get("evidence_id") else [],
                "confidence": 0.45,
                "uncertainty": "Dry run output is deterministic and not a substantive audit judgment.",
            }
        )
    if not issue_findings:
        span = spans[0] if spans else passage[:240]
        issue_findings.append(
            {
                "issue_tag": "policy obligation or risk",
                "finding": "The passage should be reviewed for explicit obligations, responsible actors, and risk controls.",
                "actor": _infer_actor(passage),
                "obligation_or_risk": "policy obligation or risk",
                "supporting_policy_spans": [span] if span else [],
                "evidence_ids": [],
                "confidence": 0.35,
                "uncertainty": "No additional evidence items were available.",
            }
        )
    return {
        "summary": "Dry run report generated for pipeline validation.",
        "issue_findings": issue_findings,
        "confounds_or_alternatives": ["The same passage can support narrower or broader legal readings."],
        "internal_evidence_used": [
            _as_text(item.get("evidence_id")) for item in items if item.get("evidence_id") and item.get("tool") != "blackbox"
        ],
        "unsupported_or_low_confidence_claims": [],
        "overall_confidence": 0.4,
    }


def normalize_audit_report(
    report: dict[str, Any],
    package: dict[str, Any] | None = None,
    *,
    max_findings: int = 3,
    max_spans: int = 2,
    max_evidence_ids: int = 5,
    max_internal_ids: int = 8,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize a model audit report to the expected JSON schema."""

    notes: list[str] = []
    normalized: dict[str, Any] = {}
    normalized["summary"] = _as_text(report.get("summary"))
    if not normalized["summary"]:
        notes.append("missing_summary")

    valid_ids = None
    if package is not None:
        valid_ids = {
            _as_text(item.get("evidence_id"))
            for item in package.get("evidence_items", [])
            if item.get("evidence_id")
        }

    findings = report.get("issue_findings")
    if not isinstance(findings, list):
        findings = []
        notes.append("issue_findings_not_list")
    if len(findings) > max_findings:
        notes.append("issue_findings_truncated")
    normalized_findings: list[dict[str, Any]] = []
    for finding in findings[:max_findings]:
        if not isinstance(finding, dict):
            notes.append("non_dict_finding_removed")
            continue
        spans = finding.get("supporting_policy_spans") or []
        if isinstance(spans, str):
            spans = [spans]
        if not isinstance(spans, list):
            spans = []
            notes.append("supporting_policy_spans_not_list")
        if len(spans) > max_spans:
            notes.append("supporting_policy_spans_truncated")
        evidence_ids = finding.get("evidence_ids") or []
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        if not isinstance(evidence_ids, list):
            evidence_ids = []
            notes.append("evidence_ids_not_list")
        cleaned_ids = [_as_text(item) for item in evidence_ids if _as_text(item)]
        if valid_ids is not None:
            invalid_count = sum(1 for item in cleaned_ids if item not in valid_ids)
            if invalid_count:
                notes.append("invalid_evidence_ids_removed")
            cleaned_ids = [item for item in cleaned_ids if item in valid_ids]
        if len(cleaned_ids) > max_evidence_ids:
            notes.append("evidence_ids_truncated")
        confidence = _safe_float(finding.get("confidence"))
        if confidence is None:
            confidence = 0.0
            notes.append("finding_confidence_missing")
        confidence = min(max(confidence, 0.0), 1.0)
        normalized_findings.append(
            {
                "issue_tag": _as_text(finding.get("issue_tag")),
                "finding": _as_text(finding.get("finding")),
                "actor": _as_text(finding.get("actor")),
                "obligation_or_risk": _as_text(finding.get("obligation_or_risk")),
                "supporting_policy_spans": [_as_text(span) for span in spans[:max_spans] if _as_text(span)],
                "evidence_ids": cleaned_ids[:max_evidence_ids],
                "confidence": confidence,
                "uncertainty": _as_text(finding.get("uncertainty")),
            }
        )
    normalized["issue_findings"] = normalized_findings

    for key in ("confounds_or_alternatives", "unsupported_or_low_confidence_claims"):
        value = report.get(key) or []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = []
            notes.append(f"{key}_not_list")
        normalized[key] = [_as_text(item) for item in value if _as_text(item)]

    internal_ids = report.get("internal_evidence_used") or []
    if isinstance(internal_ids, str):
        internal_ids = [internal_ids]
    if not isinstance(internal_ids, list):
        internal_ids = []
        notes.append("internal_evidence_used_not_list")
    cleaned_internal = [_as_text(item) for item in internal_ids if _as_text(item)]
    if valid_ids is not None:
        invalid_count = sum(1 for item in cleaned_internal if item not in valid_ids)
        if invalid_count:
            notes.append("invalid_internal_evidence_ids_removed")
        cleaned_internal = [item for item in cleaned_internal if item in valid_ids]
    if len(cleaned_internal) > max_internal_ids:
        notes.append("internal_evidence_used_truncated")
    normalized["internal_evidence_used"] = cleaned_internal[:max_internal_ids]

    overall_confidence = _safe_float(report.get("overall_confidence"))
    if overall_confidence is None:
        overall_confidence = 0.0
        notes.append("overall_confidence_missing")
    normalized["overall_confidence"] = min(max(overall_confidence, 0.0), 1.0)
    return normalized, sorted(set(notes))


def _infer_actor(text: str) -> str:
    lower = text.lower()
    for actor in ("provider", "operator", "deployer", "user", "authority", "commission", "developer"):
        if actor in lower:
            return actor
    return ""


def build_gold_template(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create human fillable gold brief templates."""

    templates: list[dict[str, Any]] = []
    for package in packages:
        passage = _as_text(package.get("passage"))
        templates.append(
            {
                "case_id": package["case_id"],
                "needs_review": True,
                "gold_issue_tags": [],
                "gold_actors": [],
                "gold_obligations": [],
                "gold_support_spans": [],
                "known_confounds": [],
                "unsupported_claims_to_penalize": [],
                "passage_preview": passage[:500],
                "reviewer_notes": "",
            }
        )
    return templates


def _norm(value: Any) -> str:
    text = _as_text(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _norm_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [item for item in (_norm(value) for value in values) if item]


def _extract_gold_values(gold: dict[str, Any], key: str, issue_field: str | None = None) -> list[str]:
    values = gold.get(key)
    if values:
        return _norm_list(values)
    issues = gold.get("issues")
    if issue_field and isinstance(issues, list):
        extracted = []
        for issue in issues:
            if isinstance(issue, dict) and issue.get(issue_field):
                extracted.append(issue[issue_field])
        return _norm_list(extracted)
    return []


def _f1(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    if not gold and not predicted:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 0.0, 0.0
    if not predicted:
        return 0.0, 0.0, 0.0
    matched_gold: set[int] = set()
    tp = 0
    for pred in predicted:
        for index, target in enumerate(gold):
            if index in matched_gold:
                continue
            if pred == target or pred in target or target in pred:
                matched_gold.add(index)
                tp += 1
                break
    precision = tp / max(len(predicted), 1)
    recall = tp / max(len(gold), 1)
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def _report_dict(report_record: dict[str, Any]) -> dict[str, Any]:
    report = report_record.get("report")
    if isinstance(report, dict):
        return report
    if isinstance(report, str):
        try:
            return extract_json_object(report)
        except ValueError:
            return {}
    return report_record


def _report_values(report: dict[str, Any], field: str) -> list[str]:
    values: list[str] = []
    for finding in report.get("issue_findings", []) or []:
        if isinstance(finding, dict):
            raw = finding.get(field)
            if isinstance(raw, list):
                values.extend(_as_text(item) for item in raw)
            elif raw:
                values.append(_as_text(raw))
    return _norm_list(values)


def _report_citation_ids(report: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for finding in report.get("issue_findings", []) or []:
        if isinstance(finding, dict):
            raw_ids = finding.get("evidence_ids") or []
            if isinstance(raw_ids, list):
                ids.extend(_as_text(item) for item in raw_ids)
    internal = report.get("internal_evidence_used") or []
    if isinstance(internal, list):
        ids.extend(_as_text(item) for item in internal)
    return [item for item in ids if item]


def _citation_validity(report: dict[str, Any], package: dict[str, Any]) -> float:
    cited = _report_citation_ids(report)
    if not cited:
        return 1.0
    valid_ids = {_as_text(item.get("evidence_id")) for item in package.get("evidence_items", []) if item.get("evidence_id")}
    if not valid_ids:
        return 0.0
    valid = sum(1 for item in cited if item in valid_ids)
    return valid / len(cited)


def score_audit_report(
    report_record: dict[str, Any],
    gold: dict[str, Any],
    package: dict[str, Any],
) -> dict[str, Any]:
    """Score one audit report against a human gold brief."""

    report = _report_dict(report_record)
    base = {
        "case_id": package.get("case_id") or report_record.get("case_id") or gold.get("case_id"),
        "condition_id": package.get("condition_id") or report_record.get("condition_id"),
        "condition_name": package.get("condition_name"),
    }
    if not gold or parse_bool(gold.get("needs_review")):
        return {
            **base,
            "score_status": "needs_gold",
            "grounded_audit_quality": None,
            "reason": "Gold brief is missing or marked as needs_review.",
        }

    issue_gold = _extract_gold_values(gold, "gold_issue_tags", "issue_tag")
    actor_gold = _extract_gold_values(gold, "gold_actors", "actor")
    obligation_gold = _extract_gold_values(gold, "gold_obligations", "obligation_or_risk")
    span_gold = _extract_gold_values(gold, "gold_support_spans", "supporting_policy_span")
    confound_gold = _extract_gold_values(gold, "known_confounds", "confound")

    issue_pred = _report_values(report, "issue_tag")
    actor_pred = _report_values(report, "actor")
    obligation_pred = _report_values(report, "obligation_or_risk")
    span_pred = _report_values(report, "supporting_policy_spans")
    confound_pred = _norm_list(report.get("confounds_or_alternatives") or [])

    issue_f1, issue_precision, issue_recall = _f1(issue_pred, issue_gold)
    actor_f1, actor_precision, actor_recall = _f1(actor_pred, actor_gold)
    obligation_f1, obligation_precision, obligation_recall = _f1(obligation_pred, obligation_gold)
    span_f1, span_precision, span_recall = _f1(span_pred, span_gold)
    confound_f1, confound_precision, confound_recall = _f1(confound_pred, confound_gold)
    citation_validity = _citation_validity(report, package)

    unsupported = report.get("unsupported_or_low_confidence_claims") or []
    unsupported_count = len(unsupported) if isinstance(unsupported, list) else 1
    expected_confidence = max(issue_f1, span_f1)
    reported_confidence = _safe_float(report.get("overall_confidence")) or 0.0
    calibration_error = min(abs(reported_confidence - expected_confidence), 1.0)

    actor_obligation_score = (actor_f1 + obligation_f1) / 2.0
    evidence_grounding = (span_f1 + citation_validity) / 2.0
    traceability = citation_validity if package.get("evidence_items") else span_f1
    non_hallucination = max(0.0, 1.0 - unsupported_count / 3.0)
    grounded_audit_quality = (
        25.0 * issue_f1
        + 15.0 * actor_obligation_score
        + 20.0 * evidence_grounding
        + 15.0 * traceability
        + 10.0 * confound_recall
        + 10.0 * (1.0 - calibration_error)
        + 5.0 * non_hallucination
    )

    return {
        **base,
        "score_status": "scored",
        "grounded_audit_quality": grounded_audit_quality,
        "issue_f1": issue_f1,
        "issue_precision": issue_precision,
        "issue_recall": issue_recall,
        "actor_f1": actor_f1,
        "actor_precision": actor_precision,
        "actor_recall": actor_recall,
        "obligation_f1": obligation_f1,
        "obligation_precision": obligation_precision,
        "obligation_recall": obligation_recall,
        "supporting_span_f1": span_f1,
        "supporting_span_precision": span_precision,
        "supporting_span_recall": span_recall,
        "evidence_citation_validity": citation_validity,
        "traceability_score": traceability,
        "confound_f1": confound_f1,
        "confound_precision": confound_precision,
        "confound_recall": confound_recall,
        "unsupported_claim_count": unsupported_count,
        "confidence_calibration_error": calibration_error,
        "evidence_item_count": len(package.get("evidence_items", [])),
    }


def summarize_scores(scores: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize scored reports by condition."""

    if not scores:
        return pd.DataFrame()
    frame = pd.DataFrame(scores)
    scored = frame[frame["score_status"] == "scored"].copy()
    if scored.empty:
        return pd.DataFrame(
            [
                {
                    "score_status": "needs_gold",
                    "report_count": len(frame),
                }
            ]
        )
    numeric_cols = [
        "grounded_audit_quality",
        "issue_f1",
        "actor_f1",
        "obligation_f1",
        "supporting_span_f1",
        "evidence_citation_validity",
        "traceability_score",
        "confound_recall",
        "unsupported_claim_count",
        "confidence_calibration_error",
        "evidence_item_count",
    ]
    summary = (
        scored.groupby(["condition_id", "condition_name"], dropna=False)[numeric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary


def write_prompt_templates(output_root: str | Path) -> dict[str, Path]:
    """Write reusable prompt templates to disk."""

    root = ensure_dir(Path(output_root) / "prompts")
    templates = {
        "auditor_prompt_v1.md": (
            "# Auditor Prompt Template\n\n"
            "Use `policy_interp.audit_evidence_suite.render_auditor_prompt(package)` to render a case specific prompt. "
            "The auditor must return a single JSON object with issue findings, policy spans, evidence ids, confounds, "
            "unsupported claims, and overall confidence.\n"
        ),
        "blackbox_generator_prompt_v1.md": (
            "# Black Box Observation Prompt Template\n\n"
            "Use `policy_interp.audit_evidence_suite.render_blackbox_generator_prompt(package)` to render a text only "
            "observation prompt. The generator must not receive hidden state, feature, logit, steering, oracle, proxy, "
            "or retrieval labels.\n"
        ),
        "judge_prompt_v1.md": (
            "# Judge Prompt Template\n\n"
            "Use `policy_interp.audit_evidence_suite.render_judge_prompt(report, gold, package)` for a calibrated local "
            "LLM judge. Human scores remain primary.\n"
        ),
    }
    paths: dict[str, Path] = {}
    for name, content in templates.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        paths[name] = path
    return paths


def write_evidence_bundle(
    packages: dict[str, list[dict[str, Any]]],
    output_root: str | Path,
    *,
    pilot_case_count: int = 12,
) -> AuditEvidenceBuildResult:
    """Write packages, case manifests, gold templates, and prompt templates."""

    root = ensure_dir(output_root)
    package_root = ensure_dir(root / "packages")
    case_root = ensure_dir(root / "cases")
    gold_root = ensure_dir(root / "gold")

    package_paths: dict[str, Path] = {}
    manifest_rows: list[dict[str, Any]] = []
    for condition_id, condition_packages in packages.items():
        path = package_root / f"{condition_id}.jsonl"
        write_jsonl(condition_packages, path)
        package_paths[condition_id] = path
        for package in condition_packages:
            manifest_rows.append(
                {
                    "case_id": package["case_id"],
                    "condition_id": condition_id,
                    "condition_name": package["condition_name"],
                    "package_hash": package["package_hash"],
                    "evidence_item_count": len(package.get("evidence_items", [])),
                    "package_path": str(path),
                }
            )

    c0_packages = packages.get("C0_passage_only", [])
    case_rows = [
        {
            "case_id": package["case_id"],
            "passage_length": len(_as_text(package.get("passage"))),
            **package.get("visible_metadata", {}),
        }
        for package in c0_packages
    ]
    case_manifest = pd.DataFrame(case_rows)
    case_manifest_path = case_root / "case_manifest.csv"
    case_manifest.to_csv(case_manifest_path, index=False)
    pilot_case_manifest_path = case_root / "pilot_case_manifest.csv"
    case_manifest.head(pilot_case_count).to_csv(pilot_case_manifest_path, index=False)

    gold_template_path = gold_root / "gold_briefs_template.jsonl"
    write_jsonl(build_gold_template(c0_packages), gold_template_path)

    package_manifest = pd.DataFrame(manifest_rows)
    package_manifest_path = package_root / "package_manifest.csv"
    package_manifest.to_csv(package_manifest_path, index=False)

    prompt_paths = write_prompt_templates(root)
    return AuditEvidenceBuildResult(
        output_root=root,
        package_paths=package_paths,
        case_manifest_path=case_manifest_path,
        pilot_case_manifest_path=pilot_case_manifest_path,
        gold_template_path=gold_template_path,
        package_manifest_path=package_manifest_path,
        prompt_paths=prompt_paths,
    )
