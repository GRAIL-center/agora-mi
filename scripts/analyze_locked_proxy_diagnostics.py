from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PRIMARY_PROXY_ORDER = [
    "risk_factors_bias",
    "harms_discrimination",
    "risk_factors_privacy",
    "harms_violation_of_civil_or_human_rights_including_privacy",
    "risk_factors_transparency",
    "risk_factors_interpretability_and_explainability",
]

MODEL_SPECS = [
    ("Gemma 2 2B", "policy_feature_benchmark_locked_2b"),
    ("Gemma 2 9B", "policy_feature_benchmark_locked_9b_primary"),
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _subset_ratio(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set:
        return 0.0
    return len(left_set & right_set) / len(left_set)


def _safe_float(value: Any) -> float:
    try:
        if value in {"", None, "nan", "NaN"}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _is_positive(value: float) -> bool:
    return not math.isnan(value) and value > 0.0


def _normalize_title(raw_title: Any, doc_id: str, doc_titles: dict[str, str]) -> str:
    title = str(raw_title or "").strip()
    if title and not title.isdigit():
        return title
    official = doc_titles.get(str(doc_id), "").strip()
    if official:
        return official
    return f"Document {doc_id}"


def _load_document_titles(manifest_root: Path, needed_doc_ids: set[str]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for family_dir in manifest_root.iterdir():
        proxies_dir = family_dir / "proxies"
        if not proxies_dir.exists():
            continue
        for proxy_dir in proxies_dir.iterdir():
            if not proxy_dir.is_dir():
                continue
            for split_name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
                path = proxy_dir / split_name
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        doc_id = str(row.get("document_id") or row.get("doc_id") or "")
                        if doc_id not in needed_doc_ids or doc_id in titles:
                            continue
                        metadata = row.get("metadata", {}) or {}
                        official_name = str(metadata.get("official_name") or "").strip()
                        if official_name:
                            titles[doc_id] = official_name
                        if len(titles) == len(needed_doc_ids):
                            return titles
    return titles


def _proxy_sort_key(proxy_slug: str) -> tuple[int, str]:
    if proxy_slug in PRIMARY_PROXY_ORDER:
        return (PRIMARY_PROXY_ORDER.index(proxy_slug), proxy_slug)
    return (len(PRIMARY_PROXY_ORDER), proxy_slug)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", required=True)
    parser.add_argument("--manifest_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    manifest_root = Path(args.manifest_root).resolve()
    output_root = _ensure_dir(Path(args.output_root).resolve())

    causal_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    concept_rows: list[dict[str, Any]] = []
    off_target_rows: list[dict[str, Any]] = []

    for model_label, folder_name in MODEL_SPECS:
        summary_dir = bundle_root / "results" / folder_name / "summary"
        if not summary_dir.exists():
            raise FileNotFoundError(f"Missing summary directory: {summary_dir}")

        for row in _read_csv(summary_dir / "proxy_causal_summary.csv"):
            row["model_label"] = model_label
            causal_rows.append(row)

        for row in _read_jsonl(summary_dir / "feature_dossiers.jsonl"):
            row["model_label"] = model_label
            feature_rows.append(row)

        for row in _read_jsonl(summary_dir / "feature_concept_cards.jsonl"):
            row["model_label"] = model_label
            concept_rows.append(row)

        for row in _read_csv(summary_dir / "proxy_off_target_effects.csv"):
            row["model_label"] = model_label
            off_target_rows.append(row)

    doc_ids: set[str] = set()
    for row in concept_rows:
        for context in list(row.get("top_contexts") or []):
            doc_id = str(context.get("document_id") or "")
            if doc_id:
                doc_ids.add(doc_id)
    titles = _load_document_titles(manifest_root, doc_ids)

    grouped_features: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    concept_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    feature_reuse: dict[tuple[str, int], set[str]] = defaultdict(set)

    for row in concept_rows:
        concept_lookup[(str(row["model_label"]), str(row["proxy_slug"]), int(row["feature_id"]))] = row

    for row in feature_rows:
        grouped_features[(str(row["model_label"]), str(row["proxy_slug"]))].append(row)
        feature_reuse[(str(row["model_label"]), int(row["feature_id"]))].add(str(row["proxy_slug"]))

    feature_semantic_inventory: list[dict[str, Any]] = []
    absorption_rows: list[dict[str, Any]] = []

    for (model_label, proxy_slug), rows in grouped_features.items():
        rows.sort(key=lambda item: int(item.get("feature_rank", 9999)))
        for row in rows:
            concept_row = concept_lookup.get((model_label, proxy_slug, int(row["feature_id"])), {})
            contexts = list(concept_row.get("top_contexts") or [])
            matched_counts = [len(context.get("matched_proxy_labels") or []) for context in contexts]
            target_name = str(row.get("proxy_name", ""))
            target_hits = 0
            off_target_titles: list[str] = []
            enriched_titles: list[str] = []
            for context in contexts:
                doc_id = str(context.get("document_id") or "")
                title = _normalize_title(context.get("document_title"), doc_id, titles)
                enriched_titles.append(title)
                labels = [str(label) for label in context.get("matched_proxy_labels") or []]
                if target_name in labels:
                    target_hits += 1
                else:
                    off_target_titles.append(title)
            purity = target_hits / len(contexts) if contexts else float("nan")
            multilabel_rate = sum(1 for count in matched_counts if count > 1) / len(matched_counts) if matched_counts else float("nan")
            reused_across = sorted(feature_reuse[(model_label, int(row["feature_id"]))] - {proxy_slug})
            feature_semantic_inventory.append(
                {
                    "model_label": model_label,
                    "proxy_slug": proxy_slug,
                    "proxy_name": row["proxy_name"],
                    "feature_id": row["feature_id"],
                    "feature_rank": row["feature_rank"],
                    "feature_weight": row.get("feature_weight"),
                    "bootstrap_stability": row.get("bootstrap_stability"),
                    "activation_gap": row.get("activation_gap"),
                    "contribution_gap": row.get("contribution_gap"),
                    "target_context_purity": purity,
                    "context_multilabel_rate": multilabel_rate,
                    "reused_across_proxy_count": len(reused_across),
                    "reused_across_proxy_slugs": "; ".join(reused_across),
                    "top_context_titles": " | ".join(enriched_titles[:3]),
                    "off_target_context_titles": " | ".join(off_target_titles[:2]),
                }
            )

        top_rows = rows[: min(10, len(rows))]
        for left_idx, left in enumerate(top_rows):
            left_segments = [str(value) for value in left.get("top_activating_segments") or []]
            left_sem = next(
                item for item in feature_semantic_inventory
                if item["model_label"] == model_label and item["proxy_slug"] == proxy_slug and int(item["feature_id"]) == int(left["feature_id"])
            )
            for right in top_rows[left_idx + 1 :]:
                right_segments = [str(value) for value in right.get("top_activating_segments") or []]
                right_sem = next(
                    item for item in feature_semantic_inventory
                    if item["model_label"] == model_label and item["proxy_slug"] == proxy_slug and int(item["feature_id"]) == int(right["feature_id"])
                )
                jaccard = _jaccard(left_segments, right_segments)
                subset_lr = _subset_ratio(left_segments, right_segments)
                subset_rl = _subset_ratio(right_segments, left_segments)
                possible_absorption = (
                    max(jaccard, subset_lr, subset_rl) >= 0.4
                    and _safe_float(right_sem["target_context_purity"]) <= _safe_float(left_sem["target_context_purity"]) + 1.0e-9
                    and _safe_float(right.get("contribution_gap")) <= _safe_float(left.get("contribution_gap")) + 1.0e-9
                )
                absorption_rows.append(
                    {
                        "model_label": model_label,
                        "proxy_slug": proxy_slug,
                        "proxy_name": left["proxy_name"],
                        "feature_id_left": left["feature_id"],
                        "feature_rank_left": left["feature_rank"],
                        "feature_id_right": right["feature_id"],
                        "feature_rank_right": right["feature_rank"],
                        "jaccard_top_contexts": jaccard,
                        "subset_ratio_left_in_right": subset_lr,
                        "subset_ratio_right_in_left": subset_rl,
                        "purity_left": left_sem["target_context_purity"],
                        "purity_right": right_sem["target_context_purity"],
                        "contribution_gap_left": left.get("contribution_gap"),
                        "contribution_gap_right": right.get("contribution_gap"),
                        "possible_absorption": possible_absorption,
                    }
                )

    stable_noncausal_rows: list[dict[str, Any]] = []
    off_target_lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in off_target_rows:
        off_target_lookup[(str(row["model_label"]), str(row["proxy_slug"]), str(row["set_name"]))].append(row)

    semantic_lookup = {
        (str(row["model_label"]), str(row["proxy_slug"]), int(row["feature_id"])): row
        for row in feature_semantic_inventory
    }

    for row in causal_rows:
        model_label = str(row["model_label"])
        proxy_slug = str(row["proxy_slug"])
        passed = str(row.get("passes_positive_causality", "")).lower() == "true"
        proxy_features = sorted(grouped_features.get((model_label, proxy_slug), []), key=lambda item: int(item.get("feature_rank", 9999)))
        if not proxy_features:
            continue
        top3 = proxy_features[:3]
        mean_stability = sum(_safe_float(item.get("bootstrap_stability")) for item in top3) / len(top3)
        if passed or mean_stability < 0.8:
            continue

        top3_semantics = [semantic_lookup[(model_label, proxy_slug, int(item["feature_id"]))] for item in top3]
        mean_purity = sum(_safe_float(item["target_context_purity"]) for item in top3_semantics) / len(top3_semantics)
        mean_multilabel = sum(_safe_float(item["context_multilabel_rate"]) for item in top3_semantics) / len(top3_semantics)
        reuse_count = sum(int(item["reused_across_proxy_count"]) for item in top3_semantics)
        top3_ids = {int(item["feature_id"]) for item in top3}
        pair_rows = [
            item for item in absorption_rows
            if item["model_label"] == model_label and item["proxy_slug"] == proxy_slug
            and int(item["feature_id_left"]) in top3_ids and int(item["feature_id_right"]) in top3_ids
        ]
        max_overlap = max((_safe_float(item["jaccard_top_contexts"]) for item in pair_rows), default=float("nan"))
        possible_absorption_pairs = sum(1 for item in pair_rows if bool(item["possible_absorption"]))
        best_k = f"top{row.get('best_k')}"
        off_targets = off_target_lookup.get((model_label, proxy_slug, best_k), [])
        mean_off_target = sum(abs(_safe_float(item.get("off_target_selectivity"))) for item in off_targets) / len(off_targets) if off_targets else float("nan")

        tags: list[str] = []
        if mean_purity < 0.6:
            tags.append("mixed_semantics")
        if mean_multilabel > 0.6 or reuse_count > 1:
            tags.append("cross_proxy_reuse")
        if possible_absorption_pairs > 0:
            tags.append("possible_absorption")
        if _safe_float(row.get("n_high_confidence_features")) >= 6 and abs(_safe_float(row.get("causal_selectivity"))) < 0.05:
            tags.append("distributed_nonselective_signal")
        if _is_positive(mean_off_target) and mean_off_target > 0.05:
            tags.append("off_target_spillover")
        if not tags:
            tags.append("weak_selective_effect")

        stable_noncausal_rows.append(
            {
                "model_label": model_label,
                "proxy_slug": proxy_slug,
                "proxy_name": row["proxy_name"],
                "selected_layer": row.get("selected_layer"),
                "n_high_confidence_features": row.get("n_high_confidence_features"),
                "best_k": row.get("best_k"),
                "causal_selectivity": row.get("causal_selectivity"),
                "off_target_selectivity": row.get("off_target_selectivity"),
                "mean_top3_stability": mean_stability,
                "mean_target_context_purity_top3": mean_purity,
                "mean_multilabel_rate_top3": mean_multilabel,
                "reused_proxy_count_top3": reuse_count,
                "max_top_context_overlap_top3": max_overlap,
                "possible_absorption_pairs_top3": possible_absorption_pairs,
                "diagnosis_tags": "; ".join(tags),
                "top_feature_ids": ", ".join(str(item["feature_id"]) for item in top3),
            }
        )

    failure_case_rows: list[dict[str, Any]] = []
    for row in feature_rows:
        if str(row["proxy_slug"]) not in {"risk_factors_bias", "risk_factors_privacy"} or int(row.get("feature_rank", 9999)) > 3:
            continue
        concept_row = concept_lookup.get((str(row["model_label"]), str(row["proxy_slug"]), int(row["feature_id"])), {})
        target_name = str(row.get("proxy_name", ""))
        for context in list(concept_row.get("top_contexts") or []):
            labels = [str(label) for label in context.get("matched_proxy_labels") or []]
            if target_name in labels and len(labels) <= 2:
                continue
            doc_id = str(context.get("document_id") or "")
            failure_case_rows.append(
                {
                    "model_label": row["model_label"],
                    "proxy_slug": row["proxy_slug"],
                    "proxy_name": row["proxy_name"],
                    "feature_id": row["feature_id"],
                    "feature_rank": row["feature_rank"],
                    "segment_id": context.get("segment_id"),
                    "document_id": doc_id,
                    "document_title": _normalize_title(context.get("document_title"), doc_id, titles),
                    "activation": context.get("activation"),
                    "matched_proxy_labels": "; ".join(labels),
                    "text_excerpt": context.get("text_excerpt", ""),
                }
            )

    _write_csv(output_root / "feature_semantic_inventory.csv", feature_semantic_inventory)
    _write_csv(output_root / "feature_absorption_diagnostics.csv", absorption_rows)
    _write_csv(output_root / "stable_noncausal_proxy_diagnostics.csv", stable_noncausal_rows)
    _write_csv(output_root / "feature_failure_cases.csv", failure_case_rows)

    report_lines = [
        "# Locked Proxy Diagnostics",
        "",
        "This report summarizes semantic feature inspection, approximate absorption checks, and stable but non-causal proxy diagnostics from the locked result bundle.",
        "",
        "## Primary qualitative feature readout",
        "",
    ]

    focus_rows = [
        row for row in feature_semantic_inventory
        if row["proxy_slug"] in {"risk_factors_bias", "risk_factors_privacy"} and int(row["feature_rank"]) <= 3
    ]
    focus_rows.sort(key=lambda item: (item["model_label"], _proxy_sort_key(item["proxy_slug"]), int(item["feature_rank"])))
    for row in focus_rows:
        report_lines.extend(
            [
                f"### {row['model_label']} | {row['proxy_name']} | Feature {row['feature_id']}",
                "",
                f"1. Rank: {row['feature_rank']}",
                f"2. Weight: {float(row['feature_weight']):.4f}",
                f"3. Stability: {float(row['bootstrap_stability']):.3f}",
                f"4. Activation gap: {float(row['activation_gap']):.3f}",
                f"5. Contribution gap: {float(row['contribution_gap']):.3f}",
                f"6. Target context purity: {float(row['target_context_purity']):.3f}",
                f"7. Context multilabel rate: {float(row['context_multilabel_rate']):.3f}",
                f"8. Top context titles: {row['top_context_titles']}",
            ]
        )
        if row["off_target_context_titles"]:
            report_lines.append(f"9. Potential failure contexts: {row['off_target_context_titles']}")
        if row["reused_across_proxy_slugs"]:
            report_lines.append(f"10. Cross-proxy reuse: {row['reused_across_proxy_slugs']}")
        report_lines.append("")

    report_lines.extend(["## Stable but non-causal proxies", ""])
    for row in sorted(stable_noncausal_rows, key=lambda item: (item["model_label"], _proxy_sort_key(item["proxy_slug"]))):
        report_lines.extend(
            [
                f"### {row['model_label']} | {row['proxy_name']}",
                "",
                f"1. Layer: {row['selected_layer']}",
                f"2. Top features: {row['top_feature_ids']}",
                f"3. Causal selectivity: {float(row['causal_selectivity']):.4f}",
                f"4. Mean top3 stability: {float(row['mean_top3_stability']):.3f}",
                f"5. Mean target context purity: {float(row['mean_target_context_purity_top3']):.3f}",
                f"6. Mean multilabel rate: {float(row['mean_multilabel_rate_top3']):.3f}",
                f"7. Possible absorption pairs: {row['possible_absorption_pairs_top3']}",
                f"8. Diagnosis tags: {row['diagnosis_tags']}",
                "",
            ]
        )

    report_lines.extend(["## Approximate absorption diagnostics", ""])
    top_absorption = sorted(
        [row for row in absorption_rows if row["possible_absorption"]],
        key=lambda item: (-_safe_float(item["jaccard_top_contexts"]), item["model_label"], item["proxy_slug"]),
    )[:12]
    for row in top_absorption:
        report_lines.append(
            f"1. {row['model_label']} | {row['proxy_name']} | Features {row['feature_id_left']} and {row['feature_id_right']} | Jaccard {float(row['jaccard_top_contexts']):.3f} | Purity {float(row['purity_left']):.3f} to {float(row['purity_right']):.3f}"
        )

    (output_root / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
