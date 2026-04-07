"""Compute-only audit evaluation for policy-relevant model behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from policy_interp.adapters.modeling import HuggingFaceBackboneAdapter, SaeLensAdapter
from policy_interp.batch_scorer import (
    _build_document_feature_scores,
    _build_layer_profile_summary,
    _build_proxy_overlay_summary,
    _load_feature_label_frame,
)
from policy_interp.extract import run_extraction_for_segments
from policy_interp.interventions import (
    _build_sequence_baseline_cache,
    _compute_sequence_change_metrics,
    _load_active_feature_pool,
    _mean_metric_dicts,
    _sample_random_feature_sets,
    _stable_seed_offset,
    bootstrap_ci,
    sparse_ablation_editor,
)
from policy_interp.io import read_parquet, write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir, normalize_text, set_seed


@dataclass(slots=True)
class AuditEvaluationArtifacts:
    case_manifest_path: str
    discriminativeness_summary_path: str
    robustness_summary_path: str
    autointerp_validation_path: str
    failure_transparency_path: str
    causal_concentration_path: str


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def run_audit_evaluation(config: ExperimentConfig) -> AuditEvaluationArtifacts:
    set_seed(config.splits.seed)
    audit_root = ensure_dir(config.run_root / "audit_eval")
    feature_root = config.run_root / "features"

    segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    documents = read_parquet(config.run_root / config.dataset.prepared_documents_name)
    catalog = read_parquet(feature_root / "feature_catalog.parquet")
    segment_scores = read_parquet(feature_root / "feature_catalog_segment_scores.parquet")
    overlay = read_parquet(feature_root / "feature_proxy_overlay.parquet")
    labels = _load_feature_label_frame(feature_root)
    autointerp_scores = read_parquet(feature_root / "autointerp" / "autointerp_feature_scores.parquet")
    autointerp_simulation = read_parquet(feature_root / "autointerp" / "autointerp_feature_simulation.parquet")
    exemplars = read_parquet(feature_root / "feature_catalog_exemplars.parquet")
    feature_summary = read_parquet(feature_root / "feature_summary.parquet")
    baseline_comparison = _read_optional_parquet(config.run_root / "baselines" / "baseline_comparison.parquet")
    causal_summary = _read_optional_parquet(config.run_root / "interventions" / "feature_causal_summary.parquet")

    case_manifest = _build_case_manifest(config, segments, documents)
    write_parquet(case_manifest, audit_root / "audit_case_manifest.parquet")

    case_scores = _build_case_feature_scores(case_manifest, segment_scores, catalog, labels)
    write_parquet(case_scores, audit_root / "audit_case_feature_scores.parquet")

    discriminativeness_summary, pairwise_similarity, retrieval_summary = _compute_discriminativeness(
        config=config,
        case_manifest=case_manifest,
        case_scores=case_scores,
    )
    write_parquet(discriminativeness_summary, audit_root / "audit_discriminativeness_summary.parquet")
    write_parquet(pairwise_similarity, audit_root / "audit_discriminativeness_pairwise.parquet")
    write_parquet(retrieval_summary, audit_root / "audit_discriminativeness_retrieval.parquet")

    robustness_summary, robustness_case_scores = _compute_report_robustness(
        config=config,
        case_manifest=case_manifest,
        catalog=catalog,
        labels=labels,
        overlay=overlay,
    )
    write_parquet(robustness_summary, audit_root / "audit_report_robustness_summary.parquet")
    write_parquet(robustness_case_scores, audit_root / "audit_report_robustness_case_scores.parquet")

    autointerp_validation = _compute_autointerp_validation(
        autointerp_scores=autointerp_scores,
        autointerp_simulation=autointerp_simulation,
        exemplars=exemplars,
    )
    write_parquet(autointerp_validation, audit_root / "autointerp_validation_feature_scores.parquet")
    write_parquet(_summarize_autointerp_validation(autointerp_validation, config), audit_root / "autointerp_validation_layer_summary.parquet")

    failure_transparency = _compute_failure_transparency(
        config=config,
        feature_summary=feature_summary,
        baseline_comparison=baseline_comparison,
        causal_summary=causal_summary,
        overlay=overlay,
        catalog=catalog,
    )
    write_parquet(failure_transparency, audit_root / "failure_transparency_summary.parquet")

    causal_concentration = _compute_audit_causal_concentration(
        config=config,
        case_manifest=case_manifest,
        autointerp_scores=autointerp_scores,
    )
    write_parquet(causal_concentration, audit_root / "audit_causal_concentration_summary.parquet")

    return AuditEvaluationArtifacts(
        case_manifest_path=str(audit_root / "audit_case_manifest.parquet"),
        discriminativeness_summary_path=str(audit_root / "audit_discriminativeness_summary.parquet"),
        robustness_summary_path=str(audit_root / "audit_report_robustness_summary.parquet"),
        autointerp_validation_path=str(audit_root / "autointerp_validation_feature_scores.parquet"),
        failure_transparency_path=str(audit_root / "failure_transparency_summary.parquet"),
        causal_concentration_path=str(audit_root / "audit_causal_concentration_summary.parquet"),
    )


def export_audit_reports(config: ExperimentConfig, export_root: Path) -> dict[str, str]:
    audit_root = config.run_root / "audit_eval"
    outputs: dict[str, str] = {}
    if not audit_root.exists():
        return outputs

    discriminativeness_path = audit_root / "audit_discriminativeness_summary.parquet"
    pairwise_path = audit_root / "audit_discriminativeness_pairwise.parquet"
    retrieval_path = audit_root / "audit_discriminativeness_retrieval.parquet"
    if discriminativeness_path.exists() and pairwise_path.exists() and retrieval_path.exists():
        discriminativeness = read_parquet(discriminativeness_path)
        pairwise = read_parquet(pairwise_path)
        retrieval = read_parquet(retrieval_path)
        csv_path = export_root / "audit_discriminativeness_summary.csv"
        discriminativeness.to_csv(csv_path, index=False)
        outputs["audit_discriminativeness_summary"] = str(csv_path)
        figure_path = export_root / "audit_discriminativeness.png"
        _export_audit_discriminativeness_figure(discriminativeness, pairwise, retrieval, figure_path)
        outputs["audit_discriminativeness_figure"] = str(figure_path)

    robustness_path = audit_root / "audit_report_robustness_summary.parquet"
    if robustness_path.exists():
        robustness = read_parquet(robustness_path)
        csv_path = export_root / "audit_report_robustness_summary.csv"
        robustness.to_csv(csv_path, index=False)
        outputs["audit_report_robustness_summary"] = str(csv_path)
        figure_path = export_root / "audit_report_robustness.png"
        _export_audit_robustness_figure(robustness, figure_path)
        outputs["audit_report_robustness_figure"] = str(figure_path)

    autointerp_path = audit_root / "autointerp_validation_feature_scores.parquet"
    autointerp_layer_path = audit_root / "autointerp_validation_layer_summary.parquet"
    if autointerp_path.exists() and autointerp_layer_path.exists():
        per_feature = read_parquet(autointerp_path)
        layer_summary = read_parquet(autointerp_layer_path)
        per_feature_csv = export_root / "autointerp_validation_feature_scores.csv"
        layer_csv = export_root / "autointerp_validation_layer_summary.csv"
        per_feature.to_csv(per_feature_csv, index=False)
        layer_summary.to_csv(layer_csv, index=False)
        outputs["autointerp_validation_feature_scores"] = str(per_feature_csv)
        outputs["autointerp_validation_layer_summary"] = str(layer_csv)

    failure_path = audit_root / "failure_transparency_summary.parquet"
    if failure_path.exists():
        failure = read_parquet(failure_path)
        csv_path = export_root / "failure_transparency_summary.csv"
        failure.to_csv(csv_path, index=False)
        outputs["failure_transparency_summary"] = str(csv_path)
        figure_path = export_root / "failure_transparency.png"
        _export_failure_transparency_figure(failure, figure_path)
        outputs["failure_transparency_figure"] = str(figure_path)

    causal_concentration_path = audit_root / "audit_causal_concentration_summary.parquet"
    if causal_concentration_path.exists():
        concentration = read_parquet(causal_concentration_path)
        csv_path = export_root / "audit_causal_concentration_summary.csv"
        concentration.to_csv(csv_path, index=False)
        outputs["audit_causal_concentration_summary"] = str(csv_path)
        figure_path = export_root / "audit_causal_concentration.png"
        _export_audit_causal_concentration_figure(concentration, figure_path)
        outputs["audit_causal_concentration_figure"] = str(figure_path)

    return outputs


def _build_case_manifest(
    config: ExperimentConfig,
    segments: pd.DataFrame,
    documents: pd.DataFrame,
) -> pd.DataFrame:
    split_order = {name: idx for idx, name in enumerate(config.audit.preferred_splits)}
    selected_rows: list[pd.DataFrame] = []
    documents_metadata = documents[["document_id", "Official name", "Casual name", "authority", "jurisdiction", "collection_list"]].copy()
    for family in config.audit.families:
        family_segments = segments.loc[segments["document_id"].isin(family.document_ids)].copy()
        if family_segments.empty:
            continue
        family_segments = family_segments.loc[
            family_segments["text"].astype(str).str.len().between(config.audit.segment_length_min, config.audit.segment_length_max)
        ].copy()
        if family_segments.empty:
            continue
        family_segments["split_priority"] = family_segments["split"].map(split_order).fillna(len(split_order))
        proxy_columns = list(config.dataset.proxy_columns.keys())
        family_segments["proxy_count"] = family_segments[proxy_columns].sum(axis=1)
        family_segments["segment_length"] = family_segments["text"].astype(str).str.len()
        family_segments = family_segments.sort_values(
            ["split_priority", "proxy_count", "segment_length", "segment_id"],
            ascending=[True, False, False, True],
        ).head(family.max_cases)
        family_segments["family_id"] = family.family_id
        family_segments["family_label"] = family.family_label
        selected_rows.append(family_segments)

    if not selected_rows:
        return pd.DataFrame()

    manifest = pd.concat(selected_rows, ignore_index=True)
    manifest = manifest.merge(documents_metadata, on="document_id", how="left")
    rename_candidates = {
        "authority_y": "authority",
        "jurisdiction_y": "jurisdiction",
        "collection_list_y": "collection_list",
    }
    for source_name, target_name in rename_candidates.items():
        if target_name not in manifest.columns and source_name in manifest.columns:
            manifest = manifest.rename(columns={source_name: target_name})
    manifest["case_id"] = manifest.apply(
        lambda row: f"{row['family_id']}_{str(row['segment_id']).replace('/', '_')}",
        axis=1,
    )
    return manifest[
        [
            "case_id",
            "family_id",
            "family_label",
            "document_id",
            "segment_id",
            "split",
            "text",
            "segment_length",
            "proxy_count",
            "Official name",
            "Casual name",
            "authority",
            "jurisdiction",
            "collection_list",
        ]
    ].reset_index(drop=True)


def _build_case_feature_scores(
    case_manifest: pd.DataFrame,
    segment_scores: pd.DataFrame,
    catalog: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    if case_manifest.empty:
        return pd.DataFrame()
    catalog_metadata = catalog[
        [
            "model_id",
            "sae_release",
            "layer",
            "feature_id",
            "ranking_family",
            "rank",
            "score",
            "best_proxy",
            "best_proxy_test_auc",
            "best_proxy_validated_auc",
            "causal_badge",
        ]
    ].drop_duplicates()
    labels_frame = labels[
        ["layer", "feature_id", "ranking_family", "rank", "generated_name", "semantic_tag"]
    ].drop_duplicates()
    merged = case_manifest[["case_id", "family_id", "family_label", "segment_id", "document_id", "split", "text"]].merge(
        segment_scores,
        on=["segment_id", "document_id", "split"],
        how="left",
    )
    merged = merged.merge(catalog_metadata, on=["layer", "feature_id"], how="left")
    merged = merged.merge(labels_frame, on=["layer", "feature_id", "ranking_family", "rank"], how="left")
    merged["feature_key"] = merged.apply(lambda row: f"L{int(row['layer'])}_F{int(row['feature_id'])}", axis=1)
    merged["generated_name"] = merged["generated_name"].fillna("")
    merged["semantic_tag"] = merged["semantic_tag"].fillna("unknown")
    return merged.dropna(subset=["feature_id", "layer"]).reset_index(drop=True)


def _compute_discriminativeness(
    config: ExperimentConfig,
    case_manifest: pd.DataFrame,
    case_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    views = {
        "overall_view": _build_case_feature_matrix(case_scores, ranking_family=None, top_k=config.audit.overall_top_k),
        "policy_specific_view": _build_case_feature_matrix(case_scores, ranking_family="policy_specific", top_k=config.audit.policy_specific_top_k),
        "layer_profile_view": _build_layer_profile_matrix(case_scores),
    }
    jaccard_views = {
        "overall_view": _build_topk_sets(case_scores, ranking_family=None, top_k=config.audit.overall_top_k),
        "policy_specific_view": _build_topk_sets(case_scores, ranking_family="policy_specific", top_k=config.audit.policy_specific_top_k),
    }

    pairwise_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    family_map = case_manifest.set_index("case_id")["family_id"].to_dict()

    for view_name, matrix in views.items():
        if matrix.empty:
            continue
        cosine_matrix = _cosine_similarity_matrix(matrix)
        case_ids = matrix.index.tolist()
        for left_index, left_case in enumerate(case_ids):
            for right_index in range(left_index + 1, len(case_ids)):
                right_case = case_ids[right_index]
                same_family = family_map[left_case] == family_map[right_case]
                row = {
                    "view_name": view_name,
                    "left_case_id": left_case,
                    "right_case_id": right_case,
                    "same_family": same_family,
                    "cosine_similarity": float(cosine_matrix[left_index, right_index]),
                }
                if view_name in jaccard_views:
                    left_set = jaccard_views[view_name].get(left_case, set())
                    right_set = jaccard_views[view_name].get(right_case, set())
                    row["topk_jaccard"] = _set_jaccard(left_set, right_set)
                else:
                    row["topk_jaccard"] = np.nan
                pairwise_rows.append(row)

        predicted = _nearest_family_retrieval(matrix, family_map)
        for case_id, record in predicted.items():
            retrieval_rows.append(
                {
                    "view_name": view_name,
                    "case_id": case_id,
                    "gold_family_id": family_map[case_id],
                    "predicted_family_id": record["predicted_family_id"],
                    "best_similarity": record["best_similarity"],
                    "correct": int(record["predicted_family_id"] == family_map[case_id]),
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    for view_name in sorted(views):
        subset = pairwise.loc[pairwise["view_name"] == view_name].copy()
        retrieve_subset = retrieval.loc[retrieval["view_name"] == view_name].copy()
        if subset.empty or retrieve_subset.empty:
            continue
        within = subset.loc[subset["same_family"], "cosine_similarity"]
        between = subset.loc[~subset["same_family"], "cosine_similarity"]
        jaccard_within = subset.loc[subset["same_family"], "topk_jaccard"]
        jaccard_between = subset.loc[~subset["same_family"], "topk_jaccard"]
        summary_rows.append(
            {
                "view_name": view_name,
                "mean_within_cosine": float(within.mean()),
                "mean_between_cosine": float(between.mean()),
                "cosine_gap": float(within.mean() - between.mean()),
                "mean_within_jaccard": float(jaccard_within.mean()) if not jaccard_within.isna().all() else np.nan,
                "mean_between_jaccard": float(jaccard_between.mean()) if not jaccard_between.isna().all() else np.nan,
                "jaccard_gap": float(jaccard_within.mean() - jaccard_between.mean()) if not jaccard_within.isna().all() else np.nan,
                "retrieval_accuracy": float(retrieve_subset["correct"].mean()),
                "case_count": int(len(retrieve_subset)),
            }
        )
    return pd.DataFrame(summary_rows), pairwise, retrieval


def _build_case_feature_matrix(
    case_scores: pd.DataFrame,
    ranking_family: str | None,
    top_k: int,
) -> pd.DataFrame:
    frame = case_scores.copy()
    if ranking_family is not None:
        frame = frame.loc[frame["ranking_family"] == ranking_family].copy()
    if frame.empty:
        return pd.DataFrame()
    frame = (
        frame.sort_values(["case_id", "pooled_activation", "layer", "feature_id"], ascending=[True, False, True, True])
        .groupby("case_id", group_keys=False)
        .head(top_k)
        .copy()
    )
    matrix = frame.pivot_table(index="case_id", columns="feature_key", values="pooled_activation", aggfunc="max").fillna(0.0)
    return _row_normalize(matrix)


def _build_layer_profile_matrix(case_scores: pd.DataFrame) -> pd.DataFrame:
    if case_scores.empty:
        return pd.DataFrame()
    summary = (
        case_scores.groupby(["case_id", "layer"])
        .agg(
            mean_pooled_activation=("pooled_activation", "mean"),
            max_pooled_activation=("pooled_activation", "max"),
            activated_catalog_features=("feature_id", "nunique"),
        )
        .reset_index()
    )
    pivot = summary.pivot(index="case_id", columns="layer")
    pivot.columns = [f"{metric}_L{int(layer)}" for metric, layer in pivot.columns]
    return _row_normalize(pivot.fillna(0.0))


def _build_topk_sets(
    case_scores: pd.DataFrame,
    ranking_family: str | None,
    top_k: int,
) -> dict[str, set[str]]:
    frame = case_scores.copy()
    if ranking_family is not None:
        frame = frame.loc[frame["ranking_family"] == ranking_family].copy()
    if frame.empty:
        return {}
    trimmed = (
        frame.sort_values(["case_id", "pooled_activation", "layer", "feature_id"], ascending=[True, False, True, True])
        .groupby("case_id", group_keys=False)
        .head(top_k)
        .copy()
    )
    return {
        case_id: set(group["feature_key"].astype(str).tolist())
        for case_id, group in trimmed.groupby("case_id")
    }


def _row_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    values = frame.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = values / norms
    return pd.DataFrame(normalized, index=frame.index, columns=frame.columns)


def _cosine_similarity_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    return values @ values.T


def _set_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def _nearest_family_retrieval(
    matrix: pd.DataFrame,
    family_map: dict[str, str],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    values = matrix.to_numpy(dtype=float)
    case_ids = matrix.index.tolist()
    family_ids = sorted({family_map[case_id] for case_id in case_ids})
    for index, case_id in enumerate(case_ids):
        row = values[index]
        best_family = None
        best_similarity = -1.0
        for family_id in family_ids:
            member_indices = [
                member_index
                for member_index, other_case in enumerate(case_ids)
                if other_case != case_id and family_map[other_case] == family_id
            ]
            if not member_indices:
                continue
            centroid = values[member_indices].mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0.0:
                continue
            similarity = float(row @ (centroid / centroid_norm))
            if similarity > best_similarity:
                best_similarity = similarity
                best_family = family_id
        results[case_id] = {
            "predicted_family_id": best_family or family_map[case_id],
            "best_similarity": best_similarity if best_similarity > -1.0 else 0.0,
        }
    return results


def _compute_report_robustness(
    config: ExperimentConfig,
    case_manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    labels: pd.DataFrame,
    overlay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if case_manifest.empty:
        return pd.DataFrame(), pd.DataFrame()
    variants = _build_perturbed_case_manifest(config, case_manifest)
    scored = _score_case_variants(config, variants, catalog, labels, overlay)
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()

    original_scores = scored.loc[scored["perturbation"] == "original"].copy()
    summary_rows: list[dict[str, object]] = []
    for perturbation in config.audit.perturbations:
        perturbed_scores = scored.loc[scored["perturbation"] == perturbation].copy()
        if perturbed_scores.empty:
            continue
        comparison = _compare_original_and_perturbed(
            original_scores=original_scores,
            perturbed_scores=perturbed_scores,
        )
        comparison["perturbation"] = perturbation
        summary_rows.append(comparison)
    return pd.DataFrame(summary_rows), scored


def _build_perturbed_case_manifest(config: ExperimentConfig, case_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in case_manifest.itertuples(index=False):
        base_row = row._asdict()
        base_row["source_case_id"] = row.case_id
        base_row["perturbation"] = "original"
        rows.append(base_row)
        for perturbation in config.audit.perturbations:
            perturbed_text = _apply_perturbation(str(row.text), perturbation, config)
            perturbed_row = dict(base_row)
            perturbed_row["case_id"] = f"{row.case_id}__{perturbation}"
            perturbed_row["text"] = perturbed_text
            perturbed_row["perturbation"] = perturbation
            rows.append(perturbed_row)
    return pd.DataFrame(rows)


def _apply_perturbation(text: str, perturbation: str, config: ExperimentConfig) -> str:
    normalized = normalize_text(text)
    if perturbation == "heading_removal":
        stripped = re.sub(
            r"^\s*(chapter\s+[ivxlcdm]+[:.]?\s*|section\s+\d+[:.]?\s*|article\s+\d+[:.]?\s*)",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"^\s*[A-Z][A-Za-z0-9 ,()'\"/]{0,80}:\s*", "", stripped)
        return normalize_text(stripped)
    if perturbation == "sentence_compression":
        sentences = [item for item in re.split(r"(?<=[.!?])\s+", normalized) if item]
        if len(sentences) <= config.audit.sentence_compression_keep_sentences:
            return normalized
        return normalize_text(" ".join(sentences[: config.audit.sentence_compression_keep_sentences]))
    if perturbation == "lexical_anchor_masking":
        content_words = re.findall(r"[A-Za-z]{5,}", normalized)
        anchors: list[str] = []
        for word in content_words:
            lowered = word.lower()
            if lowered not in anchors:
                anchors.append(lowered)
            if len(anchors) >= config.audit.perturbation_anchor_top_k:
                break
        masked = normalized
        for anchor in anchors:
            masked = re.sub(rf"\b{re.escape(anchor)}\b", "[MASK]", masked, flags=re.IGNORECASE)
        return normalize_text(masked)
    return normalized


def _score_case_variants(
    config: ExperimentConfig,
    variants: pd.DataFrame,
    catalog: pd.DataFrame,
    labels: pd.DataFrame,
    overlay: pd.DataFrame,
) -> pd.DataFrame:
    scoring_segments = variants[["case_id", "text"]].copy().rename(columns={"case_id": "segment_id"})
    scoring_segments["document_id"] = np.arange(len(scoring_segments), dtype=int)
    scoring_segments["split"] = "audit_eval"

    extraction_root = ensure_dir(config.run_root / "audit_eval" / "perturbation_extraction")
    scoring_config = config.model_copy(deep=True)
    scoring_config.extract.layers = sorted(catalog["layer"].dropna().astype(int).unique().tolist())
    scoring_config.extract.segment_top_feature_count = max(scoring_config.extract.segment_top_feature_count, 256)

    backbone = HuggingFaceBackboneAdapter(scoring_config.backbone).load()
    sae_loader = SaeLensAdapter(scoring_config.sae)
    try:
        run_extraction_for_segments(
            config=scoring_config,
            prepared_segments=scoring_segments[["segment_id", "document_id", "split", "text"]],
            extraction_root=extraction_root,
            backbone_bundle=backbone,
            sae_loader=sae_loader,
            sae_cache={},
        )
    finally:
        if hasattr(backbone, "model"):
            del backbone.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    extraction_frames: list[pd.DataFrame] = []
    for layer in scoring_config.extract.layers:
        path = extraction_root / f"segment_top_features_layer_{layer}.parquet"
        if path.exists():
            extraction_frames.append(read_parquet(path))
    if not extraction_frames:
        return pd.DataFrame()
    extraction_features = pd.concat(extraction_frames, ignore_index=True)
    feature_scores = _build_document_feature_scores(
        extraction_features=extraction_features,
        catalog=catalog,
        labels=labels,
        segments=scoring_segments[["segment_id", "text"]],
    )
    feature_scores = feature_scores.merge(
        variants[["case_id", "source_case_id", "family_id", "family_label", "perturbation"]].rename(columns={"case_id": "segment_id"}),
        on="segment_id",
        how="left",
    )

    overlay_summary_rows: list[pd.DataFrame] = []
    for segment_id, group in feature_scores.groupby("segment_id"):
        proxy_summary = _build_proxy_overlay_summary(group, overlay)
        if proxy_summary.empty:
            continue
        proxy_summary["segment_id"] = segment_id
        overlay_summary_rows.append(proxy_summary)
    overlay_summary = pd.concat(overlay_summary_rows, ignore_index=True) if overlay_summary_rows else pd.DataFrame()

    policy_sets = _build_variant_top_feature_sets(feature_scores, ranking_family="policy_specific", top_k=config.audit.policy_specific_top_k)
    overall_sets = _build_variant_top_feature_sets(feature_scores, ranking_family=None, top_k=config.audit.overall_top_k)
    proxy_ranks = _build_variant_proxy_rankings(overlay_summary)
    layer_vectors = _build_variant_layer_vectors(feature_scores)

    rows: list[dict[str, object]] = []
    variant_rows = variants.drop_duplicates(["case_id", "source_case_id", "perturbation"])[
        ["case_id", "source_case_id", "family_id", "family_label", "perturbation"]
    ]
    for row in variant_rows.itertuples(index=False):
        rows.append(
            {
                "case_id": row.case_id,
                "source_case_id": row.source_case_id,
                "family_id": row.family_id,
                "family_label": row.family_label,
                "perturbation": row.perturbation,
                "policy_specific_feature_set": sorted(policy_sets.get(row.case_id, set())),
                "overall_feature_set": sorted(overall_sets.get(row.case_id, set())),
                "proxy_ranking": proxy_ranks.get(row.case_id, []),
                "layer_vector": layer_vectors.get(row.case_id, {}),
            }
        )
    return pd.DataFrame(rows)


def _build_variant_top_feature_sets(
    feature_scores: pd.DataFrame,
    ranking_family: str | None,
    top_k: int,
) -> dict[str, set[str]]:
    frame = feature_scores.copy()
    if ranking_family is not None:
        frame = frame.loc[frame["ranking_family"] == ranking_family].copy()
    if frame.empty:
        return {}
    frame["feature_key"] = frame.apply(lambda row: f"L{int(row['layer'])}_F{int(row['feature_id'])}", axis=1)
    trimmed = (
        frame.sort_values(["segment_id", "pooled_activation", "layer", "feature_id"], ascending=[True, False, True, True])
        .groupby("segment_id", group_keys=False)
        .head(top_k)
    )
    return {
        segment_id: set(group["feature_key"].astype(str).tolist())
        for segment_id, group in trimmed.groupby("segment_id")
    }


def _build_variant_proxy_rankings(proxy_summary: pd.DataFrame) -> dict[str, list[str]]:
    if proxy_summary.empty:
        return {}
    return {
        segment_id: group.sort_values("total_weighted_proxy_signal", ascending=False)["proxy"].astype(str).tolist()
        for segment_id, group in proxy_summary.groupby("segment_id")
    }


def _build_variant_layer_vectors(feature_scores: pd.DataFrame) -> dict[str, dict[str, float]]:
    if feature_scores.empty:
        return {}
    summary = (
        feature_scores.groupby(["segment_id", "layer"])
        .agg(
            mean_pooled_activation=("pooled_activation", "mean"),
            max_pooled_activation=("pooled_activation", "max"),
            activated_catalog_features=("feature_id", "nunique"),
        )
        .reset_index()
    )
    vectors: dict[str, dict[str, float]] = {}
    for segment_id, group in summary.groupby("segment_id"):
        vector: dict[str, float] = {}
        for row in group.itertuples(index=False):
            vector[f"L{int(row.layer)}_mean"] = float(row.mean_pooled_activation)
            vector[f"L{int(row.layer)}_max"] = float(row.max_pooled_activation)
            vector[f"L{int(row.layer)}_count"] = float(row.activated_catalog_features)
        vectors[str(segment_id)] = vector
    return vectors


def _compare_original_and_perturbed(
    original_scores: pd.DataFrame,
    perturbed_scores: pd.DataFrame,
) -> dict[str, object]:
    original = original_scores.set_index("source_case_id")
    perturbed = perturbed_scores.set_index("source_case_id")
    common_ids = sorted(set(original.index.tolist()) & set(perturbed.index.tolist()))
    policy_overlap: list[float] = []
    overall_overlap: list[float] = []
    proxy_rank_stability: list[float] = []
    layer_profile_stability: list[float] = []
    for case_id in common_ids:
        orig = original.loc[case_id]
        pert = perturbed.loc[case_id]
        policy_overlap.append(_set_jaccard(set(orig["policy_specific_feature_set"]), set(pert["policy_specific_feature_set"])))
        overall_overlap.append(_set_jaccard(set(orig["overall_feature_set"]), set(pert["overall_feature_set"])))
        proxy_rank_stability.append(_rank_list_spearman(orig["proxy_ranking"], pert["proxy_ranking"]))
        layer_profile_stability.append(_dict_cosine(orig["layer_vector"], pert["layer_vector"]))
    return {
        "case_count": len(common_ids),
        "mean_policy_specific_jaccard": float(np.nanmean(policy_overlap)) if policy_overlap else np.nan,
        "mean_overall_jaccard": float(np.nanmean(overall_overlap)) if overall_overlap else np.nan,
        "mean_proxy_rank_spearman": float(np.nanmean(proxy_rank_stability)) if proxy_rank_stability else np.nan,
        "mean_layer_profile_cosine": float(np.nanmean(layer_profile_stability)) if layer_profile_stability else np.nan,
    }


def _rank_list_spearman(left: list[str], right: list[str]) -> float:
    items = list(dict.fromkeys([*left, *right]))
    if not items:
        return np.nan
    default_rank = len(items) + 1
    left_ranks = np.array([left.index(item) + 1 if item in left else default_rank for item in items], dtype=float)
    right_ranks = np.array([right.index(item) + 1 if item in right else default_rank for item in items], dtype=float)
    if np.all(left_ranks == left_ranks[0]) or np.all(right_ranks == right_ranks[0]):
        return np.nan
    return float(pd.Series(left_ranks).corr(pd.Series(right_ranks), method="spearman"))


def _dict_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left.keys()) | set(right.keys()))
    if not keys:
        return np.nan
    left_vec = np.array([float(left.get(key, 0.0)) for key in keys], dtype=float)
    right_vec = np.array([float(right.get(key, 0.0)) for key in keys], dtype=float)
    left_norm = np.linalg.norm(left_vec)
    right_norm = np.linalg.norm(right_vec)
    if left_norm == 0.0 or right_norm == 0.0:
        return np.nan
    return float((left_vec @ right_vec) / (left_norm * right_norm))


def _compute_autointerp_validation(
    autointerp_scores: pd.DataFrame,
    autointerp_simulation: pd.DataFrame,
    exemplars: pd.DataFrame,
) -> pd.DataFrame:
    if autointerp_scores.empty:
        return pd.DataFrame()
    simulation = autointerp_simulation.copy()
    if simulation.empty:
        output = autointerp_scores.copy()
        output["contrastive_accuracy"] = np.nan
        output["lexicality_penalty"] = np.nan
        return output

    simulation["signed_confidence"] = simulation.apply(
        lambda row: (float(row["confidence"]) / 100.0) if int(row["predicted_label"]) == 1 else -(float(row["confidence"]) / 100.0),
        axis=1,
    )

    contrastive_rows: list[dict[str, object]] = []
    for key, group in simulation.groupby(["layer", "feature_id", "primary_ranking_family"]):
        positives = group.loc[group["gold_label"] == 1, "signed_confidence"].astype(float).tolist()
        negatives = group.loc[group["gold_label"] == 0, "signed_confidence"].astype(float).tolist()
        if not positives or not negatives:
            contrastive_accuracy = np.nan
        else:
            wins = [
                1.0 if positive > negative else 0.5 if math.isclose(positive, negative) else 0.0
                for positive in positives
                for negative in negatives
            ]
            contrastive_accuracy = float(np.mean(wins))
        contrastive_rows.append(
            {
                "layer": int(key[0]),
                "feature_id": int(key[1]),
                "primary_ranking_family": str(key[2]),
                "contrastive_accuracy": contrastive_accuracy,
            }
        )
    contrastive = pd.DataFrame(contrastive_rows)

    lexicality_rows: list[dict[str, object]] = []
    positive_exemplars = exemplars.loc[exemplars["example_kind"] == "positive"].copy()
    for key, group in positive_exemplars.groupby(["layer", "feature_id", "ranking_family"]):
        snippets = " ".join(group["top_token_span_text"].fillna("").astype(str).tolist())
        lexicality_rows.append(
            {
                "layer": int(key[0]),
                "feature_id": int(key[1]),
                "primary_ranking_family": str(key[2]),
                "snippet_text": snippets,
            }
        )
    lexicality = pd.DataFrame(lexicality_rows)

    merged = autointerp_scores.merge(
        contrastive,
        on=["layer", "feature_id", "primary_ranking_family"],
        how="left",
    ).merge(
        lexicality,
        on=["layer", "feature_id", "primary_ranking_family"],
        how="left",
    )
    merged["lexicality_penalty"] = merged.apply(
        lambda row: _lexical_overlap_score(
            " ".join(
                [
                    str(row.get("feature_name", "")),
                    str(row.get("activation_hypothesis", "")),
                    str(row.get("boundary_text", "")),
                ]
            ),
            str(row.get("snippet_text", "")),
        ),
        axis=1,
    )
    return merged.drop(columns=["snippet_text"], errors="ignore")


def _summarize_autointerp_validation(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    threshold = config.report.autointerp_high_faithfulness_threshold
    return (
        frame.groupby("layer", as_index=False)
        .agg(
            feature_count=("feature_id", "count"),
            mean_faithfulness=("faithfulness_score", "mean"),
            count_high_faithfulness=("faithfulness_score", lambda col: int((col >= threshold).sum())),
            mean_contrastive_accuracy=("contrastive_accuracy", "mean"),
            mean_lexicality_penalty=("lexicality_penalty", "mean"),
        )
        .sort_values("layer")
    )


def _lexical_overlap_score(label_text: str, span_text: str) -> float:
    label_tokens = _content_tokens(label_text)
    span_tokens = _content_tokens(span_text)
    if not label_tokens or not span_tokens:
        return np.nan
    overlap = len(label_tokens & span_tokens)
    return float(overlap / max(1, len(label_tokens)))


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "this",
        "that",
        "when",
        "then",
        "text",
        "feature",
        "activates",
        "activate",
        "policy",
        "document",
        "documents",
        "should",
        "stay",
        "strongly",
        "related",
        "context",
        "lack",
        "same",
        "concept",
        "pattern",
    }
    return {token for token in re.findall(r"[A-Za-z]{4,}", text.lower()) if token not in stopwords}


def _compute_failure_transparency(
    config: ExperimentConfig,
    feature_summary: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    causal_summary: pd.DataFrame,
    overlay: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not feature_summary.empty:
        top_global = (
            catalog.loc[catalog["ranking_family"] == "global_dominance", ["layer", "feature_id", "rank"]]
            .sort_values(["layer", "rank"])
            .groupby("layer", group_keys=False)
            .head(config.audit.overall_top_k)
        )
        merged_global = top_global.merge(
            feature_summary[["layer", "feature_id", "activation_frequency"]],
            on=["layer", "feature_id"],
            how="left",
        )
        rows.append(
            {
                "metric_name": "top_overall_activation_frequency_one_rate",
                "metric_value": float((merged_global["activation_frequency"] >= 0.999999).mean()),
                "metric_note": "Fraction of top global-dominance features with activation frequency effectively equal to one.",
            }
        )

    if not baseline_comparison.empty:
        paired = baseline_comparison.pivot_table(index="proxy", columns="method", values="test_auc", aggfunc="mean")
        if {"sparse_individual_feature_probe", "sparse_module_probe"}.issubset(set(paired.columns)):
            gaps = paired["sparse_individual_feature_probe"] - paired["sparse_module_probe"]
            rows.append(
                {
                    "metric_name": "individual_minus_module_auc_gap",
                    "metric_value": float(gaps.mean()),
                    "metric_note": "Mean test AUC gap between sparse individual features and sparse modules.",
                }
            )

    if not causal_summary.empty:
        single = causal_summary.loc[causal_summary["target_type"] == "autointerp_single_feature"].copy()
        if not single.empty:
            near_zero = (
                single["kl_divergence_delta"].abs() <= config.audit.near_zero_kl_threshold
            ) & (
                single["paired_delta"].abs() <= config.audit.near_zero_paired_threshold
            )
            rows.append(
                {
                    "metric_name": "single_feature_near_zero_effect_rate",
                    "metric_value": float(near_zero.mean()),
                    "metric_note": "Fraction of AutoInterp single-feature targets with near-zero proxy-free and proxy-dependent effects.",
                }
            )

    if not overlay.empty:
        top_policy = (
            catalog.loc[catalog["ranking_family"] == "policy_specific", ["layer", "feature_id", "rank"]]
            .sort_values(["layer", "rank"])
            .groupby("layer", group_keys=False)
            .head(config.audit.policy_specific_top_k)
        )
        overlay_subset = overlay.merge(top_policy[["layer", "feature_id"]], on=["layer", "feature_id"], how="inner")
        margins: list[float] = []
        for _, group in overlay_subset.groupby(["layer", "feature_id"]):
            scores = group.sort_values("test_auc", ascending=False)["test_auc"].astype(float).tolist()
            if len(scores) >= 2:
                margins.append(scores[0] - scores[1])
        if margins:
            rows.append(
                {
                    "metric_name": "flat_proxy_overlay_rate",
                    "metric_value": float(np.mean(np.array(margins) < config.audit.flat_proxy_margin_threshold)),
                    "metric_note": "Fraction of top policy-specific features with shallow best-versus-runner-up proxy margin.",
                }
            )
    return pd.DataFrame(rows)


def _compute_audit_causal_concentration(
    config: ExperimentConfig,
    case_manifest: pd.DataFrame,
    autointerp_scores: pd.DataFrame,
) -> pd.DataFrame:
    if case_manifest.empty or autointerp_scores.empty:
        return pd.DataFrame()

    candidate_families = set(config.ablation.autointerp_target_ranking_families)
    filtered = autointerp_scores.loc[
        autointerp_scores["primary_ranking_family"].isin(candidate_families)
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    scoring_config = config.model_copy(deep=True)
    scoring_config.ablation.bootstrap_iterations = min(
        int(config.audit.causal_concentration_bootstrap_iterations),
        int(config.ablation.bootstrap_iterations),
    )
    rows = (
        case_manifest.sort_values(["family_id", "split", "segment_id"])
        .groupby("family_id", group_keys=False)
        .head(int(config.audit.causal_concentration_cases_per_family))[["segment_id", "document_id", "split", "text"]]
        .copy()
    )
    if rows.empty:
        return pd.DataFrame()
    rows["text"] = rows["text"].map(_compress_for_causal_snippet)
    concentration_rows: list[dict[str, object]] = []
    feature_pool_cache: dict[int, list[int]] = {}
    sae_cache: dict[int, object] = {}
    backbone = HuggingFaceBackboneAdapter(scoring_config.backbone).load()
    sae_loader = SaeLensAdapter(scoring_config.sae)
    baseline_cache: dict[str, object] = {}

    try:
        for row in rows.itertuples(index=False):
            cache = _build_sequence_baseline_cache(
                model_bundle=backbone,
                text=str(row.text),
                max_length=min(192, int(scoring_config.backbone.max_length)),
            )
            if cache is not None:
                baseline_cache[str(row.segment_id)] = cache
        for layer, layer_frame in filtered.groupby("layer", sort=True):
            ranked = layer_frame.sort_values(
                ["faithfulness_score", "simulation_accuracy", "rank"],
                ascending=[False, False, True],
            ).drop_duplicates(["feature_id"])
            if float(config.ablation.autointerp_min_faithfulness) > 0:
                qualified = ranked.loc[ranked["faithfulness_score"] >= float(config.ablation.autointerp_min_faithfulness)].copy()
                if not qualified.empty:
                    ranked = qualified
            if ranked.empty:
                continue

            feature_pool = feature_pool_cache.get(int(layer))
            if feature_pool is None:
                feature_pool = _load_active_feature_pool(config, int(layer))
                feature_pool_cache[int(layer)] = feature_pool
            sae = sae_cache.get(int(layer))
            if sae is None:
                sae = sae_loader.load_for_layer(int(layer))
                sae_cache[int(layer)] = sae

            set_sizes = [1] + sorted({int(size) for size in config.ablation.autointerp_feature_set_sizes if int(size) > 1})
            for set_size in set_sizes:
                selected = ranked.head(set_size).copy()
                if len(selected) < set_size:
                    continue
                feature_ids = [int(value) for value in selected["feature_id"].tolist()]
                random_sets = _sample_random_feature_sets(
                    feature_pool=feature_pool,
                    exclude_features=feature_ids,
                    feature_count=set_size,
                    trials=min(
                        int(config.audit.causal_concentration_random_trials),
                        int(config.ablation.random_control_trials),
                    ),
                    seed=int(config.splits.seed + int(layer) + set_size + _stable_seed_offset(f"audit_{layer}_{set_size}")),
                )
                proxy_free_metrics = _evaluate_cached_proxy_free_effects(
                    rows=rows,
                    baseline_cache=baseline_cache,
                    model_bundle=backbone,
                    layer=int(layer),
                    sae=sae,
                    target_features=feature_ids,
                    random_feature_sets=random_sets,
                    config=scoring_config,
                )
                concentration_rows.append(
                    {
                        "layer": int(layer),
                        "set_size": int(set_size),
                        "target_id": f"audit_layer_{int(layer)}_top_{int(set_size)}",
                        "target_type": "autointerp_faithful_set",
                        "feature_ids": feature_ids,
                        "feature_names": selected["feature_name"].fillna("").astype(str).tolist(),
                        "primary_proxy": str(selected["best_proxy"].astype(str).mode().iloc[0]) if "best_proxy" in selected.columns and not selected["best_proxy"].dropna().empty else "unknown",
                        "mean_faithfulness": float(selected["faithfulness_score"].mean()),
                        "mean_simulation_accuracy": float(selected["simulation_accuracy"].mean()),
                        "case_count": int(len(rows)),
                        "kl_divergence_delta": proxy_free_metrics["kl_divergence_delta"],
                        "kl_divergence_delta_ci_low": proxy_free_metrics["kl_divergence_delta_ci_low"],
                        "kl_divergence_delta_ci_high": proxy_free_metrics["kl_divergence_delta_ci_high"],
                        "perplexity_shift_delta": proxy_free_metrics["perplexity_shift_delta"],
                        "perplexity_shift_delta_ci_low": proxy_free_metrics["perplexity_shift_delta_ci_low"],
                        "perplexity_shift_delta_ci_high": proxy_free_metrics["perplexity_shift_delta_ci_high"],
                        "top1_change_rate_delta": proxy_free_metrics["top1_change_rate_delta"],
                        "top1_change_rate_delta_ci_low": proxy_free_metrics["top1_change_rate_delta_ci_low"],
                        "top1_change_rate_delta_ci_high": proxy_free_metrics["top1_change_rate_delta_ci_high"],
                    }
                )
    finally:
        if hasattr(backbone, "model"):
            del backbone.model
        sae_cache.clear()
        feature_pool_cache.clear()
        baseline_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(concentration_rows)


def _compress_for_causal_snippet(text: str) -> str:
    normalized = normalize_text(text)
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", normalized) if item.strip()]
    if sentences:
        normalized = sentences[0]
    if len(normalized) > 240:
        normalized = normalized[:237].rstrip() + "..."
    return normalized


def _evaluate_cached_proxy_free_effects(
    rows: pd.DataFrame,
    baseline_cache: dict[str, object],
    model_bundle: object,
    layer: int,
    sae: object,
    target_features: list[int],
    random_feature_sets: list[list[int]],
    config: ExperimentConfig,
) -> dict[str, float]:
    kl_deltas: list[float] = []
    perplexity_deltas: list[float] = []
    top1_deltas: list[float] = []

    for record in rows.itertuples(index=False):
        cache = baseline_cache.get(str(record.segment_id))
        if cache is None:
            continue
        target_metrics = _compute_sequence_change_metrics(
            model_bundle=model_bundle,
            baseline_cache=cache,
            editor=lambda hidden, features=target_features: sparse_ablation_editor(hidden, sae, features),
            layer=layer,
        )
        random_metrics = [
            _compute_sequence_change_metrics(
                model_bundle=model_bundle,
                baseline_cache=cache,
                editor=lambda hidden, features=random_features: sparse_ablation_editor(hidden, sae, features),
                layer=layer,
            )
            for random_features in random_feature_sets
        ]
        random_means = _mean_metric_dicts(random_metrics)
        kl_deltas.append(target_metrics["kl_divergence"] - random_means["kl_divergence"])
        perplexity_deltas.append(target_metrics["perplexity_shift"] - random_means["perplexity_shift"])
        top1_deltas.append(target_metrics["top1_change_rate"] - random_means["top1_change_rate"])

    kl_ci = bootstrap_ci(kl_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    perplexity_ci = bootstrap_ci(perplexity_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    top1_ci = bootstrap_ci(top1_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    return {
        "kl_divergence_delta": float(np.mean(kl_deltas)) if kl_deltas else np.nan,
        "kl_divergence_delta_ci_low": kl_ci[0],
        "kl_divergence_delta_ci_high": kl_ci[1],
        "perplexity_shift_delta": float(np.mean(perplexity_deltas)) if perplexity_deltas else np.nan,
        "perplexity_shift_delta_ci_low": perplexity_ci[0],
        "perplexity_shift_delta_ci_high": perplexity_ci[1],
        "top1_change_rate_delta": float(np.mean(top1_deltas)) if top1_deltas else np.nan,
        "top1_change_rate_delta_ci_low": top1_ci[0],
        "top1_change_rate_delta_ci_high": top1_ci[1],
    }


def _export_audit_discriminativeness_figure(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    retrieval: pd.DataFrame,
    target: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_frame = pairwise.copy()
    plot_frame["pair_type"] = plot_frame["same_family"].map({True: "within family", False: "between family"})
    sns.barplot(
        data=plot_frame,
        x="view_name",
        y="cosine_similarity",
        hue="pair_type",
        estimator=np.mean,
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_title("Similarity gap by audit view")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean cosine similarity")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=summary, x="view_name", y="retrieval_accuracy", color="#4C72B0", ax=axes[1])
    axes[1].set_title("Nearest-family retrieval accuracy")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _export_audit_robustness_figure(summary: pd.DataFrame, target: Path) -> None:
    if summary.empty:
        return
    plot = summary.set_index("perturbation")[
        [
            "mean_policy_specific_jaccard",
            "mean_overall_jaccard",
            "mean_proxy_rank_spearman",
            "mean_layer_profile_cosine",
        ]
    ]
    plt.figure(figsize=(7, max(4, 0.8 * len(plot))))
    sns.heatmap(plot, cmap="crest", annot=True, fmt=".2f", linewidths=0.2)
    plt.title("Audit report robustness under perturbation")
    plt.tight_layout()
    plt.savefig(target, dpi=180, bbox_inches="tight")
    plt.close()


def _export_failure_transparency_figure(summary: pd.DataFrame, target: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(8, max(3.5, 0.8 * len(summary))))
    sns.barplot(data=summary, y="metric_name", x="metric_value", color="#C44E52")
    plt.title("Failure transparency summary")
    plt.xlabel("Rate or mean gap")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(target, dpi=180, bbox_inches="tight")
    plt.close()


def _export_audit_causal_concentration_figure(summary: pd.DataFrame, target: Path) -> None:
    if summary.empty:
        return
    plot = summary.copy()
    plot["set_label"] = plot["set_size"].map({1: "single", 3: "top 3", 5: "top 5"}).fillna(plot["set_size"].astype(str))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    metric_specs = [
        ("kl_divergence_delta", "KL delta"),
        ("perplexity_shift_delta", "Perplexity delta"),
        ("top1_change_rate_delta", "Top-1 change delta"),
    ]
    for ax, (metric_name, title) in zip(axes, metric_specs):
        sns.lineplot(
            data=plot,
            x="layer",
            y=metric_name,
            hue="set_label",
            marker="o",
            linewidth=2.0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("")
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Feature set size")
    for ax in axes[:-1]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    plt.tight_layout()
    plt.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
