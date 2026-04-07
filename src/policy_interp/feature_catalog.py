"""Feature first catalog construction, overlays, and feature labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score, roc_auc_score

from policy_interp.adapters.modeling import HuggingFaceBackboneAdapter, SaeLensAdapter
from policy_interp.extract import (
    _build_token_budget_batches,
    _decode_token_span,
    _heaps_to_records,
    _pool_sequence,
    _update_context_heap,
)
from policy_interp.io import read_jsonl, read_parquet, write_jsonl, write_parquet, write_safetensors
from policy_interp.labeling import (
    _looks_like_label_name,
    _looks_like_rationale,
    _maybe_load_generator,
    _normalize_generation_lines,
    _normalize_sentence_text,
)
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir, set_seed


CATALOG_ROOT = "features"
BACKFILL_ROOT = "backfill"


@dataclass(slots=True)
class FeatureCatalogArtifacts:
    summary_path: str
    top_features_path: str
    contexts_path: str
    catalog_path: str
    segment_scores_path: str
    decoder_vectors_path: str
    logit_attribution_path: str
    exemplars_path: str
    overlap_path: str
    decoder_similarity_path: str
    lineage_path: str
    proxy_overlay_path: str


@dataclass(slots=True)
class FeatureLabelArtifacts:
    labels_path: str


def build_feature_catalog(config: ExperimentConfig) -> FeatureCatalogArtifacts:
    set_seed(config.splits.seed)
    root = ensure_dir(config.run_root / CATALOG_ROOT)
    segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    matches_path = config.run_root / "matching" / "matched_negatives.parquet"
    matches = read_parquet(matches_path) if matches_path.exists() else pd.DataFrame()
    interventions = _load_feature_causal_summary(config)

    backbone = HuggingFaceBackboneAdapter(config.backbone).load()
    sae_loader = SaeLensAdapter(config.sae)
    try:
        summary_frame, top_feature_frame, context_frame = _build_combined_feature_tables(config, backbone)
        summary_path = write_parquet(summary_frame, root / "feature_summary.parquet")
        top_features_path = write_parquet(top_feature_frame, root / "segment_top_features.parquet")
        contexts_path = root / "feature_top_contexts.parquet"

        preliminary = _build_preliminary_catalog(summary_frame, config)
        candidate_vectors = _load_decoder_vectors_for_candidates(preliminary, sae_loader)
        unique_scores = _compute_layer_unique_scores(summary_frame, candidate_vectors, config)
        summary_frame = summary_frame.merge(
            unique_scores,
            on=["model_id", "sae_release", "layer", "feature_id"],
            how="left",
            suffixes=("", "_computed"),
        )
        if "max_cross_layer_decoder_cosine_computed" in summary_frame.columns:
            summary_frame["max_cross_layer_decoder_cosine"] = summary_frame[
                "max_cross_layer_decoder_cosine_computed"
            ].fillna(summary_frame.get("max_cross_layer_decoder_cosine", 0.0))
            summary_frame = summary_frame.drop(columns=["max_cross_layer_decoder_cosine_computed"])
        summary_frame["max_cross_layer_decoder_cosine"] = summary_frame["max_cross_layer_decoder_cosine"].fillna(0.0)
        summary_frame["layer_unique_score"] = (
            summary_frame["policy_specific_score"] * (1.0 - summary_frame["max_cross_layer_decoder_cosine"])
        )
        summary_path = write_parquet(summary_frame, root / "feature_summary.parquet")

        catalog_frame = _build_final_catalog(summary_frame, config)
        final_vectors, decoder_manifest = _load_final_catalog_decoder_vectors(catalog_frame, sae_loader)
        if config.feature_catalog.store_decoder_vectors:
            write_safetensors(final_vectors, root / "feature_catalog_decoder_vectors.safetensors")

        backfill_root = ensure_dir(root / BACKFILL_ROOT)
        backfill_candidates = _select_evidence_backfill_candidates(
            catalog_frame=catalog_frame,
            top_feature_frame=top_feature_frame,
            context_frame=context_frame,
            config=config,
        )
        write_parquet(backfill_candidates, backfill_root / "selected_feature_candidates.parquet")
        backfilled_segment_scores = pd.DataFrame()
        backfilled_contexts = pd.DataFrame()
        if config.feature_catalog.evidence_backfill_enabled and not backfill_candidates.empty:
            backfilled_segment_scores, backfilled_contexts = _backfill_selected_feature_evidence(
                candidates=backfill_candidates,
                segments=segments,
                backbone=backbone,
                sae_loader=sae_loader,
                config=config,
            )
        write_parquet(backfilled_segment_scores, backfill_root / "selected_feature_segment_scores.parquet")
        write_parquet(backfilled_contexts, backfill_root / "selected_feature_contexts.parquet")
        context_frame = _merge_backfilled_contexts(context_frame, backfilled_contexts)
        contexts_path = write_parquet(context_frame, root / "feature_top_contexts.parquet")

        segment_scores = _build_catalog_segment_scores(
            catalog_frame=catalog_frame,
            top_feature_frame=top_feature_frame,
            segments=segments,
            config=config,
            backfilled_segment_scores=backfilled_segment_scores,
        )
        if config.feature_catalog.store_catalog_segment_scores:
            segment_scores_path = write_parquet(segment_scores, root / "feature_catalog_segment_scores.parquet")
        else:
            segment_scores_path = root / "feature_catalog_segment_scores.parquet"

        overlay_frame = _build_feature_proxy_overlay(catalog_frame, segment_scores, segments, matches, config)
        overlay_path = write_parquet(overlay_frame, root / "feature_proxy_overlay.parquet")
        logit_frame = _build_logit_attribution_table(catalog_frame, final_vectors, decoder_manifest, backbone, config)
        logit_attribution_path = write_parquet(logit_frame, root / "feature_catalog_logit_attribution.parquet")
        exemplars_frame = _build_feature_exemplars(
            catalog_frame=catalog_frame,
            segment_scores=segment_scores,
            top_feature_frame=top_feature_frame,
            segments=segments,
            overlay_frame=overlay_frame,
            matches=matches,
            config=config,
        )
        exemplars_path = write_parquet(exemplars_frame, root / "feature_catalog_exemplars.parquet")

        overlap_frame, decoder_similarity_frame = _build_cross_layer_overlap_tables(
            catalog_frame=catalog_frame,
            decoder_manifest=decoder_manifest,
            final_vectors=final_vectors,
            config=config,
        )
        overlap_path = write_parquet(overlap_frame, root / "cross_layer_feature_overlap.parquet")
        decoder_similarity_path = write_parquet(decoder_similarity_frame, root / "cross_layer_decoder_similarity.parquet")
        lineage_frame = _build_feature_lineage_candidates(
            catalog_frame=catalog_frame,
            sae_loader=sae_loader,
            decoder_manifest=decoder_manifest,
            final_vectors=final_vectors,
            config=config,
        )
        lineage_path = write_parquet(lineage_frame, root / "feature_lineage_candidates.parquet")

        catalog_frame = _enrich_catalog_with_overlays_and_causality(
            catalog_frame=catalog_frame,
            overlay_frame=overlay_frame,
            interventions=interventions,
        )
        catalog_path = write_parquet(catalog_frame, root / "feature_catalog.parquet")
    finally:
        if hasattr(backbone, "model"):
            del backbone.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return FeatureCatalogArtifacts(
        summary_path=str(summary_path),
        top_features_path=str(top_features_path),
        contexts_path=str(contexts_path),
        catalog_path=str(catalog_path),
        segment_scores_path=str(segment_scores_path),
        decoder_vectors_path=str(root / "feature_catalog_decoder_vectors.safetensors"),
        logit_attribution_path=str(logit_attribution_path),
        exemplars_path=str(exemplars_path),
        overlap_path=str(overlap_path),
        decoder_similarity_path=str(decoder_similarity_path),
        lineage_path=str(lineage_path),
        proxy_overlay_path=str(overlay_path),
    )


def run_feature_labeling(config: ExperimentConfig) -> FeatureLabelArtifacts:
    root = config.run_root / CATALOG_ROOT
    catalog = read_parquet(root / "feature_catalog.parquet")
    contexts = read_parquet(root / "feature_top_contexts.parquet")
    overlay = read_parquet(root / "feature_proxy_overlay.parquet")
    exemplars = read_parquet(root / "feature_catalog_exemplars.parquet")
    logit_table = read_parquet(root / "feature_catalog_logit_attribution.parquet")
    generator = _maybe_load_generator(config)
    rows: list[dict[str, object]] = []

    for feature in catalog.itertuples(index=False):
        feature_contexts = contexts.loc[
            (contexts["layer"] == int(feature.layer)) & (contexts["feature_id"] == int(feature.feature_id))
        ].sort_values("rank")
        feature_exemplars = exemplars.loc[
            (exemplars["layer"] == int(feature.layer)) & (exemplars["feature_id"] == int(feature.feature_id))
        ].copy()
        positive_exemplars = feature_exemplars.loc[feature_exemplars["example_kind"] == "positive"].head(
            config.feature_catalog.top_exemplars_per_feature
        )
        negative_exemplars = feature_exemplars.loc[feature_exemplars["example_kind"] == "negative"].head(
            config.labeling.num_negative_controls
        )
        overlay_rows = overlay.loc[
            (overlay["layer"] == int(feature.layer)) & (overlay["feature_id"] == int(feature.feature_id))
        ].sort_values("test_auc", ascending=False)
        best_proxy = overlay_rows.iloc[0]["proxy"] if not overlay_rows.empty else "unknown"
        logit_row = logit_table.loc[
            (logit_table["layer"] == int(feature.layer)) & (logit_table["feature_id"] == int(feature.feature_id))
        ]
        top_positive_tokens = logit_row.iloc[0]["top_positive_tokens"] if not logit_row.empty else []
        template_summary = _feature_template_summary(
            feature=feature,
            best_proxy=str(best_proxy),
            contexts=feature_contexts,
            positive_exemplars=positive_exemplars,
            negative_exemplars=negative_exemplars,
            top_positive_tokens=top_positive_tokens,
        )
        generated = _generate_feature_name_and_rationale(
            generator=generator,
            config=config,
            feature=feature,
            best_proxy=str(best_proxy),
            contexts=feature_contexts,
            positive_exemplars=positive_exemplars,
            negative_exemplars=negative_exemplars,
            top_positive_tokens=top_positive_tokens,
        )
        semantic_tag = _infer_feature_semantic_tag(feature_contexts, top_positive_tokens)
        rows.append(
            {
                "model_id": feature.model_id,
                "sae_release": feature.sae_release,
                "layer": int(feature.layer),
                "feature_id": int(feature.feature_id),
                "ranking_family": feature.ranking_family,
                "rank": int(feature.rank),
                "best_proxy": best_proxy,
                "template_summary": template_summary,
                "generated_name": generated["name"],
                "generated_rationale": generated["rationale"],
                "semantic_tag": semantic_tag,
            }
        )

    labels_path = root / "feature_labels.jsonl"
    write_jsonl(rows, labels_path)
    return FeatureLabelArtifacts(str(labels_path))


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def _load_feature_causal_summary(config: ExperimentConfig) -> pd.DataFrame:
    feature_first_path = config.run_root / "interventions" / "feature_causal_summary.parquet"
    legacy_path = config.run_root / "interventions" / "ablation_sparse_vs_dense.parquet"
    if feature_first_path.exists():
        return read_parquet(feature_first_path)
    if legacy_path.exists():
        return read_parquet(legacy_path)
    return pd.DataFrame()


def _build_combined_feature_tables(
    config: ExperimentConfig,
    backbone: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    top_feature_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    model_depth = int(backbone.model_depth)
    model_id = str(backbone.model_id)

    for layer in config.extract.layers:
        raw_summary = read_parquet(config.run_root / "extraction" / f"feature_summary_layer_{layer}.parquet")
        top_features = read_parquet(config.run_root / "extraction" / f"segment_top_features_layer_{layer}.parquet")
        contexts = read_jsonl(config.run_root / "extraction" / f"feature_top_contexts_layer_{layer}.jsonl")

        enriched_summary = _ensure_feature_summary_schema(
            raw_summary=raw_summary,
            top_features=top_features,
            model_id=model_id,
            sae_release=config.sae.release,
            model_depth=model_depth,
            layer=int(layer),
        )
        summary_frames.append(enriched_summary)
        top_feature_frames.append(
            _ensure_top_feature_schema(
                top_features=top_features,
                model_id=model_id,
                sae_release=config.sae.release,
                model_depth=model_depth,
                layer=int(layer),
            )
        )
        context_frames.append(
            _ensure_context_schema(
                contexts=contexts,
                model_id=model_id,
                sae_release=config.sae.release,
                model_depth=model_depth,
                layer=int(layer),
            )
        )

    summary_frame = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    top_feature_frame = pd.concat(top_feature_frames, ignore_index=True) if top_feature_frames else pd.DataFrame()
    context_frame = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame()
    return summary_frame, top_feature_frame, context_frame


def _ensure_feature_summary_schema(
    raw_summary: pd.DataFrame,
    top_features: pd.DataFrame,
    model_id: str,
    sae_release: str,
    model_depth: int,
    layer: int,
) -> pd.DataFrame:
    frame = raw_summary.copy()
    layer_fraction = _layer_depth_fraction(layer, model_depth)
    layer_stage = _layer_stage(layer_fraction)

    if "model_id" not in frame.columns:
        frame["model_id"] = model_id
    if "sae_release" not in frame.columns:
        frame["sae_release"] = sae_release
    if "model_depth" not in frame.columns:
        frame["model_depth"] = model_depth
    if "layer_depth_fraction" not in frame.columns:
        frame["layer_depth_fraction"] = layer_fraction
    if "layer_stage" not in frame.columns:
        frame["layer_stage"] = layer_stage
    if "mean_all_activation" not in frame.columns:
        if "mean_magnitude" in frame.columns:
            frame["mean_all_activation"] = frame["mean_magnitude"].astype(float)
        else:
            frame["mean_all_activation"] = 0.0
    if "mean_magnitude" not in frame.columns:
        frame["mean_magnitude"] = frame["mean_all_activation"].astype(float)

    activation_count = frame["activation_count"].astype(float).to_numpy()
    activation_frequency = frame["activation_frequency"].astype(float).to_numpy()
    mean_all = frame["mean_all_activation"].astype(float).to_numpy()
    total_segments = int(round(np.nanmax(np.divide(activation_count, np.maximum(activation_frequency, 1e-12), where=activation_frequency > 0)))) if np.any(activation_frequency > 0) else 0
    if total_segments <= 0:
        total_segments = max(int(raw_summary["activation_count"].max()) if "activation_count" in raw_summary.columns else 0, 1)

    if "mean_positive_activation" not in frame.columns:
        numerator = mean_all * total_segments
        frame["mean_positive_activation"] = np.divide(
            numerator,
            np.maximum(activation_count, 1.0),
            out=np.zeros_like(numerator, dtype=np.float64),
            where=activation_count > 0,
        )
    grouped = _group_feature_sparse_stats(top_features)
    if "max_activation" not in frame.columns:
        frame["max_activation"] = frame["feature_id"].map(grouped["max_activation"]).fillna(0.0)
    if "top20_mean_activation" not in frame.columns:
        frame["top20_mean_activation"] = frame["feature_id"].map(grouped["top20_mean_activation"]).fillna(frame["mean_positive_activation"])
    if "top100_mean_activation" not in frame.columns:
        frame["top100_mean_activation"] = frame["feature_id"].map(grouped["top100_mean_activation"]).fillna(frame["mean_positive_activation"])
    if "activation_gini" not in frame.columns:
        frame["activation_gini"] = frame["feature_id"].map(grouped["activation_gini"]).fillna(0.0)
    if "document_frequency" not in frame.columns:
        frame["document_frequency"] = frame["feature_id"].map(grouped["document_frequency"]).fillna(0).astype(int)
    if "mean_token_peak" not in frame.columns:
        frame["mean_token_peak"] = frame["feature_id"].map(grouped["mean_token_peak"]).fillna(frame["mean_positive_activation"])

    frame["global_dominance_score"] = frame["activation_frequency"].astype(float) * frame["mean_positive_activation"].astype(float)
    frame["policy_specific_score"] = frame["top20_mean_activation"].astype(float) * (1.0 - frame["activation_frequency"].astype(float))
    if "layer_unique_score" not in frame.columns:
        frame["layer_unique_score"] = 0.0
    if "max_cross_layer_decoder_cosine" not in frame.columns:
        frame["max_cross_layer_decoder_cosine"] = 0.0
    return frame


def _group_feature_sparse_stats(top_features: pd.DataFrame) -> dict[str, pd.Series]:
    if top_features.empty:
        empty = pd.Series(dtype=float)
        return {
            "max_activation": empty,
            "top20_mean_activation": empty,
            "top100_mean_activation": empty,
            "activation_gini": empty,
            "document_frequency": pd.Series(dtype=int),
            "mean_token_peak": empty,
        }
    stats = top_features.copy()
    stats["peak_token_value"] = stats["token_values"].apply(_first_numeric_or_default)
    grouped = stats.groupby("feature_id")
    max_activation = grouped["pooled_activation"].max()
    top20 = grouped["pooled_activation"].apply(lambda col: float(np.mean(sorted(col.tolist(), reverse=True)[:20])))
    top100 = grouped["pooled_activation"].apply(lambda col: float(np.mean(sorted(col.tolist(), reverse=True)[:100])))
    document_frequency = grouped["document_id"].nunique()
    mean_token_peak = grouped["peak_token_value"].mean()
    activation_gini = grouped["pooled_activation"].apply(_approx_sparse_gini)
    return {
        "max_activation": max_activation,
        "top20_mean_activation": top20,
        "top100_mean_activation": top100,
        "activation_gini": activation_gini,
        "document_frequency": document_frequency,
        "mean_token_peak": mean_token_peak,
    }


def _approx_sparse_gini(values: pd.Series) -> float:
    array = np.asarray(values.tolist(), dtype=np.float64)
    if array.size == 0:
        return 0.0
    if np.allclose(array, 0.0):
        return 0.0
    ordered = np.sort(array)
    n = ordered.size
    index = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2.0 * index - n - 1.0) * ordered)
    denominator = n * np.sum(ordered)
    if denominator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _ensure_top_feature_schema(
    top_features: pd.DataFrame,
    model_id: str,
    sae_release: str,
    model_depth: int,
    layer: int,
) -> pd.DataFrame:
    frame = top_features.copy()
    layer_fraction = _layer_depth_fraction(layer, model_depth)
    layer_stage = _layer_stage(layer_fraction)
    for column, value in {
        "model_id": model_id,
        "sae_release": sae_release,
        "model_depth": model_depth,
        "layer_depth_fraction": layer_fraction,
        "layer_stage": layer_stage,
        "layer": layer,
    }.items():
        if column not in frame.columns:
            frame[column] = value
    if "top_token_span_text" not in frame.columns:
        frame["top_token_span_text"] = ""
    if "top_token_span_start" not in frame.columns:
        frame["top_token_span_start"] = None
    if "top_token_span_end" not in frame.columns:
        frame["top_token_span_end"] = None
    return frame


def _ensure_context_schema(
    contexts: list[dict[str, object]],
    model_id: str,
    sae_release: str,
    model_depth: int,
    layer: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(contexts)
    layer_fraction = _layer_depth_fraction(layer, model_depth)
    layer_stage = _layer_stage(layer_fraction)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "sae_release",
                "model_depth",
                "layer_depth_fraction",
                "layer_stage",
                "layer",
                "feature_id",
                "rank",
                "segment_id",
                "document_id",
                "split",
                "activation",
                "context_text",
                "top_token_span_text",
                "left_context",
                "right_context",
            ]
        )
    for column, value in {
        "model_id": model_id,
        "sae_release": sae_release,
        "model_depth": model_depth,
        "layer_depth_fraction": layer_fraction,
        "layer_stage": layer_stage,
        "layer": layer,
    }.items():
        if column not in frame.columns:
            frame[column] = value
    if "top_token_span_text" not in frame.columns:
        frame["top_token_span_text"] = ""
    if "left_context" not in frame.columns:
        frame["left_context"] = ""
    if "right_context" not in frame.columns:
        frame["right_context"] = ""
    return frame


def _build_preliminary_catalog(summary_frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidate_pool = max(config.feature_catalog.top_n_per_family, config.feature_catalog.layer_unique_candidate_pool)
    for layer in sorted(summary_frame["layer"].dropna().astype(int).unique().tolist()):
        layer_frame = summary_frame.loc[summary_frame["layer"] == layer].copy()
        global_top = layer_frame.sort_values("global_dominance_score", ascending=False).head(candidate_pool).copy()
        global_top["ranking_family"] = "global_dominance"
        global_top["rank"] = np.arange(1, len(global_top) + 1)
        policy_top = layer_frame.sort_values("policy_specific_score", ascending=False).head(candidate_pool).copy()
        policy_top["ranking_family"] = "policy_specific"
        policy_top["rank"] = np.arange(1, len(policy_top) + 1)
        rows.extend([global_top, policy_top])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_decoder_vectors_for_candidates(preliminary: pd.DataFrame, sae_loader: SaeLensAdapter) -> dict[int, dict[int, np.ndarray]]:
    vectors: dict[int, dict[int, np.ndarray]] = {}
    for layer in sorted(preliminary["layer"].dropna().astype(int).unique().tolist()):
        feature_ids = sorted(set(preliminary.loc[preliminary["layer"] == layer, "feature_id"].astype(int).tolist()))
        if not feature_ids:
            continue
        sae = sae_loader.load_for_layer(layer)
        weight = sae.W_dec.detach().cpu().to(torch.float32).numpy()
        vectors[layer] = {feature_id: weight[feature_id] for feature_id in feature_ids}
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return vectors


def _compute_layer_unique_scores(
    summary_frame: pd.DataFrame,
    candidate_vectors: dict[int, dict[int, np.ndarray]],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    normalized_vectors: dict[int, tuple[list[int], np.ndarray]] = {}
    for layer, mapping in candidate_vectors.items():
        feature_ids = sorted(mapping.keys())
        matrix = np.vstack([mapping[feature_id] for feature_id in feature_ids]).astype(np.float32)
        matrix = _l2_normalize(matrix)
        normalized_vectors[layer] = (feature_ids, matrix)

    for layer, (feature_ids, matrix) in normalized_vectors.items():
        feature_to_max: dict[int, float] = {feature_id: 0.0 for feature_id in feature_ids}
        for other_layer, (_, other_matrix) in normalized_vectors.items():
            if other_layer == layer or other_matrix.size == 0:
                continue
            similarity = matrix @ other_matrix.T
            best = similarity.max(axis=1)
            for feature_id, best_value in zip(feature_ids, best.tolist()):
                feature_to_max[feature_id] = max(feature_to_max[feature_id], float(best_value))
        layer_subset = summary_frame.loc[summary_frame["layer"] == layer]
        model_id = layer_subset["model_id"].iloc[0]
        sae_release = layer_subset["sae_release"].iloc[0]
        for feature_id, best_value in feature_to_max.items():
            rows.append(
                {
                    "model_id": model_id,
                    "sae_release": sae_release,
                    "layer": layer,
                    "feature_id": feature_id,
                    "max_cross_layer_decoder_cosine": best_value,
                }
            )
    return pd.DataFrame(rows)


def _build_final_catalog(summary_frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family in config.feature_catalog.ranking_families:
        score_column = f"{family}_score"
        if score_column not in summary_frame.columns:
            continue
        for layer in sorted(summary_frame["layer"].dropna().astype(int).unique().tolist()):
            layer_frame = summary_frame.loc[summary_frame["layer"] == layer].copy()
            ranked = layer_frame.sort_values(score_column, ascending=False).head(config.feature_catalog.top_n_per_family).copy()
            ranked["ranking_family"] = family
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            ranked["score"] = ranked[score_column].astype(float)
            rows.append(ranked)
    if not rows:
        return pd.DataFrame()
    catalog = pd.concat(rows, ignore_index=True)
    catalog["catalog_key"] = catalog.apply(
        lambda row: f"{row['model_id']}|{row['sae_release']}|layer_{int(row['layer'])}|feature_{int(row['feature_id'])}|{row['ranking_family']}",
        axis=1,
    )
    catalog["decoder_tensor_key"] = catalog.apply(
        lambda row: f"layer_{int(row['layer'])}_feature_{int(row['feature_id'])}",
        axis=1,
    )
    catalog["causal_badge"] = ""
    return catalog


def _load_final_catalog_decoder_vectors(
    catalog_frame: pd.DataFrame,
    sae_loader: SaeLensAdapter,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    vector_map: dict[str, torch.Tensor] = {}
    manifest_rows: list[dict[str, object]] = []
    for layer in sorted(catalog_frame["layer"].dropna().astype(int).unique().tolist()):
        sae = sae_loader.load_for_layer(layer)
        weight = sae.W_dec.detach().cpu().to(torch.float32)
        layer_rows = catalog_frame.loc[catalog_frame["layer"] == layer, ["feature_id", "decoder_tensor_key", "catalog_key"]].drop_duplicates()
        for row in layer_rows.itertuples(index=False):
            tensor_key = str(row.decoder_tensor_key)
            vector_map[tensor_key] = weight[int(row.feature_id)].clone()
            manifest_rows.append(
                {
                    "layer": layer,
                    "feature_id": int(row.feature_id),
                    "decoder_tensor_key": tensor_key,
                    "catalog_key": str(row.catalog_key),
                }
            )
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return vector_map, pd.DataFrame(manifest_rows)


def _select_evidence_backfill_candidates(
    catalog_frame: pd.DataFrame,
    top_feature_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    if not config.feature_catalog.evidence_backfill_enabled or catalog_frame.empty:
        return pd.DataFrame()
    selected = catalog_frame.loc[
        catalog_frame["ranking_family"].isin(config.feature_catalog.evidence_backfill_ranking_families)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = (
        selected.sort_values(["layer", "ranking_family", "rank"])
        .groupby(["layer", "ranking_family"], group_keys=False)
        .head(config.feature_catalog.evidence_backfill_top_n_per_family)
        .copy()
    )
    observed_segment_hits = (
        top_feature_frame.groupby(["layer", "feature_id"]).size().rename("observed_segment_hits")
        if not top_feature_frame.empty
        else pd.Series(dtype=int, name="observed_segment_hits")
    )
    observed_context_hits = (
        context_frame.groupby(["layer", "feature_id"]).size().rename("observed_context_hits")
        if not context_frame.empty
        else pd.Series(dtype=int, name="observed_context_hits")
    )
    selected = selected.merge(
        observed_segment_hits.reset_index(),
        on=["layer", "feature_id"],
        how="left",
    )
    selected = selected.merge(
        observed_context_hits.reset_index(),
        on=["layer", "feature_id"],
        how="left",
    )
    selected["observed_segment_hits"] = selected["observed_segment_hits"].fillna(0).astype(int)
    selected["observed_context_hits"] = selected["observed_context_hits"].fillna(0).astype(int)
    required_positive_examples = max(
        config.feature_catalog.evidence_backfill_min_positive_examples,
        config.autointerp.num_train_positive + config.autointerp.num_holdout_positive if config.autointerp.enabled else 0,
    )
    selected["needs_backfill"] = (
        (selected["observed_segment_hits"] < required_positive_examples)
        | (selected["observed_context_hits"] < config.feature_catalog.top_contexts_per_feature)
    )
    return selected.loc[selected["needs_backfill"]].reset_index(drop=True)


def _backfill_selected_feature_evidence(
    candidates: pd.DataFrame,
    segments: pd.DataFrame,
    backbone: object,
    sae_loader: SaeLensAdapter,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    tokenizer = backbone.tokenizer
    model = backbone.model
    model_depth = int(backbone.model_depth)
    model_id = str(backbone.model_id)
    segment_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []

    for layer in sorted(candidates["layer"].dropna().astype(int).unique().tolist()):
        layer_candidates = candidates.loc[candidates["layer"] == layer].copy()
        feature_ids = sorted(layer_candidates["feature_id"].astype(int).unique().tolist())
        if not feature_ids:
            continue
        sae = sae_loader.load_for_layer(layer)
        feature_index = torch.tensor(feature_ids, device=backbone.device, dtype=torch.long)
        layer_fraction = _layer_depth_fraction(layer, model_depth)
        layer_stage = _layer_stage(layer_fraction)
        layer_context_heaps: dict[int, list[tuple[float, str, dict[str, object]]]] = {}
        for batch_indices in _build_token_budget_batches(segments, tokenizer, config):
            batch = segments.iloc[batch_indices].copy()
            encoded = tokenizer(
                batch["text"].tolist(),
                padding=True,
                truncation=True,
                max_length=config.backbone.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(backbone.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[layer + 1]
                latents = sae.encode(hidden)
                selected_latents = latents.index_select(dim=2, index=feature_index)
                pooled_selected = _pool_sequence(selected_latents, encoded["attention_mask"], config.extract.pooling_method)

            pooled_cpu = pooled_selected.detach().cpu()
            selected_cpu = selected_latents.detach().cpu()
            token_ids = encoded["input_ids"].detach().cpu()
            attention = encoded["attention_mask"].detach().cpu()

            for row_idx, record in enumerate(batch.itertuples(index=False)):
                token_mask = attention[row_idx].bool()
                tokens_for_row = token_ids[row_idx, token_mask]
                latent_row = selected_cpu[row_idx, token_mask, :]
                pooled_row = pooled_cpu[row_idx]
                for local_index, feature_id in enumerate(feature_ids):
                    pooled_value = float(pooled_row[local_index].item())
                    token_values = latent_row[:, local_index]
                    if token_values.numel() == 0:
                        peak_token_value = 0.0
                        peak_token_position = -1
                        token_positions: list[int] = []
                        token_value_list: list[float] = []
                        token_texts: list[str] = []
                        span_text = ""
                    else:
                        local_top_k = min(config.extract.segment_top_token_positions_per_feature, token_values.shape[0])
                        token_top_values, token_top_indices = torch.topk(token_values, k=local_top_k)
                        peak_token_value = float(token_top_values[0].item())
                        if peak_token_value > 0:
                            token_positions = token_top_indices.tolist()
                            token_value_list = [float(value) for value in token_top_values.tolist()]
                            peak_token_position = int(token_positions[0])
                            token_texts = tokenizer.convert_ids_to_tokens(tokens_for_row[token_top_indices].tolist())
                            _, _, span_text = _decode_token_span(
                                tokenizer=tokenizer,
                                tokens=tokens_for_row,
                                token_positions=token_positions,
                                enabled=config.extract.store_token_span_text,
                            )
                        else:
                            token_positions = []
                            token_value_list = []
                            token_texts = []
                            peak_token_position = -1
                            span_text = ""

                    segment_rows.append(
                        {
                            "model_id": model_id,
                            "sae_release": config.sae.release,
                            "model_depth": model_depth,
                            "layer_depth_fraction": layer_fraction,
                            "layer_stage": layer_stage,
                            "segment_id": record.segment_id,
                            "document_id": int(record.document_id),
                            "split": record.split,
                            "layer": layer,
                            "feature_id": int(feature_id),
                            "pooled_activation": pooled_value,
                            "peak_token_value": peak_token_value,
                            "peak_token_position": peak_token_position,
                            "top_token_span_text": span_text,
                        }
                    )
                    if pooled_value > config.extract.catalog_activation_threshold and token_positions:
                        _update_context_heap(
                            context_heaps=layer_context_heaps,
                            model_id=model_id,
                            sae_release=config.sae.release,
                            model_depth=model_depth,
                            layer_fraction=layer_fraction,
                            layer_stage=layer_stage,
                            feature_id=int(feature_id),
                            pooled_value=pooled_value,
                            record=record,
                            token_positions=token_positions,
                            token_texts=token_texts,
                            tokenizer=tokenizer,
                            tokens_for_row=tokens_for_row,
                            window=config.extract.context_window,
                            top_k=config.feature_catalog.top_contexts_per_feature,
                        )

            del outputs, hidden, latents, selected_latents, pooled_selected
            del pooled_cpu, selected_cpu, token_ids, attention, encoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        layer_context_rows = _heaps_to_records(layer_context_heaps)
        for item in layer_context_rows:
            item["layer"] = layer
        context_rows.extend(layer_context_rows)

    segment_score_frame = pd.DataFrame(segment_rows)
    context_frame = pd.DataFrame(context_rows)
    return segment_score_frame, context_frame


def _build_catalog_segment_scores(
    catalog_frame: pd.DataFrame,
    top_feature_frame: pd.DataFrame,
    segments: pd.DataFrame,
    config: ExperimentConfig,
    backfilled_segment_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    segment_base = segments[["segment_id", "document_id", "split"]].drop_duplicates().copy()
    grouped = (
        top_feature_frame.groupby(["layer", "segment_id", "feature_id"])
        .agg(
            pooled_activation=("pooled_activation", "max"),
            peak_token_value=("token_values", lambda values: _first_numeric_or_default(values.iloc[0])),
            peak_token_position=("token_positions", lambda positions: _first_integer_or_default(positions.iloc[0], default=-1)),
            top_token_span_text=("top_token_span_text", "first"),
        )
        .reset_index()
    )
    for layer in sorted(catalog_frame["layer"].dropna().astype(int).unique().tolist()):
        layer_catalog = catalog_frame.loc[catalog_frame["layer"] == layer].copy()
        unique_features = layer_catalog[["feature_id"]].drop_duplicates().copy()
        unique_features["__join_key"] = 1
        layer_segments = segment_base.copy()
        layer_segments["__join_key"] = 1
        layer_pairs = layer_segments.merge(unique_features, on="__join_key", how="inner").drop(columns="__join_key")
        observed = grouped.loc[grouped["layer"] == layer].copy()
        merged = layer_pairs.merge(observed, on=["segment_id", "feature_id"], how="left")
        merged["pooled_activation"] = merged["pooled_activation"].fillna(0.0)
        merged["peak_token_value"] = merged["peak_token_value"].fillna(0.0)
        merged["peak_token_position"] = merged["peak_token_position"].fillna(-1).astype(int)
        merged["top_token_span_text"] = merged["top_token_span_text"].fillna("")
        merged["layer"] = layer
        if backfilled_segment_scores is not None and not backfilled_segment_scores.empty:
            layer_backfill = backfilled_segment_scores.loc[
                backfilled_segment_scores["layer"] == layer,
                ["segment_id", "feature_id", "pooled_activation", "peak_token_value", "peak_token_position", "top_token_span_text"],
            ].copy()
            if not layer_backfill.empty:
                merged = merged.merge(
                    layer_backfill,
                    on=["segment_id", "feature_id"],
                    how="left",
                    suffixes=("", "_backfill"),
                )
                for column in ["pooled_activation", "peak_token_value", "peak_token_position"]:
                    merged[column] = merged[f"{column}_backfill"].where(
                        merged[f"{column}_backfill"].notna(),
                        merged[column],
                    )
                merged["top_token_span_text"] = np.where(
                    merged["top_token_span_text_backfill"].notna() & (merged["top_token_span_text_backfill"] != ""),
                    merged["top_token_span_text_backfill"],
                    merged["top_token_span_text"],
                )
                merged = merged.drop(
                    columns=[
                        "pooled_activation_backfill",
                        "peak_token_value_backfill",
                        "peak_token_position_backfill",
                        "top_token_span_text_backfill",
                    ]
                )
        layer_metadata = layer_catalog[
            ["feature_id", "model_id", "sae_release", "model_depth", "layer_depth_fraction", "layer_stage"]
        ].drop_duplicates()
        merged = merged.merge(layer_metadata, on="feature_id", how="left")
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_feature_proxy_overlay(
    catalog_frame: pd.DataFrame,
    segment_scores: pd.DataFrame,
    segments: pd.DataFrame,
    matches: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segment_lookup = segments.set_index("segment_id")
    for feature in catalog_frame[["model_id", "sae_release", "layer", "feature_id"]].drop_duplicates().itertuples(index=False):
        score_frame = segment_scores.loc[
            (segment_scores["layer"] == int(feature.layer)) & (segment_scores["feature_id"] == int(feature.feature_id))
        ].copy()
        if score_frame.empty:
            continue
        aligned = score_frame.merge(
            segments[["segment_id", "split", *config.dataset.proxy_columns.keys()]],
            on=["segment_id", "split"],
            how="left",
        )
        for proxy in config.dataset.proxy_columns.keys():
            values = aligned["pooled_activation"].to_numpy(dtype=float)
            labels = aligned[proxy].fillna(False).astype(int).to_numpy()
            split_values = aligned["split"].astype(str).tolist()
            train_auc = _safe_auc(values, labels, split_values, "train")
            dev_auc = _safe_auc(values, labels, split_values, "dev")
            test_auc = _safe_auc(values, labels, split_values, "test")
            validated_auc = _safe_auc(values, labels, split_values, "validated")
            positive_mean = float(aligned.loc[aligned[proxy].fillna(False), "pooled_activation"].mean()) if np.any(labels == 1) else 0.0
            negative_mean = float(aligned.loc[~aligned[proxy].fillna(False), "pooled_activation"].mean()) if np.any(labels == 0) else 0.0
            overall_mean = float(aligned["pooled_activation"].mean()) if not aligned.empty else 0.0
            matched_contrast = _matched_negative_contrast_score(
                proxy=proxy,
                layer=int(feature.layer),
                feature_id=int(feature.feature_id),
                segment_scores=segment_scores,
                matches=matches,
            )
            rows.append(
                {
                    "model_id": feature.model_id,
                    "sae_release": feature.sae_release,
                    "layer": int(feature.layer),
                    "feature_id": int(feature.feature_id),
                    "proxy": proxy,
                    "train_auc": train_auc,
                    "dev_auc": dev_auc,
                    "test_auc": test_auc,
                    "validated_auc": validated_auc,
                    "mutual_information": _feature_mutual_information(values, labels),
                    "positive_class_lift": (positive_mean / max(overall_mean, 1e-8)) if overall_mean > 0 else 0.0,
                    "positive_negative_gap": positive_mean - negative_mean,
                    "matched_negative_contrast_score": matched_contrast,
                }
            )
    return pd.DataFrame(rows)


def _merge_backfilled_contexts(base_context_frame: pd.DataFrame, backfilled_contexts: pd.DataFrame) -> pd.DataFrame:
    if backfilled_contexts.empty:
        return base_context_frame
    if base_context_frame.empty:
        return backfilled_contexts.sort_values(["layer", "feature_id", "rank"]).reset_index(drop=True)
    replacement_keys = backfilled_contexts[["layer", "feature_id"]].drop_duplicates()
    filtered = base_context_frame.merge(
        replacement_keys.assign(_replace_key=1),
        on=["layer", "feature_id"],
        how="left",
    )
    filtered = filtered.loc[filtered["_replace_key"].isna()].drop(columns="_replace_key")
    combined = pd.concat([filtered, backfilled_contexts], ignore_index=True, sort=False)
    return combined.sort_values(["layer", "feature_id", "rank"]).reset_index(drop=True)


def _safe_auc(values: np.ndarray, labels: np.ndarray, splits: list[str], split_name: str) -> float:
    split_mask = np.asarray([item == split_name for item in splits], dtype=bool)
    if split_mask.sum() == 0 or np.unique(labels[split_mask]).size < 2:
        return np.nan
    return float(roc_auc_score(labels[split_mask], values[split_mask]))


def _feature_mutual_information(values: np.ndarray, labels: np.ndarray) -> float:
    active = (values > 0).astype(int)
    if np.unique(labels).size < 2:
        return np.nan
    return float(mutual_info_score(labels, active))


def _matched_negative_contrast_score(
    proxy: str,
    layer: int,
    feature_id: int,
    segment_scores: pd.DataFrame,
    matches: pd.DataFrame,
) -> float:
    if matches.empty:
        return np.nan
    feature_scores = segment_scores.loc[
        (segment_scores["layer"] == layer) & (segment_scores["feature_id"] == feature_id),
        ["segment_id", "pooled_activation"],
    ].copy()
    score_lookup = dict(zip(feature_scores["segment_id"].astype(str), feature_scores["pooled_activation"].astype(float)))
    match_rows = matches.loc[matches["proxy"] == proxy].copy()
    if match_rows.empty:
        return np.nan
    deltas: list[float] = []
    for row in match_rows.itertuples(index=False):
        positive_value = score_lookup.get(str(row.positive_segment_id), 0.0)
        negative_value = score_lookup.get(str(row.negative_segment_id), 0.0)
        deltas.append(float(positive_value - negative_value))
    if not deltas:
        return np.nan
    return float(np.mean(deltas))


def _build_logit_attribution_table(
    catalog_frame: pd.DataFrame,
    final_vectors: dict[str, torch.Tensor],
    decoder_manifest: pd.DataFrame,
    backbone: object,
    config: ExperimentConfig,
) -> pd.DataFrame:
    if not config.feature_catalog.store_logit_attribution or decoder_manifest.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    tokenizer = backbone.tokenizer
    lm_head = backbone.model.lm_head
    norm = backbone.model.model.norm if hasattr(backbone.model, "model") and hasattr(backbone.model.model, "norm") else None
    device = backbone.device
    weight_dtype = lm_head.weight.dtype
    metadata = catalog_frame[
        ["catalog_key", "model_id", "sae_release", "layer", "feature_id", "ranking_family", "rank"]
    ].drop_duplicates()

    for item in decoder_manifest.itertuples(index=False):
        vector = final_vectors[str(item.decoder_tensor_key)].to(device=device, dtype=weight_dtype)
        with torch.inference_mode():
            normalized = norm(vector.unsqueeze(0)) if norm is not None else vector.unsqueeze(0)
            logits = F.linear(normalized, lm_head.weight).squeeze(0).to(torch.float32)
            top_positive_values, top_positive_indices = torch.topk(logits, k=config.feature_catalog.logit_top_k)
            top_negative_values, top_negative_indices = torch.topk(-logits, k=config.feature_catalog.logit_top_k)
        meta = metadata.loc[metadata["catalog_key"] == item.catalog_key].iloc[0]
        rows.append(
            {
                "catalog_key": item.catalog_key,
                "model_id": meta["model_id"],
                "sae_release": meta["sae_release"],
                "layer": int(meta["layer"]),
                "feature_id": int(meta["feature_id"]),
                "ranking_family": meta["ranking_family"],
                "rank": int(meta["rank"]),
                "top_positive_tokens": tokenizer.convert_ids_to_tokens(top_positive_indices.detach().cpu().tolist()),
                "top_positive_scores": [float(value) for value in top_positive_values.detach().cpu().tolist()],
                "top_negative_tokens": tokenizer.convert_ids_to_tokens(top_negative_indices.detach().cpu().tolist()),
                "top_negative_scores": [float(-value) for value in top_negative_values.detach().cpu().tolist()],
            }
        )
    return pd.DataFrame(rows)


def _build_feature_exemplars(
    catalog_frame: pd.DataFrame,
    segment_scores: pd.DataFrame,
    top_feature_frame: pd.DataFrame,
    segments: pd.DataFrame,
    overlay_frame: pd.DataFrame,
    matches: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segment_text = segments[["segment_id", "document_id", "text", "authority", "jurisdiction", "document_form", "collection_list"]].copy()
    del top_feature_frame
    for feature in catalog_frame.itertuples(index=False):
        feature_scores = segment_scores.loc[
            (segment_scores["layer"] == int(feature.layer)) & (segment_scores["feature_id"] == int(feature.feature_id))
        ].copy()
        if feature_scores.empty:
            continue
        feature_scores = feature_scores.merge(segment_text, on=["segment_id", "document_id"], how="left")
        positives = feature_scores.sort_values("pooled_activation", ascending=False).head(config.feature_catalog.top_exemplars_per_feature)
        best_proxy = _best_proxy_for_feature(
            overlay_frame,
            layer=int(feature.layer),
            feature_id=int(feature.feature_id),
        )
        negatives = _negative_exemplars_for_feature(
            positives=positives,
            all_scores=feature_scores,
            matches=matches,
            proxy=best_proxy,
            feature_layer=int(feature.layer),
            feature_id=int(feature.feature_id),
            top_count=config.labeling.num_negative_controls,
        )
        rows.extend(_exemplar_rows_from_frame(feature, positives, "positive"))
        rows.extend(_exemplar_rows_from_frame(feature, negatives, "negative"))
    return pd.DataFrame(rows)


def _best_proxy_for_feature(overlay_frame: pd.DataFrame, layer: int, feature_id: int) -> str:
    rows = overlay_frame.loc[
        (overlay_frame["layer"] == layer) & (overlay_frame["feature_id"] == feature_id)
    ].sort_values("test_auc", ascending=False)
    if rows.empty:
        return "unknown"
    return str(rows.iloc[0]["proxy"])


def _negative_exemplars_for_feature(
    positives: pd.DataFrame,
    all_scores: pd.DataFrame,
    matches: pd.DataFrame,
    proxy: str,
    feature_layer: int,
    feature_id: int,
    top_count: int,
) -> pd.DataFrame:
    del feature_layer, feature_id
    if positives.empty:
        return pd.DataFrame(columns=all_scores.columns)
    collected: list[pd.Series] = []
    if proxy != "unknown" and not matches.empty:
        proxy_matches = matches.loc[matches["proxy"] == proxy].copy()
        for positive in positives.itertuples(index=False):
            matched = proxy_matches.loc[proxy_matches["positive_segment_id"] == positive.segment_id]
            if matched.empty:
                continue
            negative_ids = matched["negative_segment_id"].astype(str).tolist()
            candidate = all_scores.loc[all_scores["segment_id"].astype(str).isin(negative_ids)].sort_values(
                "pooled_activation",
                ascending=True,
            )
            if candidate.empty:
                continue
            collected.append(candidate.iloc[0])
            if len(collected) >= top_count:
                break
    if len(collected) < top_count:
        fallback = all_scores.sort_values("pooled_activation", ascending=True).head(top_count - len(collected))
        for row in fallback.itertuples(index=False):
            collected.append(pd.Series(row._asdict()))
    if not collected:
        return pd.DataFrame(columns=all_scores.columns)
    return pd.DataFrame(collected).head(top_count)


def _exemplar_rows_from_frame(feature: object, frame: pd.DataFrame, example_kind: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(frame.itertuples(index=False), start=1):
        rows.append(
            {
                "catalog_key": feature.catalog_key,
                "model_id": feature.model_id,
                "sae_release": feature.sae_release,
                "layer": int(feature.layer),
                "feature_id": int(feature.feature_id),
                "ranking_family": feature.ranking_family,
                "rank": int(feature.rank),
                "example_kind": example_kind,
                "example_rank": rank,
                "segment_id": row.segment_id,
                "document_id": int(row.document_id),
                "pooled_activation": float(row.pooled_activation),
                "top_token_span_text": getattr(row, "top_token_span_text", ""),
                "text": row.text,
                "authority": row.authority,
                "jurisdiction": row.jurisdiction,
                "document_form": row.document_form,
                "collection_list": row.collection_list,
            }
        )
    return rows


def _build_cross_layer_overlap_tables(
    catalog_frame: pd.DataFrame,
    decoder_manifest: pd.DataFrame,
    final_vectors: dict[str, torch.Tensor],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_rows: list[dict[str, object]] = []
    similarity_rows: list[dict[str, object]] = []
    metadata = decoder_manifest.merge(
        catalog_frame[["catalog_key", "layer", "feature_id", "ranking_family"]],
        on=["catalog_key", "layer", "feature_id"],
        how="left",
    )
    for family in config.feature_catalog.ranking_families:
        family_rows = metadata.loc[metadata["ranking_family"] == family].copy()
        layers = sorted(family_rows["layer"].dropna().astype(int).unique().tolist())
        for left_index, left_layer in enumerate(layers):
            left_rows = family_rows.loc[family_rows["layer"] == left_layer].copy()
            if left_rows.empty:
                continue
            left_matrix = _decoder_matrix_from_rows(left_rows, final_vectors)
            for right_layer in layers[left_index + 1 :]:
                right_rows = family_rows.loc[family_rows["layer"] == right_layer].copy()
                if right_rows.empty:
                    continue
                right_matrix = _decoder_matrix_from_rows(right_rows, final_vectors)
                match_count, semantic_jaccard, mean_left, mean_right, max_similarity = _semantic_overlap_stats(
                    left_matrix,
                    right_matrix,
                    threshold=config.feature_catalog.cross_layer_similarity_threshold,
                )
                overlap_rows.append(
                    {
                        "ranking_family": family,
                        "left_layer": left_layer,
                        "right_layer": right_layer,
                        "match_count": match_count,
                        "semantic_jaccard": semantic_jaccard,
                        "threshold": config.feature_catalog.cross_layer_similarity_threshold,
                    }
                )
                similarity_rows.append(
                    {
                        "ranking_family": family,
                        "left_layer": left_layer,
                        "right_layer": right_layer,
                        "mean_best_left_to_right_cosine": mean_left,
                        "mean_best_right_to_left_cosine": mean_right,
                        "max_pairwise_cosine": max_similarity,
                        "threshold": config.feature_catalog.cross_layer_similarity_threshold,
                    }
                )
    return pd.DataFrame(overlap_rows), pd.DataFrame(similarity_rows)


def _decoder_matrix_from_rows(rows: pd.DataFrame, final_vectors: dict[str, torch.Tensor]) -> np.ndarray:
    if rows.empty:
        return np.zeros((0, 1), dtype=np.float32)
    matrix = np.vstack([final_vectors[str(key)].cpu().numpy() for key in rows["decoder_tensor_key"].tolist()]).astype(np.float32)
    return _l2_normalize(matrix)


def _semantic_overlap_stats(left_matrix: np.ndarray, right_matrix: np.ndarray, threshold: float) -> tuple[int, float, float, float, float]:
    if left_matrix.size == 0 or right_matrix.size == 0:
        return 0, 0.0, np.nan, np.nan, np.nan
    similarity = left_matrix @ right_matrix.T
    left_best = similarity.max(axis=1)
    right_best = similarity.max(axis=0)
    match_count = _greedy_match_count(similarity, threshold)
    semantic_jaccard = match_count / max(left_matrix.shape[0] + right_matrix.shape[0] - match_count, 1)
    return (
        int(match_count),
        float(semantic_jaccard),
        float(left_best.mean()),
        float(right_best.mean()),
        float(similarity.max()),
    )


def _greedy_match_count(similarity: np.ndarray, threshold: float) -> int:
    working = similarity.copy()
    count = 0
    while working.size > 0:
        flat_index = int(np.argmax(working))
        best_value = float(working.flat[flat_index])
        if best_value < threshold:
            break
        row_index, col_index = np.unravel_index(flat_index, working.shape)
        count += 1
        working[row_index, :] = -np.inf
        working[:, col_index] = -np.inf
    return count


def _build_feature_lineage_candidates(
    catalog_frame: pd.DataFrame,
    sae_loader: SaeLensAdapter,
    decoder_manifest: pd.DataFrame,
    final_vectors: dict[str, torch.Tensor],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_layer_cache: dict[int, np.ndarray] = {}
    target_feature_ids_cache: dict[int, list[int]] = {}
    metadata = decoder_manifest.set_index("catalog_key")
    unique_catalog = catalog_frame[["catalog_key", "layer", "feature_id"]].drop_duplicates()

    for source in unique_catalog.itertuples(index=False):
        source_vector = final_vectors[str(metadata.loc[source.catalog_key, "decoder_tensor_key"])].cpu().numpy().astype(np.float32)
        source_vector = _l2_normalize(source_vector.reshape(1, -1)).reshape(-1)
        for target_layer in sorted(unique_catalog["layer"].dropna().astype(int).unique().tolist()):
            if int(target_layer) == int(source.layer):
                continue
            if target_layer not in target_layer_cache:
                sae = sae_loader.load_for_layer(int(target_layer))
                matrix = sae.W_dec.detach().cpu().to(torch.float32).numpy()
                target_layer_cache[target_layer] = _l2_normalize(matrix)
                target_feature_ids_cache[target_layer] = list(range(matrix.shape[0]))
                del sae
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            similarity = target_layer_cache[target_layer] @ source_vector
            top_indices = np.argsort(similarity)[::-1][:5]
            for neighbor_rank, neighbor_index in enumerate(top_indices.tolist(), start=1):
                rows.append(
                    {
                        "source_catalog_key": source.catalog_key,
                        "source_layer": int(source.layer),
                        "source_feature_id": int(source.feature_id),
                        "target_layer": int(target_layer),
                        "target_feature_id": int(target_feature_ids_cache[target_layer][neighbor_index]),
                        "neighbor_rank": neighbor_rank,
                        "decoder_cosine": float(similarity[neighbor_index]),
                        "threshold": config.feature_catalog.cross_layer_similarity_threshold,
                    }
                )
    return pd.DataFrame(rows)


def _enrich_catalog_with_overlays_and_causality(
    catalog_frame: pd.DataFrame,
    overlay_frame: pd.DataFrame,
    interventions: pd.DataFrame,
) -> pd.DataFrame:
    frame = catalog_frame.copy()
    if not overlay_frame.empty:
        best_overlay = overlay_frame.sort_values("test_auc", ascending=False).drop_duplicates(["layer", "feature_id"])
        best_overlay = best_overlay[
            [
                "layer",
                "feature_id",
                "proxy",
                "test_auc",
                "validated_auc",
                "matched_negative_contrast_score",
            ]
        ].rename(
            columns={
                "proxy": "best_proxy",
                "test_auc": "best_proxy_test_auc",
                "validated_auc": "best_proxy_validated_auc",
                "matched_negative_contrast_score": "best_proxy_contrast",
            }
        )
        frame = frame.merge(best_overlay, on=["layer", "feature_id"], how="left")
    if interventions.empty:
        return frame
    badges: dict[tuple[int, int], str] = {}
    for row in interventions.itertuples(index=False):
        feature_ids = [int(feature_id) for feature_id in row.feature_ids]
        if getattr(row, "kl_divergence_delta_ci_low", np.nan) > 0:
            badge = "proxy_free_causal"
        elif getattr(row, "paired_delta_ci_low", np.nan) > 0:
            badge = "proxy_selective"
        else:
            badge = "observed_target"
        for feature_id in feature_ids:
            badges[(int(row.layer), feature_id)] = badge
    frame["causal_badge"] = frame.apply(
        lambda row: badges.get((int(row["layer"]), int(row["feature_id"])), row.get("causal_badge", "")),
        axis=1,
    )
    return frame


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (matrix / norms).astype(np.float32)


def _first_numeric_or_default(values: object, default: float = 0.0) -> float:
    if values is None:
        return default
    if isinstance(values, np.ndarray):
        return float(values[0]) if values.size > 0 else default
    if isinstance(values, (list, tuple)):
        return float(values[0]) if len(values) > 0 else default
    return default


def _first_integer_or_default(values: object, default: int = -1) -> int:
    if values is None:
        return default
    if isinstance(values, np.ndarray):
        return int(values[0]) if values.size > 0 else default
    if isinstance(values, (list, tuple)):
        return int(values[0]) if len(values) > 0 else default
    return default


def _feature_template_summary(
    feature: object,
    best_proxy: str,
    contexts: pd.DataFrame,
    positive_exemplars: pd.DataFrame,
    negative_exemplars: pd.DataFrame,
    top_positive_tokens: list[str],
) -> str:
    context_snippets = [str(item) for item in contexts["context_text"].head(3).tolist()]
    exemplar_snippets = [str(item)[:180] for item in positive_exemplars["text"].head(2).tolist()]
    negative_snippets = [str(item)[:140] for item in negative_exemplars["text"].head(2).tolist()]
    token_snippets = [str(item) for item in list(top_positive_tokens)[:8]]
    return (
        f"Layer {feature.layer} feature {feature.feature_id} ranked {feature.rank} in {feature.ranking_family}. "
        f"Best proxy overlay is {best_proxy}. "
        f"Representative contexts: {' | '.join(context_snippets)}. "
        f"High activation examples: {' | '.join(exemplar_snippets)}. "
        f"Contrastive negatives: {' | '.join(negative_snippets)}. "
        f"Top logit tokens: {' | '.join(token_snippets)}."
    )


def _generate_feature_name_and_rationale(
    generator: dict[str, object] | None,
    config: ExperimentConfig,
    feature: object,
    best_proxy: str,
    contexts: pd.DataFrame,
    positive_exemplars: pd.DataFrame,
    negative_exemplars: pd.DataFrame,
    top_positive_tokens: list[str],
) -> dict[str, str]:
    fallback_name = f"Layer {feature.layer} feature {feature.feature_id}"
    fallback_rationale = (
        f"This feature is prominent in {feature.ranking_family} ranking and aligns most strongly with {best_proxy} in post hoc overlay."
    )
    if generator is None:
        return {"name": fallback_name, "rationale": fallback_rationale}
    prompt = (
        "You are naming an interpretable sparse feature for policy text analysis.\n"
        "Return two lines only.\n"
        "Line 1: short feature name.\n"
        "Line 2: two sentence rationale.\n\n"
        f"Ranking family: {feature.ranking_family}\n"
        f"Best proxy: {best_proxy}\n"
        f"Top contexts: {contexts['context_text'].head(config.feature_catalog.top_contexts_per_feature).tolist()}\n"
        f"Top token spans: {contexts['top_token_span_text'].head(config.feature_catalog.top_contexts_per_feature).tolist()}\n"
        f"Positive exemplars: {positive_exemplars['text'].head(config.feature_catalog.top_exemplars_per_feature).tolist()}\n"
        f"Negative exemplars: {negative_exemplars['text'].head(config.labeling.num_negative_controls).tolist()}\n"
        f"Top logit tokens: {list(top_positive_tokens)[:config.feature_catalog.logit_top_k]}\n"
    )
    tokenizer = generator["tokenizer"]
    model = generator["model"]
    encoded = tokenizer(prompt, return_tensors="pt").to(config.backbone.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=config.labeling.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = decoded[len(prompt) :].strip()
    lines = _normalize_generation_lines(generated)
    if len(lines) < 2:
        return {"name": fallback_name, "rationale": fallback_rationale}
    name = lines[0]
    rationale = _normalize_sentence_text(" ".join(lines[1:]))
    if not _looks_like_label_name(name) or not _looks_like_rationale(rationale):
        return {"name": fallback_name, "rationale": fallback_rationale}
    return {"name": name, "rationale": rationale}


def _infer_feature_semantic_tag(contexts: pd.DataFrame, top_positive_tokens: list[str]) -> str:
    context_text = " ".join(contexts["context_text"].astype(str).head(5).tolist()).lower()
    token_blob = " ".join(map(str, top_positive_tokens)).lower()
    surface_markers = ["section", "article", "paragraph", "subsection", "shall", "(", ")", "chapter"]
    semantic_markers = ["privacy", "data", "fair", "bias", "discrimination", "security", "transparency", "rights"]
    surface_score = sum(marker in context_text or marker in token_blob for marker in surface_markers)
    semantic_score = sum(marker in context_text or marker in token_blob for marker in semantic_markers)
    return "surface" if surface_score > semantic_score else "semantic"


def _layer_depth_fraction(layer: int, model_depth: int) -> float:
    if model_depth <= 1:
        return 0.0
    return float(layer / max(model_depth - 1, 1))


def _layer_stage(layer_fraction: float) -> str:
    if layer_fraction < 0.33:
        return "early"
    if layer_fraction < 0.66:
        return "mid"
    return "late"
