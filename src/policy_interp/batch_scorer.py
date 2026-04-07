"""Offline feature first batch scorer for new documents or segment bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
import torch
from jinja2 import Environment, FileSystemLoader

from policy_interp.adapters.modeling import HuggingFaceBackboneAdapter, SaeLensAdapter
from policy_interp.extract import run_extraction_for_segments
from policy_interp.io import read_jsonl, read_parquet, write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir, normalize_text, set_seed


@dataclass(slots=True)
class BatchScorerArtifacts:
    document_feature_scores_path: str
    top_feature_evidence_path: str
    layer_profile_summary_path: str
    proxy_overlay_summary_path: str
    causal_notes_path: str
    report_path: str


def run_batch_scorer(
    config: ExperimentConfig,
    input_path: str | Path,
    output_name: str = "batch_scorer",
    segment_mode: str = "paragraph",
) -> BatchScorerArtifacts:
    set_seed(config.splits.seed)
    feature_root = config.run_root / "features"
    catalog_path = feature_root / "feature_catalog.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError("Feature catalog not found. Run build-feature-catalog first.")

    catalog = read_parquet(catalog_path)
    if catalog.empty:
        raise ValueError("Feature catalog is empty. Build a populated feature catalog before batch scoring.")

    overlay = read_parquet(feature_root / "feature_proxy_overlay.parquet") if (feature_root / "feature_proxy_overlay.parquet").exists() else pd.DataFrame()
    labels = _load_feature_label_frame(feature_root)
    causal_path = config.run_root / "interventions" / "feature_causal_summary.parquet"
    if causal_path.exists():
        causal = read_parquet(causal_path)
    elif (config.run_root / "interventions" / "ablation_sparse_vs_dense.parquet").exists():
        causal = read_parquet(config.run_root / "interventions" / "ablation_sparse_vs_dense.parquet")
    else:
        causal = pd.DataFrame()

    raw_text = Path(input_path).read_text(encoding="utf-8")
    segments = _segment_document_text(raw_text, output_name=output_name, segment_mode=segment_mode)
    output_root = ensure_dir(config.run_root / "features" / "batch_scorer" / output_name)
    extraction_root = ensure_dir(output_root / "extraction")

    scoring_config = config.model_copy(deep=True)
    scoring_config.extract.layers = sorted(catalog["layer"].dropna().astype(int).unique().tolist())
    per_layer_catalog = int(
        catalog.groupby("layer")["feature_id"].nunique().max()
        if not catalog.empty
        else scoring_config.extract.segment_top_feature_count
    )
    scoring_config.extract.segment_top_feature_count = max(
        scoring_config.extract.segment_top_feature_count,
        min(per_layer_catalog + 32, 1024),
    )

    backbone = HuggingFaceBackboneAdapter(scoring_config.backbone).load()
    sae_loader = SaeLensAdapter(scoring_config.sae)
    try:
        run_extraction_for_segments(
            config=scoring_config,
            prepared_segments=segments,
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

    extraction_features = _read_scored_top_features(extraction_root, scoring_config.extract.layers)
    document_feature_scores = _build_document_feature_scores(extraction_features, catalog, labels, segments)
    overlay_summary = _build_proxy_overlay_summary(document_feature_scores, overlay)
    causal_notes = _build_causal_notes(document_feature_scores, causal)
    layer_profile = _build_layer_profile_summary(document_feature_scores)
    top_feature_evidence = document_feature_scores.sort_values("pooled_activation", ascending=False).head(50).copy()

    document_feature_scores_path = write_parquet(document_feature_scores, output_root / "document_feature_scores.parquet")
    top_feature_evidence_path = write_parquet(top_feature_evidence, output_root / "top_feature_evidence.parquet")
    layer_profile_summary_path = write_parquet(layer_profile, output_root / "layer_profile_summary.parquet")
    proxy_overlay_summary_path = write_parquet(overlay_summary, output_root / "proxy_overlay_summary.parquet")
    causal_notes_path = write_parquet(causal_notes, output_root / "causal_notes.parquet")

    report_path = output_root / f"batch_scorer_report.{config.report.dossier_format}"
    _render_batch_report(
        report_path=report_path,
        output_name=output_name,
        segment_mode=segment_mode,
        segment_count=len(segments),
        top_feature_evidence=top_feature_evidence,
        layer_profile=layer_profile,
        overlay_summary=overlay_summary,
        causal_notes=causal_notes,
    )

    return BatchScorerArtifacts(
        document_feature_scores_path=str(document_feature_scores_path),
        top_feature_evidence_path=str(top_feature_evidence_path),
        layer_profile_summary_path=str(layer_profile_summary_path),
        proxy_overlay_summary_path=str(proxy_overlay_summary_path),
        causal_notes_path=str(causal_notes_path),
        report_path=str(report_path),
    )


def _segment_document_text(raw_text: str, output_name: str, segment_mode: str) -> pd.DataFrame:
    cleaned_text = raw_text.replace("\r\n", "\n")
    if segment_mode == "whole":
        units = [normalize_text(cleaned_text)]
    elif segment_mode == "line":
        units = [normalize_text(line) for line in cleaned_text.splitlines() if normalize_text(line)]
    else:
        units = [normalize_text(part) for part in re.split(r"\n\s*\n+", cleaned_text) if normalize_text(part)]
    if not units:
        units = [normalize_text(cleaned_text)]

    rows: list[dict[str, object]] = []
    document_id = 0
    for index, text in enumerate(units):
        rows.append(
            {
                "segment_id": f"{output_name}_segment_{index:04d}",
                "document_id": document_id,
                "split": "inference",
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def _read_scored_top_features(extraction_root: Path, layers: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for layer in layers:
        path = extraction_root / f"segment_top_features_layer_{layer}.parquet"
        if path.exists():
            frames.append(read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_feature_label_frame(feature_root: Path) -> pd.DataFrame:
    key_columns = ["layer", "feature_id", "ranking_family", "rank"]
    default_columns = [
        "layer",
        "feature_id",
        "ranking_family",
        "rank",
        "generated_name",
        "generated_rationale",
        "template_summary",
        "semantic_tag",
        "best_proxy",
        "faithfulness_score",
        "boundary_text",
    ]
    frames: list[pd.DataFrame] = []
    autointerp_path = feature_root / "autointerp" / "autointerp_feature_labels.jsonl"
    if autointerp_path.exists():
        frame = pd.DataFrame(read_jsonl(autointerp_path))
        if not frame.empty:
            frames.append(
                frame.rename(
                    columns={
                        "primary_ranking_family": "ranking_family",
                        "feature_name": "generated_name",
                        "activation_hypothesis": "generated_rationale",
                        "context_summary": "template_summary",
                    }
                )
            )
    default_path = feature_root / "feature_labels.jsonl"
    if default_path.exists():
        default_frame = pd.DataFrame(read_jsonl(default_path))
        if not default_frame.empty:
            frames.append(default_frame)
    if not frames:
        return pd.DataFrame(columns=default_columns)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    for column in default_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged = merged.drop_duplicates(key_columns, keep="first")
    for column in ["generated_name", "generated_rationale", "template_summary", "semantic_tag", "best_proxy", "boundary_text"]:
        merged[column] = merged[column].fillna("")
    return merged[default_columns]


def _build_document_feature_scores(
    extraction_features: pd.DataFrame,
    catalog: pd.DataFrame,
    labels: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    if extraction_features.empty:
        return pd.DataFrame()

    label_columns = ["layer", "feature_id", "ranking_family", "rank", "generated_name", "generated_rationale", "semantic_tag"]
    labels_frame = labels[label_columns].copy() if not labels.empty else pd.DataFrame(columns=label_columns)
    catalog_metadata = catalog[
        [
            "model_id",
            "sae_release",
            "layer",
            "model_depth",
            "layer_depth_fraction",
            "layer_stage",
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
    merged = extraction_features.merge(
        catalog_metadata,
        on=["layer", "feature_id"],
        how="inner",
        suffixes=("", "_catalog"),
    )
    merged = merged.merge(
        labels_frame,
        on=["layer", "feature_id", "ranking_family", "rank"],
        how="left",
    )
    merged = merged.merge(
        segments[["segment_id", "text"]],
        on="segment_id",
        how="left",
        suffixes=("", "_segment"),
    )
    merged["display_name"] = merged["generated_name"].fillna(
        merged.apply(lambda row: f"Layer {int(row['layer'])} feature {int(row['feature_id'])}", axis=1)
    )
    return merged.sort_values(["pooled_activation", "layer", "feature_id"], ascending=[False, True, True]).reset_index(drop=True)


def _build_proxy_overlay_summary(document_feature_scores: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    if document_feature_scores.empty or overlay.empty:
        return pd.DataFrame()
    merged = document_feature_scores[
        ["segment_id", "layer", "feature_id", "pooled_activation", "display_name"]
    ].merge(
        overlay[["layer", "feature_id", "proxy", "test_auc", "validated_auc", "matched_negative_contrast_score"]],
        on=["layer", "feature_id"],
        how="left",
    )
    merged["weighted_proxy_signal"] = merged["pooled_activation"].astype(float) * merged["test_auc"].fillna(0.0).astype(float)
    summary = (
        merged.groupby("proxy")
        .agg(
            activated_feature_count=("feature_id", "nunique"),
            mean_test_auc=("test_auc", "mean"),
            mean_validated_auc=("validated_auc", "mean"),
            mean_matched_negative_contrast=("matched_negative_contrast_score", "mean"),
            total_weighted_proxy_signal=("weighted_proxy_signal", "sum"),
        )
        .reset_index()
        .sort_values("total_weighted_proxy_signal", ascending=False)
    )
    return summary


def _build_causal_notes(document_feature_scores: pd.DataFrame, causal: pd.DataFrame) -> pd.DataFrame:
    if document_feature_scores.empty or causal.empty or "feature_ids" not in causal.columns:
        return pd.DataFrame()
    expanded_rows: list[dict[str, object]] = []
    for row in causal.itertuples(index=False):
        feature_ids = _coerce_feature_id_list(getattr(row, "feature_ids", None))
        for feature_id in feature_ids:
            expanded_rows.append(
                {
                    "target_id": row.target_id,
                    "target_name": row.target_name,
                    "target_kind": getattr(row, "target_kind", getattr(row, "target_type", "unknown")),
                    "primary_proxy": row.primary_proxy,
                    "layer": int(row.layer),
                    "feature_id": feature_id,
                    "paired_delta": getattr(row, "paired_delta", None),
                    "kl_divergence_delta": getattr(row, "kl_divergence_delta", None),
                    "perplexity_shift_delta": getattr(row, "perplexity_shift_delta", None),
                    "top1_change_rate_delta": getattr(row, "top1_change_rate_delta", None),
                    "causal_badge": getattr(row, "causal_badge", None),
                }
            )
    expanded = pd.DataFrame(expanded_rows)
    if expanded.empty:
        return expanded
    merged = document_feature_scores[
        ["segment_id", "layer", "feature_id", "display_name", "pooled_activation"]
    ].merge(
        expanded,
        on=["layer", "feature_id"],
        how="inner",
    )
    return merged.sort_values("pooled_activation", ascending=False).reset_index(drop=True)


def _coerce_feature_id_list(values: object) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [int(values)]


def _build_layer_profile_summary(document_feature_scores: pd.DataFrame) -> pd.DataFrame:
    if document_feature_scores.empty:
        return pd.DataFrame()
    frame = document_feature_scores.copy()
    frame["semantic_tag"] = frame["semantic_tag"].fillna("unknown")
    summary = (
        frame.groupby(["layer", "layer_stage"])
        .agg(
            activated_catalog_features=("feature_id", "nunique"),
            mean_pooled_activation=("pooled_activation", "mean"),
            max_pooled_activation=("pooled_activation", "max"),
            semantic_feature_count=("semantic_tag", lambda values: int(sum(item == "semantic" for item in values))),
            surface_feature_count=("semantic_tag", lambda values: int(sum(item == "surface" for item in values))),
        )
        .reset_index()
        .sort_values("layer")
    )
    return summary


def _render_batch_report(
    report_path: Path,
    output_name: str,
    segment_mode: str,
    segment_count: int,
    top_feature_evidence: pd.DataFrame,
    layer_profile: pd.DataFrame,
    overlay_summary: pd.DataFrame,
    causal_notes: pd.DataFrame,
) -> None:
    template_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("batch_report.md.j2")
    rendered = template.render(
        output_name=output_name,
        segment_mode=segment_mode,
        segment_count=segment_count,
        top_feature_evidence=top_feature_evidence.head(15).to_dict(orient="records"),
        layer_profile=layer_profile.to_dict(orient="records"),
        overlay_summary=overlay_summary.head(8).to_dict(orient="records"),
        causal_notes=causal_notes.head(12).to_dict(orient="records"),
    )
    report_path.write_text(rendered, encoding="utf-8")
