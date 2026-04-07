"""Paper facing exports for module first and feature first artifacts."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from jinja2 import Environment, FileSystemLoader

from policy_interp.audit_eval import export_audit_reports
from policy_interp.io import read_jsonl, read_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir


def export_reports(config: ExperimentConfig) -> dict[str, str]:
    export_root = ensure_dir(config.run_root / config.report.export_dir)
    outputs: dict[str, str] = {}

    outputs["module_heatmap"] = str(export_heatmap(config, export_root))
    outputs["module_dossiers"] = str(export_module_dossiers(config, export_root))

    if config.report.export_feature_catalog:
        outputs.update(export_feature_catalog_reports(config, export_root))
    if config.report.export_layer_overlap:
        outputs.update(export_layer_overlap_reports(config, export_root))
    if config.report.export_causal_feature_tables:
        outputs.update(export_feature_causal_reports(config, export_root))
    if config.report.export_autointerp_reports:
        outputs.update(export_autointerp_reports(config, export_root))
    if config.report.export_autointerp_causal_reports:
        outputs.update(export_autointerp_causal_reports(config, export_root))
    if config.report.export_audit_reports:
        outputs.update(export_audit_reports(config, export_root))

    return outputs


def export_heatmap(config: ExperimentConfig, export_root: Path) -> Path:
    alignment_path = config.run_root / "discovery" / "module_proxy_alignment.parquet"
    target = export_root / "module_proxy_heatmap.png"
    if not alignment_path.exists():
        return _write_empty_figure(target, title="Module Proxy Alignment")

    alignment = read_parquet(alignment_path)
    if alignment.empty:
        return _write_empty_figure(target, title="Module Proxy Alignment")

    best = alignment.sort_values("test_auc", ascending=False).drop_duplicates(["stable_module_id", "proxy"])
    pivot = best.pivot(index="stable_module_id", columns="proxy", values="test_auc").fillna(0.0)
    plt.figure(figsize=(10, max(4, 0.35 * len(pivot))))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.2)
    plt.title("Module Proxy Alignment")
    plt.tight_layout()
    plt.savefig(target, dpi=config.report.heatmap_dpi)
    plt.close()
    return target


def export_module_dossiers(config: ExperimentConfig, export_root: Path) -> Path:
    labels_path = config.run_root / config.report.export_dir / "module_labels.jsonl"
    dossier_dir = ensure_dir(export_root / "dossiers")
    if not labels_path.exists():
        return dossier_dir

    labels = read_jsonl(labels_path)
    template = _template_env().get_template("dossier.md.j2")
    for row in labels:
        rendered = template.render(**row)
        target = dossier_dir / f"{row['stable_module_id']}.md"
        target.write_text(rendered, encoding="utf-8")
    return dossier_dir


def export_feature_catalog_reports(config: ExperimentConfig, export_root: Path) -> dict[str, str]:
    feature_root = config.run_root / "features"
    catalog_path = feature_root / "feature_catalog.parquet"
    outputs: dict[str, str] = {}
    if not catalog_path.exists():
        return outputs

    catalog = read_parquet(catalog_path)
    if catalog.empty:
        return outputs

    labels_frame = _feature_labels_frame(feature_root / "feature_labels.jsonl")
    exemplars_path = feature_root / "feature_catalog_exemplars.parquet"
    exemplars = read_parquet(exemplars_path) if exemplars_path.exists() else pd.DataFrame()
    overlay_path = feature_root / "feature_proxy_overlay.parquet"
    overlay = read_parquet(overlay_path) if overlay_path.exists() else pd.DataFrame()
    logit_path = feature_root / "feature_catalog_logit_attribution.parquet"
    logit_frame = read_parquet(logit_path) if logit_path.exists() else pd.DataFrame()
    audit_validation_path = config.run_root / "audit_eval" / "autointerp_validation_feature_scores.parquet"
    audit_validation = read_parquet(audit_validation_path) if audit_validation_path.exists() else pd.DataFrame()

    merged = catalog.merge(
        labels_frame,
        on=["layer", "feature_id", "ranking_family", "rank"],
        how="left",
    )
    if not audit_validation.empty:
        validation_columns = [
            "layer",
            "feature_id",
            "primary_ranking_family",
            "contrastive_accuracy",
            "lexicality_penalty",
            "faithfulness_score",
        ]
        available_columns = [column for column in validation_columns if column in audit_validation.columns]
        validation_frame = audit_validation[available_columns].drop_duplicates(
            ["layer", "feature_id", "primary_ranking_family"]
        )
        validation_frame = validation_frame.rename(columns={"primary_ranking_family": "ranking_family"})
        merged = merged.merge(
            validation_frame,
            on=["layer", "feature_id", "ranking_family"],
            how="left",
            suffixes=("", "_audit"),
        )
    catalog_table_path = export_root / "feature_catalog_table.csv"
    merged.to_csv(catalog_table_path, index=False)
    outputs["feature_catalog_table"] = str(catalog_table_path)

    heatmap_path = export_root / "feature_catalog_heatmap.png"
    _export_feature_catalog_heatmap(merged, heatmap_path, config.report.heatmap_dpi)
    outputs["feature_catalog_heatmap"] = str(heatmap_path)

    dossier_dir = ensure_dir(export_root / "feature_dossiers")
    template = _template_env().get_template("feature_dossier.md.j2")
    best_overlay = _best_proxy_overlay(overlay)
    best_logit = logit_frame.drop_duplicates(["layer", "feature_id"], keep="first") if not logit_frame.empty else pd.DataFrame()

    for row in merged.itertuples(index=False):
        row_overlay = best_overlay.loc[
            (best_overlay["layer"] == int(row.layer)) & (best_overlay["feature_id"] == int(row.feature_id))
        ]
        row_logit = best_logit.loc[
            (best_logit["layer"] == int(row.layer)) & (best_logit["feature_id"] == int(row.feature_id))
        ]
        row_exemplars = exemplars.loc[
            (exemplars["layer"] == int(row.layer)) & (exemplars["feature_id"] == int(row.feature_id))
        ]
        overlay_record = _normalize_record(row_overlay.iloc[0].to_dict()) if not row_overlay.empty else {}
        if overlay_record and "proxy" in overlay_record and "best_proxy" not in overlay_record:
            overlay_record["best_proxy"] = overlay_record["proxy"]
        logit_record = _normalize_record(row_logit.iloc[0].to_dict()) if not row_logit.empty else {}
        rendered = template.render(
            feature=row,
            overlay=overlay_record,
            logit=logit_record,
            positive_exemplars=row_exemplars.loc[row_exemplars["example_kind"] == "positive"].head(5).to_dict(orient="records"),
            negative_exemplars=row_exemplars.loc[row_exemplars["example_kind"] == "negative"].head(5).to_dict(orient="records"),
        )
        target = dossier_dir / f"layer_{int(row.layer)}_feature_{int(row.feature_id)}_{row.ranking_family}.md"
        target.write_text(rendered, encoding="utf-8")
    outputs["feature_dossiers"] = str(dossier_dir)
    return outputs


def export_layer_overlap_reports(config: ExperimentConfig, export_root: Path) -> dict[str, str]:
    feature_root = config.run_root / "features"
    overlap_path = feature_root / "cross_layer_feature_overlap.parquet"
    similarity_path = feature_root / "cross_layer_decoder_similarity.parquet"
    outputs: dict[str, str] = {}

    if overlap_path.exists():
        overlap = read_parquet(overlap_path)
        target = export_root / "cross_layer_feature_overlap.png"
        _export_layer_pair_matrix(
            frame=overlap,
            value_column="semantic_jaccard",
            title="Cross Layer Feature Overlap",
            target=target,
            dpi=config.report.heatmap_dpi,
        )
        outputs["cross_layer_feature_overlap"] = str(target)

    if similarity_path.exists():
        similarity = read_parquet(similarity_path)
        target = export_root / "cross_layer_decoder_similarity.png"
        _export_layer_pair_matrix(
            frame=similarity,
            value_column="mean_best_left_to_right_cosine",
            title="Cross Layer Decoder Similarity",
            target=target,
            dpi=config.report.heatmap_dpi,
        )
        outputs["cross_layer_decoder_similarity"] = str(target)

    return outputs


def export_feature_causal_reports(config: ExperimentConfig, export_root: Path) -> dict[str, str]:
    intervention_root = config.run_root / "interventions"
    feature_summary_path = intervention_root / "feature_causal_summary.parquet"
    legacy_path = intervention_root / "ablation_sparse_vs_dense.parquet"
    outputs: dict[str, str] = {}

    if feature_summary_path.exists():
        causal = read_parquet(feature_summary_path)
    elif legacy_path.exists():
        causal = read_parquet(legacy_path)
    else:
        return outputs

    csv_path = export_root / "feature_causal_summary.csv"
    parquet_path = export_root / "feature_causal_summary.parquet"
    causal.to_csv(csv_path, index=False)
    causal.to_parquet(parquet_path, index=False)
    outputs["feature_causal_summary_csv"] = str(csv_path)
    outputs["feature_causal_summary_parquet"] = str(parquet_path)
    return outputs


def export_autointerp_reports(config: ExperimentConfig, export_root: Path) -> dict[str, str]:
    feature_root = config.run_root / "features"
    scores_path = feature_root / "autointerp" / "autointerp_feature_scores.parquet"
    labels_path = feature_root / "autointerp" / "autointerp_feature_labels.jsonl"
    catalog_path = feature_root / "feature_catalog.parquet"
    outputs: dict[str, str] = {}
    if not scores_path.exists() or not labels_path.exists() or not catalog_path.exists():
        return outputs

    scores = read_parquet(scores_path)
    labels = pd.DataFrame(read_jsonl(labels_path))
    catalog = read_parquet(catalog_path)
    if scores.empty or labels.empty or catalog.empty:
        return outputs

    merged = scores.merge(
        labels,
        on=["model_id", "sae_release", "layer", "feature_id", "primary_ranking_family", "rank"],
        how="left",
        suffixes=("", "_label"),
    )
    merged = merged.merge(
        catalog[
            [
                "layer",
                "feature_id",
                "ranking_family",
                "rank",
                "layer_stage",
                "score",
                "best_proxy_test_auc",
                "best_proxy_validated_auc",
            ]
        ].rename(columns={"ranking_family": "primary_ranking_family"}),
        on=["layer", "feature_id", "primary_ranking_family", "rank"],
        how="left",
        suffixes=("", "_catalog"),
    )
    merged["feature_name"] = merged["feature_name"].fillna(merged.get("generated_name", ""))
    merged["activation_hypothesis"] = merged["activation_hypothesis"].fillna(merged.get("generated_rationale", ""))
    merged["best_proxy"] = merged["best_proxy"].fillna(merged.get("best_proxy_catalog", ""))

    top_features = (
        merged.sort_values(["layer", "faithfulness_score", "simulation_accuracy", "rank"], ascending=[True, False, False, True])
        .groupby("layer", group_keys=False)
        .head(config.report.autointerp_top_features_per_layer)
        .copy()
    )
    top_features["high_faithfulness"] = top_features["faithfulness_score"] >= config.report.autointerp_high_faithfulness_threshold
    top_features_path = export_root / "autointerp_top_features.csv"
    top_features.to_csv(top_features_path, index=False)
    outputs["autointerp_top_features"] = str(top_features_path)

    layer_summary = (
        merged.groupby(["layer", "layer_stage"], as_index=False)
        .agg(
            feature_count=("feature_id", "count"),
            mean_faithfulness=("faithfulness_score", "mean"),
            median_faithfulness=("faithfulness_score", "median"),
            max_faithfulness=("faithfulness_score", "max"),
            mean_accuracy=("simulation_accuracy", "mean"),
            count_high_faithfulness=("faithfulness_score", lambda col: int((col >= config.report.autointerp_high_faithfulness_threshold).sum())),
        )
        .sort_values("layer")
    )
    layer_summary_path = export_root / "autointerp_layer_summary.csv"
    layer_summary.to_csv(layer_summary_path, index=False)
    outputs["autointerp_layer_summary"] = str(layer_summary_path)

    family_summary = (
        merged.groupby(["layer", "primary_ranking_family"], as_index=False)
        .agg(
            feature_count=("feature_id", "count"),
            mean_faithfulness=("faithfulness_score", "mean"),
            max_faithfulness=("faithfulness_score", "max"),
            mean_accuracy=("simulation_accuracy", "mean"),
        )
        .sort_values(["layer", "primary_ranking_family"])
    )
    family_summary_path = export_root / "autointerp_family_summary.csv"
    family_summary.to_csv(family_summary_path, index=False)
    outputs["autointerp_family_summary"] = str(family_summary_path)

    layer_heatmap_path = export_root / "autointerp_layer_summary.png"
    _export_autointerp_layer_summary_heatmap(layer_summary, layer_heatmap_path, config.report.heatmap_dpi)
    outputs["autointerp_layer_summary_heatmap"] = str(layer_heatmap_path)

    family_heatmap_path = export_root / "autointerp_family_heatmap.png"
    _export_autointerp_family_heatmap(family_summary, family_heatmap_path, config.report.heatmap_dpi)
    outputs["autointerp_family_heatmap"] = str(family_heatmap_path)

    gallery_path = export_root / "autointerp_feature_gallery.md"
    _write_autointerp_feature_gallery(top_features, gallery_path)
    outputs["autointerp_feature_gallery"] = str(gallery_path)

    policy_feature_map = _build_policy_feature_map_frame(
        top_features=top_features,
        causal_frame=_read_optional_frame(config.run_root / "interventions" / "feature_causal_summary.parquet"),
        overlay_frame=_read_optional_frame(feature_root / "feature_proxy_overlay.parquet"),
    )
    policy_feature_map_csv = export_root / "policy_feature_map.csv"
    policy_feature_map.to_csv(policy_feature_map_csv, index=False)
    outputs["policy_feature_map_csv"] = str(policy_feature_map_csv)

    policy_feature_map_path = export_root / "policy_feature_map.png"
    _export_policy_feature_map(policy_feature_map, policy_feature_map_path, config.report.heatmap_dpi)
    outputs["policy_feature_map"] = str(policy_feature_map_path)
    return outputs


def export_autointerp_causal_reports(config: ExperimentConfig, export_root: Path) -> dict[str, str]:
    feature_summary_path = config.run_root / "interventions" / "feature_causal_summary.parquet"
    outputs: dict[str, str] = {}
    if not feature_summary_path.exists():
        return outputs

    summary = read_parquet(feature_summary_path)
    if summary.empty:
        return outputs

    autointerp_rows = summary.loc[summary["target_type"] == "autointerp_single_feature"].copy()
    if autointerp_rows.empty:
        return outputs

    autointerp_rows["feature_id"] = autointerp_rows["feature_ids"].apply(
        lambda value: int(value[0]) if isinstance(value, (list, tuple)) and value else pd.NA
    )
    autointerp_rows["feature_name"] = autointerp_rows["target_name"].astype(str).str.replace(
        r"^AutoInterp layer \d+\s+",
        "",
        regex=True,
    )

    target_csv_path = export_root / "autointerp_causal_targets.csv"
    autointerp_rows.to_csv(target_csv_path, index=False)
    outputs["autointerp_causal_targets"] = str(target_csv_path)

    layer_summary = (
        autointerp_rows.groupby("layer", as_index=False)
        .agg(
            feature_count=("target_id", "count"),
            mean_kl_delta=("kl_divergence_delta", "mean"),
            max_kl_delta=("kl_divergence_delta", "max"),
            mean_paired_delta=("paired_delta", "mean"),
            max_paired_delta=("paired_delta", "max"),
            mean_top1_delta=("top1_change_rate_delta", "mean"),
            mean_perplexity_delta=("perplexity_shift_delta", "mean"),
            count_proxy_free_causal=("causal_badge", lambda col: int((col == "proxy_free_causal").sum())),
            count_proxy_selective=("causal_badge", lambda col: int((col == "proxy_selective").sum())),
        )
        .sort_values("layer")
    )
    layer_summary_path = export_root / "autointerp_causal_layer_summary.csv"
    layer_summary.to_csv(layer_summary_path, index=False)
    outputs["autointerp_causal_layer_summary"] = str(layer_summary_path)

    concentration_path = export_root / "autointerp_causal_concentration.png"
    _export_autointerp_causal_concentration(layer_summary, concentration_path, config.report.heatmap_dpi)
    outputs["autointerp_causal_concentration"] = str(concentration_path)

    return outputs


def _read_optional_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def _template_env() -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Environment(loader=FileSystemLoader(str(template_dir)))


def _write_empty_figure(target: Path, title: str) -> Path:
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(target, dpi=180, bbox_inches="tight")
    plt.close()
    return target


def _feature_labels_frame(path: Path) -> pd.DataFrame:
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
    autointerp_path = path.parent / "autointerp" / "autointerp_feature_labels.jsonl"
    if autointerp_path.exists():
        frame = pd.DataFrame(read_jsonl(autointerp_path))
        if not frame.empty:
            frame = frame.rename(
                columns={
                    "primary_ranking_family": "ranking_family",
                    "feature_name": "generated_name",
                    "activation_hypothesis": "generated_rationale",
                    "context_summary": "template_summary",
                }
            )
            frames.append(frame)
    if path.exists():
        default_frame = pd.DataFrame(read_jsonl(path))
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


def _best_proxy_overlay(overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return pd.DataFrame()
    return overlay.sort_values("test_auc", ascending=False).drop_duplicates(["layer", "feature_id"])


def _normalize_record(record: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in record.items():
        if hasattr(value, "tolist"):
            normalized[key] = value.tolist()
        else:
            normalized[key] = value
    return normalized


def _export_feature_catalog_heatmap(catalog: pd.DataFrame, target: Path, dpi: int) -> None:
    if catalog.empty:
        _write_empty_figure(target, title="Feature Catalog Heatmap")
        return
    display = catalog.copy()
    display["row_label"] = display.apply(
        lambda row: str(row.get("generated_name") or f"L{int(row['layer'])} F{int(row['feature_id'])}"),
        axis=1,
    )
    display["column_label"] = display.apply(
        lambda row: f"L{int(row['layer'])} {row['ranking_family']}",
        axis=1,
    )
    trimmed = display.sort_values(["layer", "ranking_family", "rank"]).groupby(["layer", "ranking_family"]).head(25)
    pivot = trimmed.pivot_table(index="row_label", columns="column_label", values="score", aggfunc="max").fillna(0.0)
    plt.figure(figsize=(12, max(6, 0.28 * len(pivot))))
    sns.heatmap(pivot, cmap="mako", linewidths=0.2)
    plt.title("Feature First Catalog Scores")
    plt.tight_layout()
    plt.savefig(target, dpi=dpi)
    plt.close()


def _export_layer_pair_matrix(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    target: Path,
    dpi: int,
) -> None:
    if frame.empty:
        _write_empty_figure(target, title=title)
        return

    layers = sorted(set(frame["left_layer"].astype(int).tolist()) | set(frame["right_layer"].astype(int).tolist()))
    matrix = pd.DataFrame(0.0, index=layers, columns=layers)
    for row in frame.itertuples(index=False):
        left_layer = int(row.left_layer)
        right_layer = int(row.right_layer)
        value = float(getattr(row, value_column))
        matrix.loc[left_layer, right_layer] = value
        matrix.loc[right_layer, left_layer] = value
    for layer in layers:
        matrix.loc[layer, layer] = 1.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, cmap="crest", annot=True, fmt=".2f", linewidths=0.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(target, dpi=dpi)
    plt.close()


def _export_autointerp_layer_summary_heatmap(frame: pd.DataFrame, target: Path, dpi: int) -> None:
    if frame.empty:
        _write_empty_figure(target, title="AutoInterp Layer Summary")
        return
    plot_frame = frame.copy()
    plot_frame["layer_label"] = plot_frame.apply(lambda row: f"L{int(row['layer'])} {row['layer_stage']}", axis=1)
    matrix = plot_frame.set_index("layer_label")[
        ["mean_faithfulness", "median_faithfulness", "max_faithfulness", "mean_accuracy"]
    ]
    plt.figure(figsize=(7, max(4, 0.7 * len(matrix))))
    sns.heatmap(matrix, cmap="mako", annot=True, fmt=".2f", linewidths=0.2)
    plt.title("AutoInterp Layer Summary")
    plt.tight_layout()
    plt.savefig(target, dpi=dpi)
    plt.close()


def _export_autointerp_family_heatmap(frame: pd.DataFrame, target: Path, dpi: int) -> None:
    if frame.empty:
        _write_empty_figure(target, title="AutoInterp Faithfulness by Layer and Family")
        return
    matrix = frame.pivot(index="layer", columns="primary_ranking_family", values="mean_faithfulness").fillna(0.0)
    plt.figure(figsize=(6, max(4, 0.8 * len(matrix))))
    sns.heatmap(matrix, cmap="crest", annot=True, fmt=".2f", linewidths=0.2)
    plt.title("AutoInterp Faithfulness by Layer and Family")
    plt.tight_layout()
    plt.savefig(target, dpi=dpi)
    plt.close()


def _export_autointerp_causal_concentration(frame: pd.DataFrame, target: Path, dpi: int) -> None:
    if frame.empty:
        _write_empty_figure(target, title="AutoInterp Causal Concentration")
        return
    matrix = frame.set_index("layer")[
        [
            "mean_kl_delta",
            "max_kl_delta",
            "mean_paired_delta",
            "mean_top1_delta",
            "mean_perplexity_delta",
        ]
    ]
    plt.figure(figsize=(7, max(4, 0.8 * len(matrix))))
    sns.heatmap(matrix, cmap="rocket", annot=True, fmt=".3f", linewidths=0.2)
    plt.title("AutoInterp Causal Concentration")
    plt.tight_layout()
    plt.savefig(target, dpi=dpi)
    plt.close()


def _build_policy_feature_map_frame(
    top_features: pd.DataFrame,
    causal_frame: pd.DataFrame,
    overlay_frame: pd.DataFrame,
) -> pd.DataFrame:
    if top_features.empty:
        return pd.DataFrame()

    frame = top_features.copy()
    frame["feature_name"] = frame["feature_name"].fillna("").astype(str).str.strip()
    frame["feature_name"] = frame.apply(
        lambda row: row["feature_name"] if row["feature_name"] else f"Feature {int(row['feature_id'])}",
        axis=1,
    )

    if not overlay_frame.empty:
        overlay = overlay_frame.sort_values("test_auc", ascending=False).drop_duplicates(["layer", "feature_id"])
        overlay = overlay.rename(
            columns={
                "proxy": "overlay_proxy",
                "test_auc": "overlay_test_auc",
                "validated_auc": "overlay_validated_auc",
                "mutual_information": "overlay_mutual_information",
            }
        )
        frame = frame.merge(
            overlay[
                [
                    "layer",
                    "feature_id",
                    "overlay_proxy",
                    "overlay_test_auc",
                    "overlay_validated_auc",
                    "overlay_mutual_information",
                ]
            ],
            on=["layer", "feature_id"],
            how="left",
        )
    else:
        frame["overlay_proxy"] = pd.NA
        frame["overlay_test_auc"] = pd.NA
        frame["overlay_validated_auc"] = pd.NA
        frame["overlay_mutual_information"] = pd.NA

    if not causal_frame.empty:
        causal = causal_frame.loc[causal_frame["target_type"] == "autointerp_single_feature"].copy()
        if not causal.empty:
            causal["feature_id"] = causal["feature_ids"].apply(
                _extract_first_feature_id
            )
            causal = causal[
                [
                    "layer",
                    "feature_id",
                    "target_id",
                    "paired_delta",
                    "paired_delta_ci_low",
                    "paired_delta_ci_high",
                    "kl_divergence_delta",
                    "kl_divergence_delta_ci_low",
                    "kl_divergence_delta_ci_high",
                    "perplexity_shift_delta",
                    "top1_change_rate_delta",
                    "causal_badge",
                ]
            ].drop_duplicates(["layer", "feature_id"])
            frame = frame.merge(causal, on=["layer", "feature_id"], how="left")
        else:
            frame = _attach_empty_causal_columns(frame)
    else:
        frame = _attach_empty_causal_columns(frame)

    frame["proxy_color_key"] = frame["best_proxy"].fillna(frame["overlay_proxy"]).fillna("unknown")
    frame["feature_label"] = frame.apply(
        lambda row: f"L{int(row['layer'])}  {row['feature_name']}",
        axis=1,
    )
    frame["causal_available"] = frame["target_id"].notna()
    frame["causal_magnitude"] = frame["kl_divergence_delta"].fillna(0.0).abs()
    frame["proxy_selective_magnitude"] = frame["paired_delta"].fillna(0.0).abs()
    frame["map_priority"] = frame["causal_available"].astype(int) * 1000.0 + frame["faithfulness_score"].fillna(0.0) * 100.0 + frame["score"].fillna(0.0)
    frame = frame.sort_values(["layer", "map_priority", "rank"], ascending=[True, False, True]).reset_index(drop=True)
    frame["y_position"] = list(range(len(frame), 0, -1))
    return frame


def _attach_empty_causal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in [
        "target_id",
        "paired_delta",
        "paired_delta_ci_low",
        "paired_delta_ci_high",
        "kl_divergence_delta",
        "kl_divergence_delta_ci_low",
        "kl_divergence_delta_ci_high",
        "perplexity_shift_delta",
        "top1_change_rate_delta",
        "causal_badge",
    ]:
        if column not in output.columns:
            output[column] = pd.NA
    return output


def _extract_first_feature_id(value: object) -> object:
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        return _extract_first_feature_id(parsed)
    if isinstance(value, (list, tuple)):
        return int(value[0]) if len(value) > 0 else pd.NA
    if hasattr(value, "tolist"):
        as_list = value.tolist()
        if isinstance(as_list, list):
            return int(as_list[0]) if as_list else pd.NA
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    return pd.NA


def _export_policy_feature_map(frame: pd.DataFrame, target: Path, dpi: int) -> None:
    if frame.empty:
        _write_empty_figure(target, title="Policy Feature Map")
        return

    marker_map = {
        "policy_specific": "s",
        "layer_unique": "D",
        "global_dominance": "o",
    }
    proxy_color_map = {
        "privacy": "#2a9d8f",
        "bias": "#e76f51",
        "discrimination": "#f4a261",
        "transparency": "#457b9d",
        "interpretability": "#8d99ae",
        "rights_violation": "#6d597a",
        "unknown": "#bdbdbd",
    }

    display = frame.copy()
    display["layer"] = display["layer"].astype(int)
    layers = sorted(display["layer"].unique().tolist())
    x_positions = {layer: index + 1 for index, layer in enumerate(layers)}
    display["x_position"] = display["layer"].map(x_positions)
    score_min = float(display["score"].min()) if display["score"].notna().any() else 0.0
    score_span = float(display["score"].max() - score_min) if display["score"].notna().any() else 0.0
    if score_span <= 1e-12:
        display["base_size"] = 700.0
    else:
        display["base_size"] = 350.0 + 1150.0 * ((display["score"] - score_min) / score_span)

    causal_max = float(display["causal_magnitude"].max()) if display["causal_magnitude"].notna().any() else 0.0
    if causal_max <= 1e-12:
        display["causal_size"] = 0.0
    else:
        display["causal_size"] = 80.0 + 720.0 * (display["causal_magnitude"] / causal_max)

    figure_height = max(8.0, 0.42 * len(display))
    fig, ax = plt.subplots(figsize=(16, figure_height))

    stage_blocks = display.groupby("layer", sort=True)["y_position"].agg(["min", "max"]).reset_index()
    for block_index, block in stage_blocks.iterrows():
        if block_index % 2 == 0:
            ax.axhspan(float(block["min"]) - 0.5, float(block["max"]) + 0.5, color="#f7f7f7", zorder=0)

    for family, family_frame in display.groupby("primary_ranking_family", dropna=False):
        marker = marker_map.get(str(family), "o")
        edgecolors = [proxy_color_map.get(str(value), proxy_color_map["unknown"]) for value in family_frame["proxy_color_key"]]
        scatter = ax.scatter(
            family_frame["x_position"],
            family_frame["y_position"],
            s=family_frame["base_size"],
            c=family_frame["faithfulness_score"],
            cmap="mako",
            vmin=0.0,
            vmax=1.0,
            marker=marker,
            edgecolors=edgecolors,
            linewidths=2.0,
            alpha=0.92,
            zorder=3,
        )

    causal_rows = display.loc[display["causal_available"] & (display["causal_size"] > 0)].copy()
    if not causal_rows.empty:
        ax.scatter(
            causal_rows["x_position"],
            causal_rows["y_position"],
            s=causal_rows["causal_size"],
            c=causal_rows["paired_delta"],
            cmap="coolwarm",
            vmin=-0.10,
            vmax=0.10,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.95,
            zorder=4,
        )
        for row in causal_rows.itertuples(index=False):
            ax.text(
                float(row.x_position) + 0.12,
                float(row.y_position),
                f"KL {float(row.kl_divergence_delta):+.4f}\nΔM {float(row.paired_delta):+.3f}",
                fontsize=8,
                va="center",
                ha="left",
                color="#222222",
                zorder=5,
            )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([f"Layer {layer}" for layer in layers], fontsize=11)
    ax.set_yticks(display["y_position"].tolist())
    ax.set_yticklabels(display["feature_label"].tolist(), fontsize=9)
    ax.set_xlim(0.6, max(x_positions.values()) + 1.25)
    ax.set_ylim(0.5, float(display["y_position"].max()) + 0.8)
    ax.set_xlabel("Model Layer", fontsize=12)
    ax.set_ylabel("Curated AutoInterp Features", fontsize=12)
    ax.set_title("Policy Feature Map", fontsize=16, pad=14)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.6)

    proxy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="white",
            markeredgecolor=color,
            markersize=9,
            linewidth=0,
            label=proxy_name.replace("_", " "),
        )
        for proxy_name, color in proxy_color_map.items()
        if proxy_name in set(display["proxy_color_key"].astype(str))
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[family_name],
            color="#4d4d4d",
            markerfacecolor="#bfbfbf",
            markeredgecolor="#4d4d4d",
            markersize=9,
            linewidth=0,
            label=family_name.replace("_", " "),
        )
        for family_name in marker_map
        if family_name in set(display["primary_ranking_family"].astype(str))
    ]
    causal_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="black",
        markerfacecolor="#f1c40f",
        markeredgecolor="black",
        markersize=12,
        linewidth=0,
        label="Causal target",
    )
    legend_one = ax.legend(handles=family_handles + [causal_handle], title="Feature type", loc="upper right", frameon=True)
    ax.add_artist(legend_one)
    if proxy_handles:
        ax.legend(handles=proxy_handles, title="Proxy overlay", loc="lower right", frameon=True)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.7)
    colorbar.set_label("AutoInterp faithfulness", fontsize=11)

    plt.tight_layout()
    plt.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close()


def _write_autointerp_feature_gallery(frame: pd.DataFrame, target: Path) -> None:
    if frame.empty:
        target.write_text("# AutoInterp Feature Gallery\n\nNo AutoInterp features available.\n", encoding="utf-8")
        return
    lines: list[str] = ["# AutoInterp Feature Gallery", ""]
    ordered = frame.sort_values(["layer", "faithfulness_score", "simulation_accuracy", "rank"], ascending=[True, False, False, True])
    for layer, layer_rows in ordered.groupby("layer"):
        lines.append(f"## Layer {int(layer)}")
        lines.append("")
        for row in layer_rows.itertuples(index=False):
            name = str(row.feature_name or f"Layer {int(row.layer)} feature {int(row.feature_id)}")
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"Faithfulness score: {float(row.faithfulness_score):.3f}")
            lines.append(f"Ranking family: {row.primary_ranking_family}")
            lines.append(f"Best proxy: {row.best_proxy}")
            lines.append(f"Rationale: {row.activation_hypothesis}")
            lines.append(f"Boundary: {row.boundary_text or 'No boundary text available.'}")
            lines.append("")
    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
