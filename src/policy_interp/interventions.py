"""Causal validation with proxy dependent and proxy free intervention metrics."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from policy_interp.adapters.modeling import HuggingFaceBackboneAdapter, SaeLensAdapter, resolve_transformer_layer
from policy_interp.constants import PAIR_MAP
from policy_interp.feature_matrix import load_residual_matrix
from policy_interp.io import read_parquet, write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import set_seed


PROXY_LABELS = {
    "bias": "bias",
    "discrimination": "discrimination",
    "privacy": "privacy",
    "rights_violation": "rights violation",
    "transparency": "transparency",
    "interpretability": "interpretability",
}

KL_CHUNK_SIZE = 4096
MAX_EXP_NLL = 20.0


@dataclass(slots=True)
class InterventionArtifacts:
    ablation_path: str
    proxy_effects_path: str
    steering_path: str
    feature_summary_path: str
    feature_per_segment_path: str


@dataclass(slots=True)
class AblationTarget:
    target_id: str
    target_name: str
    target_type: str
    primary_proxy: str
    layer: int
    feature_ids: list[int]
    ranking_family: str | None = None
    source_module_id: str | None = None
    source_note: str | None = None


@dataclass(slots=True)
class SequenceBaselineCache:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor
    target_tokens: torch.Tensor
    baseline_logsumexp: torch.Tensor
    baseline_mean_nll: float
    baseline_perplexity: float
    baseline_top1: torch.Tensor


def run_interventions(config: ExperimentConfig) -> InterventionArtifacts:
    set_seed(config.splits.seed)
    segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    stable_modules = _read_optional_parquet(config.run_root / "discovery" / "module_stability.parquet")
    alignment = _read_optional_parquet(config.run_root / "discovery" / "module_proxy_alignment.parquet")
    matches = read_parquet(config.run_root / "matching" / "matched_negatives.parquet")
    sparse_feature_selection = _read_optional_parquet(config.run_root / "baselines" / "sparse_feature_selection.parquet")

    backbone = HuggingFaceBackboneAdapter(config.backbone).load()
    sae_loader = SaeLensAdapter(config.sae)
    intervention_root = config.run_root / "interventions"
    intervention_root.mkdir(parents=True, exist_ok=True)

    existing_summary = _read_optional_parquet(intervention_root / "ablation_sparse_vs_dense.parquet")
    existing_proxy_effects = _read_optional_parquet(intervention_root / "ablation_proxy_effects.parquet")
    existing_steering = _read_optional_parquet(intervention_root / "steering_sensitivity.parquet")
    existing_per_segment = _read_optional_parquet(intervention_root / "feature_causal_per_segment.parquet")
    existing_target_ids = set(existing_summary.get("target_id", pd.Series(dtype=str)).astype(str).tolist())

    summary_rows: list[dict[str, object]] = existing_summary.to_dict(orient="records") if not existing_summary.empty else []
    proxy_effect_rows: list[dict[str, object]] = (
        existing_proxy_effects.to_dict(orient="records") if not existing_proxy_effects.empty else []
    )
    steering_rows: list[dict[str, object]] = existing_steering.to_dict(orient="records") if not existing_steering.empty else []
    per_segment_rows: list[dict[str, object]] = existing_per_segment.to_dict(orient="records") if not existing_per_segment.empty else []
    dense_residual_cache: dict[int, pd.DataFrame] = {}
    feature_pool_cache: dict[int, list[int]] = {}
    sae_cache: dict[int, object] = {}
    steering_done: set[tuple[str, int]] = {
        (str(row.get("primary_proxy")), int(row.get("layer")))
        for row in steering_rows
        if row.get("primary_proxy") is not None and row.get("layer") is not None
    }

    try:
        targets = _build_ablation_targets(config, stable_modules, alignment, sparse_feature_selection)
        for target in targets:
            if target.target_id in existing_target_ids:
                continue
            evaluation_rows = _build_evaluation_rows(segments, matches, target.primary_proxy, config)
            if evaluation_rows.empty:
                continue

            layer = int(target.layer)
            sae = sae_cache.get(layer)
            if sae is None:
                sae = sae_loader.load_for_layer(layer)
                sae_cache[layer] = sae

            dense_residual = dense_residual_cache.get(layer)
            if dense_residual is None:
                dense_residual = load_residual_matrix(config.run_root / "extraction" / f"residual_pool_manifest_layer_{layer}.parquet")
                dense_residual_cache[layer] = dense_residual

            feature_pool = feature_pool_cache.get(layer)
            if feature_pool is None:
                feature_pool = _load_active_feature_pool(config, layer)
                feature_pool_cache[layer] = feature_pool

            random_sets = _sample_random_feature_sets(
                feature_pool=feature_pool,
                exclude_features=target.feature_ids,
                feature_count=len(target.feature_ids),
                trials=config.ablation.random_control_trials,
                seed=config.splits.seed + layer + len(target.feature_ids) + _stable_seed_offset(target.target_id),
            )
            dense_basis = _build_dense_basis(
                proxy=target.primary_proxy,
                dense_residual=dense_residual,
                segments=segments,
                rank=len(target.feature_ids),
            )

            proxy_metrics_by_proxy: dict[str, dict[str, float]] = {}
            evaluated_proxies = _evaluated_proxies(target.primary_proxy, config)
            for evaluated_proxy in evaluated_proxies:
                proxy_metrics = _evaluate_proxy_margin_effects(
                    rows=evaluation_rows,
                    evaluated_proxy=evaluated_proxy,
                    model_bundle=backbone,
                    sae=sae,
                    layer=layer,
                    target_features=target.feature_ids,
                    random_feature_sets=random_sets,
                    dense_basis=dense_basis,
                    config=config,
                )
                proxy_metrics_by_proxy[evaluated_proxy] = proxy_metrics
                proxy_effect_rows.append(
                    {
                        "target_id": target.target_id,
                        "target_name": target.target_name,
                        "target_type": target.target_type,
                        "ranking_family": target.ranking_family or _ranking_family_from_target(target.target_type),
                        "primary_proxy": target.primary_proxy,
                        "evaluated_proxy": evaluated_proxy,
                        "is_target_proxy": evaluated_proxy == target.primary_proxy,
                        "layer": layer,
                        "feature_ids": target.feature_ids,
                        "feature_count": len(target.feature_ids),
                        "source_module_id": target.source_module_id,
                        **proxy_metrics,
                    }
                )

            proxy_free_metrics, target_segment_rows = _evaluate_proxy_free_effects(
                rows=evaluation_rows,
                model_bundle=backbone,
                layer=layer,
                sae=sae,
                target_features=target.feature_ids,
                random_feature_sets=random_sets,
                dense_basis=dense_basis,
                config=config,
            )
            per_segment_rows.extend(
                _attach_target_metadata_to_segment_rows(
                    segment_rows=target_segment_rows,
                    target=target,
                )
            )
            primary_proxy_metrics = proxy_metrics_by_proxy[target.primary_proxy]
            off_target_effects = {
                proxy_name: {
                    "target_margin_drop": proxy_metrics_by_proxy[proxy_name]["target_margin_drop"],
                    "random_control_margin_drop": proxy_metrics_by_proxy[proxy_name]["random_control_margin_drop"],
                    "paired_delta": proxy_metrics_by_proxy[proxy_name]["paired_delta"],
                    "dense_margin_drop": proxy_metrics_by_proxy[proxy_name]["dense_margin_drop"],
                    "dense_paired_delta": proxy_metrics_by_proxy[proxy_name]["dense_paired_delta"],
                }
                for proxy_name in evaluated_proxies
                if proxy_name != target.primary_proxy
            }
            summary_rows.append(
                {
                    "target_id": target.target_id,
                    "target_name": target.target_name,
                    "target_type": target.target_type,
                    "ranking_family": target.ranking_family or _ranking_family_from_target(target.target_type),
                    "primary_proxy": target.primary_proxy,
                    "layer": layer,
                    "feature_ids": target.feature_ids,
                    "feature_count": len(target.feature_ids),
                    "source_module_id": target.source_module_id,
                    "source_note": target.source_note,
                    "evaluation_segment_count": int(len(evaluation_rows)),
                    "target_margin_drop": primary_proxy_metrics["target_margin_drop"],
                    "random_control_margin_drop": primary_proxy_metrics["random_control_margin_drop"],
                    "paired_delta": primary_proxy_metrics["paired_delta"],
                    "paired_delta_ci_low": primary_proxy_metrics["paired_delta_ci_low"],
                    "paired_delta_ci_high": primary_proxy_metrics["paired_delta_ci_high"],
                    "dense_margin_drop": primary_proxy_metrics["dense_margin_drop"],
                    "dense_paired_delta": primary_proxy_metrics["dense_paired_delta"],
                    "dense_paired_delta_ci_low": primary_proxy_metrics["dense_paired_delta_ci_low"],
                    "dense_paired_delta_ci_high": primary_proxy_metrics["dense_paired_delta_ci_high"],
                    "off_target_proxy_effects": json.dumps(off_target_effects, sort_keys=True),
                    **proxy_free_metrics,
                }
            )

            steering_key = (target.primary_proxy, layer)
            if config.steering.enabled and steering_key not in steering_done:
                steering_rows.append(
                    _run_proxy_steering(
                        proxy=target.primary_proxy,
                        layer=layer,
                        selected_features=target.feature_ids,
                        segments=segments,
                        matches=matches,
                        dense_residual=dense_residual,
                        model_bundle=backbone,
                        sae=sae,
                        config=config,
                    )
                )
                steering_done.add(steering_key)
    finally:
        if hasattr(backbone, "model"):
            del backbone.model
        sae_cache.clear()
        dense_residual_cache.clear()
        feature_pool_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ablation_path = intervention_root / "ablation_sparse_vs_dense.parquet"
    proxy_effects_path = intervention_root / "ablation_proxy_effects.parquet"
    steering_path = intervention_root / "steering_sensitivity.parquet"
    feature_summary_path = intervention_root / "feature_causal_summary.parquet"
    feature_per_segment_path = intervention_root / "feature_causal_per_segment.parquet"
    write_parquet(pd.DataFrame(summary_rows), ablation_path)
    write_parquet(pd.DataFrame(proxy_effect_rows), proxy_effects_path)
    write_parquet(pd.DataFrame(steering_rows), steering_path)
    write_parquet(_build_feature_first_summary_frame(summary_rows), feature_summary_path)
    write_parquet(pd.DataFrame(per_segment_rows), feature_per_segment_path)
    return InterventionArtifacts(
        str(ablation_path),
        str(proxy_effects_path),
        str(steering_path),
        str(feature_summary_path),
        str(feature_per_segment_path),
    )


def _attach_target_metadata_to_segment_rows(
    segment_rows: list[dict[str, object]],
    target: AblationTarget,
) -> list[dict[str, object]]:
    annotated_rows: list[dict[str, object]] = []
    target_kind = _target_kind_from_type(target.target_type)
    ranking_family = target.ranking_family or _ranking_family_from_target(target.target_type)
    for row in segment_rows:
        annotated_rows.append(
            {
                "target_id": target.target_id,
                "target_name": target.target_name,
                "target_kind": target_kind,
                "ranking_family": ranking_family,
                "proxy_overlay": target.primary_proxy,
                "primary_proxy": target.primary_proxy,
                "feature_ids": target.feature_ids,
                "feature_count": len(target.feature_ids),
                "source_module_id": target.source_module_id,
                "source_note": target.source_note,
                **row,
            }
        )
    return annotated_rows


def _build_feature_first_summary_frame(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not summary_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(summary_rows).copy()
    frame["target_kind"] = frame["target_type"].astype(str).map(_target_kind_from_type)
    fallback_family = frame["target_type"].astype(str).map(_ranking_family_from_target)
    if "ranking_family" in frame.columns:
        frame["ranking_family"] = frame["ranking_family"].fillna(fallback_family)
    else:
        frame["ranking_family"] = fallback_family
    frame["proxy_overlay"] = frame["primary_proxy"].astype(str)
    frame["causal_badge"] = frame.apply(_feature_causal_badge, axis=1)
    frame["ci"] = frame.apply(
        lambda row: json.dumps(
            {
                "paired_delta": [row["paired_delta_ci_low"], row["paired_delta_ci_high"]],
                "kl_divergence_delta": [row["kl_divergence_delta_ci_low"], row["kl_divergence_delta_ci_high"]],
                "perplexity_shift_delta": [row["perplexity_shift_delta_ci_low"], row["perplexity_shift_delta_ci_high"]],
                "top1_change_rate_delta": [row["top1_change_rate_delta_ci_low"], row["top1_change_rate_delta_ci_high"]],
            },
            sort_keys=True,
        ),
        axis=1,
    )
    frame["dense_controls"] = frame.apply(
        lambda row: json.dumps(
            {
                "dense_margin_drop": row["dense_margin_drop"],
                "dense_paired_delta": row["dense_paired_delta"],
                "dense_kl_divergence_delta": row["dense_kl_divergence_delta"],
                "dense_perplexity_shift_delta": row["dense_perplexity_shift_delta"],
                "dense_top1_change_rate_delta": row["dense_top1_change_rate_delta"],
            },
            sort_keys=True,
        ),
        axis=1,
    )
    frame["random_controls"] = frame.apply(
        lambda row: json.dumps(
            {
                "random_control_margin_drop": row["random_control_margin_drop"],
                "random_mean_kl_divergence": row["random_mean_kl_divergence"],
                "random_mean_perplexity_shift": row["random_mean_perplexity_shift"],
                "random_mean_top1_change_rate": row["random_mean_top1_change_rate"],
            },
            sort_keys=True,
        ),
        axis=1,
    )
    return frame


def _target_kind_from_type(target_type: str) -> str:
    mapping = {
        "stable_module_whole": "module_whole",
        "individual_probe_topk": "feature_topk",
        "autointerp_single_feature": "single_feature",
        "autointerp_feature_set": "feature_set",
    }
    return mapping.get(target_type, target_type)


def _ranking_family_from_target(target_type: str) -> str:
    if "module" in target_type:
        return "module"
    if "individual" in target_type:
        return "individual_probe"
    if "feature_set" in target_type:
        return "autointerp_set"
    if "autointerp" in target_type:
        return "autointerp"
    return "unknown"


def _feature_causal_badge(row: pd.Series) -> str:
    if float(row.get("kl_divergence_delta_ci_low", np.nan)) > 0:
        return "proxy_free_causal"
    if float(row.get("paired_delta_ci_low", np.nan)) > 0:
        return "proxy_selective"
    return "observed_target"


def _build_ablation_targets(
    config: ExperimentConfig,
    stable_modules: pd.DataFrame,
    alignment: pd.DataFrame,
    sparse_feature_selection: pd.DataFrame,
) -> list[AblationTarget]:
    targets: list[AblationTarget] = []

    privacy_module = _select_proxy_module("privacy", stable_modules, alignment)
    if privacy_module is not None:
        targets.append(
            AblationTarget(
                target_id="privacy_module_whole",
                target_name="Privacy stable module whole",
                target_type="stable_module_whole",
                primary_proxy="privacy",
                layer=int(privacy_module["layer"]),
                feature_ids=list(map(int, privacy_module["feature_ids"])),
                ranking_family="module",
                source_module_id=str(privacy_module["stable_module_id"]),
                source_note="Best privacy aligned stable module",
            )
        )

    privacy_probe_top = _select_probe_top_features("privacy", sparse_feature_selection, config.ablation.individual_probe_top_k)
    if privacy_probe_top:
        targets.append(
            AblationTarget(
                target_id="privacy_individual_top3",
                target_name="Privacy individual probe top 3",
                target_type="individual_probe_topk",
                primary_proxy="privacy",
                layer=int(_selected_probe_layer("privacy", sparse_feature_selection)),
                feature_ids=privacy_probe_top,
                ranking_family="individual_probe",
                source_note="Top individual sparse probe features for privacy",
            )
        )

    bias_probe_top = _select_probe_top_features("bias", sparse_feature_selection, config.ablation.individual_probe_top_k)
    if bias_probe_top:
        targets.append(
            AblationTarget(
                target_id="bias_individual_top3",
                target_name="Bias individual probe top 3",
                target_type="individual_probe_topk",
                primary_proxy="bias",
                layer=int(_selected_probe_layer("bias", sparse_feature_selection)),
                feature_ids=bias_probe_top,
                ranking_family="individual_probe",
                source_note="Top individual sparse probe features for bias",
            )
        )

    bias_module = _select_proxy_module("bias", stable_modules, alignment)
    if bias_module is not None:
        targets.append(
            AblationTarget(
                target_id="bias_module_whole",
                target_name="Bias stable module whole",
                target_type="stable_module_whole",
                primary_proxy="bias",
                layer=int(bias_module["layer"]),
                feature_ids=list(map(int, bias_module["feature_ids"])),
                ranking_family="module",
                source_module_id=str(bias_module["stable_module_id"]),
                source_note="Best bias aligned stable module",
            )
        )

    if config.ablation.include_autointerp_single_feature_targets:
        targets.extend(_select_autointerp_single_feature_targets(config))
    if config.ablation.include_autointerp_feature_set_targets:
        targets.extend(_select_autointerp_feature_set_targets(config))

    return targets


def _select_autointerp_single_feature_targets(config: ExperimentConfig) -> list[AblationTarget]:
    scores_path = config.run_root / "features" / "autointerp" / "autointerp_feature_scores.parquet"
    if not scores_path.exists():
        return []
    scores = read_parquet(scores_path)
    if scores.empty:
        return []

    candidate_families = set(config.ablation.autointerp_target_ranking_families)
    filtered = scores.loc[scores["primary_ranking_family"].isin(candidate_families)].copy()
    if filtered.empty:
        return []

    selected_frames: list[pd.DataFrame] = []
    top_per_layer = max(1, int(config.ablation.autointerp_top_features_per_layer))
    min_faithfulness = float(config.ablation.autointerp_min_faithfulness)
    for layer, layer_frame in filtered.groupby("layer", sort=True):
        ranked = layer_frame.sort_values(
            ["faithfulness_score", "simulation_accuracy", "rank"],
            ascending=[False, False, True],
        ).copy()
        if min_faithfulness > 0:
            qualified = ranked.loc[ranked["faithfulness_score"] >= min_faithfulness].copy()
        else:
            qualified = ranked.copy()
        chosen = qualified.head(top_per_layer) if not qualified.empty else ranked.head(top_per_layer)
        if not chosen.empty:
            selected_frames.append(chosen)

    if not selected_frames:
        return []

    selected = pd.concat(selected_frames, ignore_index=True)
    targets: list[AblationTarget] = []
    for row in selected.itertuples(index=False):
        target_proxy = _normalize_target_proxy(getattr(row, "best_proxy", None), config)
        feature_name = str(getattr(row, "feature_name", f"Feature {int(row.feature_id)}")).strip() or f"Feature {int(row.feature_id)}"
        source_note = (
            f"AutoInterp faithfulness={float(getattr(row, 'faithfulness_score', np.nan)):.3f}; "
            f"accuracy={float(getattr(row, 'simulation_accuracy', np.nan)):.3f}; "
            f"best_proxy={target_proxy}"
        )
        targets.append(
            AblationTarget(
                target_id=f"autointerp_layer_{int(row.layer)}_feature_{int(row.feature_id)}",
                target_name=f"AutoInterp layer {int(row.layer)} {feature_name}",
                target_type="autointerp_single_feature",
                primary_proxy=target_proxy,
                layer=int(row.layer),
                feature_ids=[int(row.feature_id)],
                ranking_family=str(row.primary_ranking_family),
                source_note=source_note,
            )
        )
    return targets


def _select_autointerp_feature_set_targets(config: ExperimentConfig) -> list[AblationTarget]:
    scores_path = config.run_root / "features" / "autointerp" / "autointerp_feature_scores.parquet"
    if not scores_path.exists():
        return []
    scores = read_parquet(scores_path)
    if scores.empty:
        return []

    candidate_families = set(config.ablation.autointerp_target_ranking_families)
    filtered = scores.loc[scores["primary_ranking_family"].isin(candidate_families)].copy()
    if filtered.empty:
        return []

    targets: list[AblationTarget] = []
    min_faithfulness = float(config.ablation.autointerp_min_faithfulness)
    set_sizes = sorted({int(size) for size in config.ablation.autointerp_feature_set_sizes if int(size) > 1})
    for layer, layer_frame in filtered.groupby("layer", sort=True):
        ranked = layer_frame.sort_values(
            ["faithfulness_score", "simulation_accuracy", "rank"],
            ascending=[False, False, True],
        ).copy()
        if min_faithfulness > 0:
            qualified = ranked.loc[ranked["faithfulness_score"] >= min_faithfulness].copy()
            if not qualified.empty:
                ranked = qualified
        ranked = ranked.drop_duplicates(["feature_id"])
        if ranked.empty:
            continue
        for feature_count in set_sizes:
            selected = ranked.head(feature_count).copy()
            if len(selected) < feature_count:
                continue
            feature_ids = [int(value) for value in selected["feature_id"].tolist()]
            primary_proxy = _select_feature_set_proxy(selected, config)
            feature_names = [
                str(name).strip()
                for name in selected.get("feature_name", pd.Series(dtype=str)).tolist()
                if str(name).strip()
            ]
            rendered_name = ", ".join(feature_names[:3]) if feature_names else f"top {feature_count} features"
            source_note = (
                f"AutoInterp top-{feature_count} faithful set; "
                f"mean_faithfulness={float(selected['faithfulness_score'].mean()):.3f}; "
                f"mean_accuracy={float(selected['simulation_accuracy'].mean()):.3f}; "
                f"primary_proxy={primary_proxy}; "
                f"members={rendered_name}"
            )
            targets.append(
                AblationTarget(
                    target_id=f"autointerp_layer_{int(layer)}_top_{feature_count}",
                    target_name=f"AutoInterp layer {int(layer)} top {feature_count} faithful features",
                    target_type="autointerp_feature_set",
                    primary_proxy=primary_proxy,
                    layer=int(layer),
                    feature_ids=feature_ids,
                    ranking_family="autointerp_set",
                    source_note=source_note,
                )
            )
    return targets


def _select_feature_set_proxy(selected: pd.DataFrame, config: ExperimentConfig) -> str:
    if selected.empty:
        return str(config.ablation.target_proxies[0])
    proxy_votes = (
        selected["best_proxy"]
        .astype(str)
        .map(lambda proxy_name: proxy_name if proxy_name in PROXY_LABELS else str(config.ablation.target_proxies[0]))
        .value_counts()
    )
    if proxy_votes.empty:
        return str(config.ablation.target_proxies[0])
    return str(proxy_votes.index[0])


def _normalize_target_proxy(proxy: object, config: ExperimentConfig) -> str:
    proxy_name = str(proxy).strip() if proxy is not None else ""
    if proxy_name in PROXY_LABELS:
        return proxy_name
    return str(config.ablation.target_proxies[0])


def _selected_probe_layer(proxy: str, sparse_feature_selection: pd.DataFrame) -> int:
    required_columns = {"proxy", "selected_layer"}
    if not required_columns.issubset(set(sparse_feature_selection.columns)):
        return 0
    row = sparse_feature_selection.loc[sparse_feature_selection["proxy"] == proxy].iloc[0]
    return int(row["selected_layer"])


def _select_proxy_module(
    proxy: str,
    stable_modules: pd.DataFrame,
    alignment: pd.DataFrame,
) -> dict[str, object] | None:
    required_module_columns = {"stable_module_id", "layer", "stable", "feature_ids"}
    required_alignment_columns = {"stable_module_id", "proxy", "dev_auc", "test_auc"}
    if not required_module_columns.issubset(set(stable_modules.columns)):
        return None
    if not required_alignment_columns.issubset(set(alignment.columns)):
        return None
    stable_only = stable_modules.loc[stable_modules["stable"]].copy()
    if stable_only.empty:
        return None
    valid_ids = set(stable_only["stable_module_id"].astype(str).tolist())
    scored = alignment.loc[
        (alignment["proxy"] == proxy) & (alignment["stable_module_id"].astype(str).isin(valid_ids))
    ].sort_values(["dev_auc", "test_auc"], ascending=[False, False])
    if scored.empty:
        return None
    module_id = str(scored.iloc[0]["stable_module_id"])
    module_row = stable_only.loc[stable_only["stable_module_id"] == module_id].iloc[0]
    return module_row.to_dict()


def _select_probe_top_features(proxy: str, sparse_feature_selection: pd.DataFrame, top_k: int) -> list[int]:
    required_columns = {"proxy", "selected_feature_ids"}
    if not required_columns.issubset(set(sparse_feature_selection.columns)):
        return []
    rows = sparse_feature_selection.loc[sparse_feature_selection["proxy"] == proxy]
    if rows.empty:
        return []
    feature_ids = rows.iloc[0]["selected_feature_ids"]
    if feature_ids is None or (isinstance(feature_ids, float) and math.isnan(feature_ids)):
        return []
    return list(map(int, list(feature_ids)[:top_k]))


def _read_optional_parquet(path: object) -> pd.DataFrame:
    parquet_path = pd.io.common.stringify_path(path)
    if not Path(parquet_path).exists():
        return pd.DataFrame()
    return read_parquet(path)


def _build_evaluation_rows(
    segments: pd.DataFrame,
    matches: pd.DataFrame,
    proxy: str,
    config: ExperimentConfig,
    eval_split: str | None = None,
) -> pd.DataFrame:
    split_name = eval_split or config.ablation.evaluation_split
    if config.ablation.use_matched_negative_evaluation:
        match_frame = matches.loc[(matches["proxy"] == proxy) & (matches["split"] == split_name)].copy()
        if not match_frame.empty:
            ordered_ids = list(
                dict.fromkeys(
                    match_frame["positive_segment_id"].astype(str).tolist()
                    + match_frame["negative_segment_id"].astype(str).tolist()
                )
            )
            return segments.set_index("segment_id").loc[ordered_ids].reset_index()
    return segments.loc[segments["split"] == split_name].copy().reset_index(drop=True)


def _evaluated_proxies(primary_proxy: str, config: ExperimentConfig) -> list[str]:
    ordered = [primary_proxy]
    for proxy in config.ablation.target_proxies:
        if proxy != primary_proxy and proxy not in ordered:
            ordered.append(proxy)
    return ordered


def _load_active_feature_pool(config: ExperimentConfig, layer: int) -> list[int]:
    feature_summary = read_parquet(config.run_root / "extraction" / f"feature_summary_layer_{layer}.parquet")
    active = feature_summary.loc[feature_summary["activation_count"] > 0, "feature_id"].astype(int).tolist()
    return sorted(set(active))


def _sample_random_feature_sets(
    feature_pool: list[int],
    exclude_features: list[int],
    feature_count: int,
    trials: int,
    seed: int,
) -> list[list[int]]:
    if feature_count <= 0 or trials <= 0:
        return []
    excluded = set(exclude_features)
    available = [feature_id for feature_id in feature_pool if feature_id not in excluded]
    if len(available) < feature_count:
        available = list(feature_pool)
    rng = np.random.default_rng(seed)
    random_sets: list[list[int]] = []
    for _ in range(trials):
        sampled = rng.choice(available, size=feature_count, replace=False).tolist()
        random_sets.append(list(map(int, sampled)))
    return random_sets


def _stable_seed_offset(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _evaluate_proxy_margin_effects(
    rows: pd.DataFrame,
    evaluated_proxy: str,
    model_bundle: object,
    sae: object,
    layer: int,
    target_features: list[int],
    random_feature_sets: list[list[int]],
    dense_basis: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float]:
    pair_proxy = PAIR_MAP[evaluated_proxy]
    target_drops: list[float] = []
    random_mean_drops: list[float] = []
    paired_deltas: list[float] = []
    dense_drops: list[float] = []
    dense_deltas: list[float] = []

    for record in rows.itertuples(index=False):
        prompt = build_forced_choice_prompt(str(record.text), evaluated_proxy, pair_proxy)
        baseline_margin = margin_for_prompt(model_bundle, prompt, evaluated_proxy, pair_proxy)
        target_margin = margin_for_prompt(
            model_bundle,
            prompt,
            evaluated_proxy,
            pair_proxy,
            editor=lambda hidden, features=target_features: sparse_ablation_editor(hidden, sae, features),
            layer=layer,
        )
        target_drop = baseline_margin - target_margin

        random_drops_for_text: list[float] = []
        for random_features in random_feature_sets:
            random_margin = margin_for_prompt(
                model_bundle,
                prompt,
                evaluated_proxy,
                pair_proxy,
                editor=lambda hidden, features=random_features: sparse_ablation_editor(hidden, sae, features),
                layer=layer,
            )
            random_drops_for_text.append(baseline_margin - random_margin)
        random_mean_drop = float(np.mean(random_drops_for_text)) if random_drops_for_text else np.nan

        dense_margin = margin_for_prompt(
            model_bundle,
            prompt,
            evaluated_proxy,
            pair_proxy,
            editor=lambda hidden: dense_subspace_editor(hidden, dense_basis),
            layer=layer,
        )
        dense_drop = baseline_margin - dense_margin

        target_drops.append(target_drop)
        random_mean_drops.append(random_mean_drop)
        paired_deltas.append(target_drop - random_mean_drop)
        dense_drops.append(dense_drop)
        dense_deltas.append(dense_drop - random_mean_drop)

    paired_ci = bootstrap_ci(paired_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    dense_ci = bootstrap_ci(dense_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    return {
        "target_margin_drop": float(np.mean(target_drops)) if target_drops else np.nan,
        "random_control_margin_drop": float(np.mean(random_mean_drops)) if random_mean_drops else np.nan,
        "paired_delta": float(np.mean(paired_deltas)) if paired_deltas else np.nan,
        "paired_delta_ci_low": paired_ci[0],
        "paired_delta_ci_high": paired_ci[1],
        "dense_margin_drop": float(np.mean(dense_drops)) if dense_drops else np.nan,
        "dense_paired_delta": float(np.mean(dense_deltas)) if dense_deltas else np.nan,
        "dense_paired_delta_ci_low": dense_ci[0],
        "dense_paired_delta_ci_high": dense_ci[1],
    }


def _evaluate_proxy_free_effects(
    rows: pd.DataFrame,
    model_bundle: object,
    layer: int,
    sae: object,
    target_features: list[int],
    random_feature_sets: list[list[int]],
    dense_basis: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    target_kl_values: list[float] = []
    random_kl_means: list[float] = []
    dense_kl_values: list[float] = []
    kl_deltas: list[float] = []
    dense_kl_deltas: list[float] = []

    target_perplexity_values: list[float] = []
    random_perplexity_means: list[float] = []
    dense_perplexity_values: list[float] = []
    perplexity_deltas: list[float] = []
    dense_perplexity_deltas: list[float] = []

    target_nll_values: list[float] = []
    random_nll_means: list[float] = []
    dense_nll_values: list[float] = []
    nll_deltas: list[float] = []
    dense_nll_deltas: list[float] = []

    target_top1_values: list[float] = []
    random_top1_means: list[float] = []
    dense_top1_values: list[float] = []
    top1_deltas: list[float] = []
    dense_top1_deltas: list[float] = []
    per_segment_rows: list[dict[str, object]] = []

    for record in rows.itertuples(index=False):
        baseline_cache = _build_sequence_baseline_cache(model_bundle, str(record.text), max_length=config.backbone.max_length)
        if baseline_cache is None:
            continue
        try:
            target_metrics = _compute_sequence_change_metrics(
                model_bundle=model_bundle,
                baseline_cache=baseline_cache,
                editor=lambda hidden, features=target_features: sparse_ablation_editor(hidden, sae, features),
                layer=layer,
            )
            random_metrics = [
                _compute_sequence_change_metrics(
                    model_bundle=model_bundle,
                    baseline_cache=baseline_cache,
                    editor=lambda hidden, features=random_features: sparse_ablation_editor(hidden, sae, features),
                    layer=layer,
                )
                for random_features in random_feature_sets
            ]
            dense_metrics = _compute_sequence_change_metrics(
                model_bundle=model_bundle,
                baseline_cache=baseline_cache,
                editor=lambda hidden: dense_subspace_editor(hidden, dense_basis),
                layer=layer,
            )
        finally:
            del baseline_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        random_means = _mean_metric_dicts(random_metrics)

        target_kl_values.append(target_metrics["kl_divergence"])
        random_kl_means.append(random_means["kl_divergence"])
        dense_kl_values.append(dense_metrics["kl_divergence"])
        kl_deltas.append(target_metrics["kl_divergence"] - random_means["kl_divergence"])
        dense_kl_deltas.append(dense_metrics["kl_divergence"] - random_means["kl_divergence"])

        target_perplexity_values.append(target_metrics["perplexity_shift"])
        random_perplexity_means.append(random_means["perplexity_shift"])
        dense_perplexity_values.append(dense_metrics["perplexity_shift"])
        perplexity_deltas.append(target_metrics["perplexity_shift"] - random_means["perplexity_shift"])
        dense_perplexity_deltas.append(dense_metrics["perplexity_shift"] - random_means["perplexity_shift"])

        target_nll_values.append(target_metrics["mean_nll_shift"])
        random_nll_means.append(random_means["mean_nll_shift"])
        dense_nll_values.append(dense_metrics["mean_nll_shift"])
        nll_deltas.append(target_metrics["mean_nll_shift"] - random_means["mean_nll_shift"])
        dense_nll_deltas.append(dense_metrics["mean_nll_shift"] - random_means["mean_nll_shift"])

        target_top1_values.append(target_metrics["top1_change_rate"])
        random_top1_means.append(random_means["top1_change_rate"])
        dense_top1_values.append(dense_metrics["top1_change_rate"])
        top1_deltas.append(target_metrics["top1_change_rate"] - random_means["top1_change_rate"])
        dense_top1_deltas.append(dense_metrics["top1_change_rate"] - random_means["top1_change_rate"])

        per_segment_rows.append(
            {
                "segment_id": str(record.segment_id),
                "document_id": int(record.document_id),
                "split": str(record.split),
                "layer": int(layer),
                "target_mean_kl_divergence": target_metrics["kl_divergence"],
                "random_mean_kl_divergence": random_means["kl_divergence"],
                "kl_divergence_delta": target_metrics["kl_divergence"] - random_means["kl_divergence"],
                "dense_mean_kl_divergence": dense_metrics["kl_divergence"],
                "dense_kl_divergence_delta": dense_metrics["kl_divergence"] - random_means["kl_divergence"],
                "target_mean_perplexity_shift": target_metrics["perplexity_shift"],
                "random_mean_perplexity_shift": random_means["perplexity_shift"],
                "perplexity_shift_delta": target_metrics["perplexity_shift"] - random_means["perplexity_shift"],
                "dense_mean_perplexity_shift": dense_metrics["perplexity_shift"],
                "dense_perplexity_shift_delta": dense_metrics["perplexity_shift"] - random_means["perplexity_shift"],
                "target_mean_nll_shift": target_metrics["mean_nll_shift"],
                "random_mean_nll_shift": random_means["mean_nll_shift"],
                "nll_shift_delta": target_metrics["mean_nll_shift"] - random_means["mean_nll_shift"],
                "dense_mean_nll_shift": dense_metrics["mean_nll_shift"],
                "dense_nll_shift_delta": dense_metrics["mean_nll_shift"] - random_means["mean_nll_shift"],
                "target_mean_top1_change_rate": target_metrics["top1_change_rate"],
                "random_mean_top1_change_rate": random_means["top1_change_rate"],
                "top1_change_rate_delta": target_metrics["top1_change_rate"] - random_means["top1_change_rate"],
                "dense_mean_top1_change_rate": dense_metrics["top1_change_rate"],
                "dense_top1_change_rate_delta": dense_metrics["top1_change_rate"] - random_means["top1_change_rate"],
            }
        )

    kl_ci = bootstrap_ci(kl_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    dense_kl_ci = bootstrap_ci(dense_kl_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    perplexity_ci = bootstrap_ci(perplexity_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    dense_perplexity_ci = bootstrap_ci(dense_perplexity_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    nll_ci = bootstrap_ci(nll_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    dense_nll_ci = bootstrap_ci(dense_nll_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    top1_ci = bootstrap_ci(top1_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    dense_top1_ci = bootstrap_ci(dense_top1_deltas, config.ablation.bootstrap_iterations, config.ablation.ci_level)

    return {
        "target_mean_kl_divergence": float(np.mean(target_kl_values)) if target_kl_values else np.nan,
        "random_mean_kl_divergence": float(np.mean(random_kl_means)) if random_kl_means else np.nan,
        "kl_divergence_delta": float(np.mean(kl_deltas)) if kl_deltas else np.nan,
        "kl_divergence_delta_ci_low": kl_ci[0],
        "kl_divergence_delta_ci_high": kl_ci[1],
        "dense_mean_kl_divergence": float(np.mean(dense_kl_values)) if dense_kl_values else np.nan,
        "dense_kl_divergence_delta": float(np.mean(dense_kl_deltas)) if dense_kl_deltas else np.nan,
        "dense_kl_divergence_delta_ci_low": dense_kl_ci[0],
        "dense_kl_divergence_delta_ci_high": dense_kl_ci[1],
        "target_mean_perplexity_shift": float(np.mean(target_perplexity_values)) if target_perplexity_values else np.nan,
        "random_mean_perplexity_shift": float(np.mean(random_perplexity_means)) if random_perplexity_means else np.nan,
        "perplexity_shift_delta": float(np.mean(perplexity_deltas)) if perplexity_deltas else np.nan,
        "perplexity_shift_delta_ci_low": perplexity_ci[0],
        "perplexity_shift_delta_ci_high": perplexity_ci[1],
        "dense_mean_perplexity_shift": float(np.mean(dense_perplexity_values)) if dense_perplexity_values else np.nan,
        "dense_perplexity_shift_delta": float(np.mean(dense_perplexity_deltas)) if dense_perplexity_deltas else np.nan,
        "dense_perplexity_shift_delta_ci_low": dense_perplexity_ci[0],
        "dense_perplexity_shift_delta_ci_high": dense_perplexity_ci[1],
        "target_mean_nll_shift": float(np.mean(target_nll_values)) if target_nll_values else np.nan,
        "random_mean_nll_shift": float(np.mean(random_nll_means)) if random_nll_means else np.nan,
        "nll_shift_delta": float(np.mean(nll_deltas)) if nll_deltas else np.nan,
        "nll_shift_delta_ci_low": nll_ci[0],
        "nll_shift_delta_ci_high": nll_ci[1],
        "dense_mean_nll_shift": float(np.mean(dense_nll_values)) if dense_nll_values else np.nan,
        "dense_nll_shift_delta": float(np.mean(dense_nll_deltas)) if dense_nll_deltas else np.nan,
        "dense_nll_shift_delta_ci_low": dense_nll_ci[0],
        "dense_nll_shift_delta_ci_high": dense_nll_ci[1],
        "target_mean_top1_change_rate": float(np.mean(target_top1_values)) if target_top1_values else np.nan,
        "random_mean_top1_change_rate": float(np.mean(random_top1_means)) if random_top1_means else np.nan,
        "top1_change_rate_delta": float(np.mean(top1_deltas)) if top1_deltas else np.nan,
        "top1_change_rate_delta_ci_low": top1_ci[0],
        "top1_change_rate_delta_ci_high": top1_ci[1],
        "dense_mean_top1_change_rate": float(np.mean(dense_top1_values)) if dense_top1_values else np.nan,
        "dense_top1_change_rate_delta": float(np.mean(dense_top1_deltas)) if dense_top1_deltas else np.nan,
        "dense_top1_change_rate_delta_ci_low": dense_top1_ci[0],
        "dense_top1_change_rate_delta_ci_high": dense_top1_ci[1],
    }, per_segment_rows


def _mean_metric_dicts(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {
            "kl_divergence": np.nan,
            "perplexity_shift": np.nan,
            "mean_nll_shift": np.nan,
            "top1_change_rate": np.nan,
        }
    frame = pd.DataFrame(metric_rows)
    return {
        "kl_divergence": float(frame["kl_divergence"].mean()),
        "perplexity_shift": float(frame["perplexity_shift"].mean()),
        "mean_nll_shift": float(frame["mean_nll_shift"].mean()),
        "top1_change_rate": float(frame["top1_change_rate"].mean()),
    }


def _build_sequence_baseline_cache(
    model_bundle: object,
    text: str,
    max_length: int,
) -> SequenceBaselineCache | None:
    tokenizer = model_bundle.tokenizer
    device = model_bundle.device
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if input_ids.shape[1] < 2:
        return None
    with torch.inference_mode():
        logits = model_bundle.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].detach()
    target_tokens = input_ids[:, 1:].detach()
    baseline_logsumexp = torch.logsumexp(logits.float(), dim=-1)
    baseline_token_logits = torch.gather(
        logits.float(),
        dim=-1,
        index=target_tokens.unsqueeze(-1),
    ).squeeze(-1)
    baseline_token_log_probs = baseline_token_logits - baseline_logsumexp
    baseline_mean_nll = float((-baseline_token_log_probs.mean()).detach().cpu().item())
    baseline_perplexity = _nll_to_perplexity(baseline_mean_nll)
    baseline_top1 = logits.argmax(dim=-1).detach()
    return SequenceBaselineCache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits=logits,
        target_tokens=target_tokens,
        baseline_logsumexp=baseline_logsumexp.detach(),
        baseline_mean_nll=baseline_mean_nll,
        baseline_perplexity=baseline_perplexity,
        baseline_top1=baseline_top1,
    )


def _compute_sequence_change_metrics(
    model_bundle: object,
    baseline_cache: SequenceBaselineCache,
    editor: object,
    layer: int,
) -> dict[str, float]:
    with maybe_layer_editor(model_bundle.model, layer, editor):
        with torch.inference_mode():
            edited_logits = model_bundle.model(
                input_ids=baseline_cache.input_ids,
                attention_mask=baseline_cache.attention_mask,
            ).logits[:, :-1, :].detach()

    try:
        kl_divergence = _mean_token_kl_divergence(baseline_cache.logits, edited_logits)
        edited_logsumexp = torch.logsumexp(edited_logits.float(), dim=-1)
        edited_token_logits = torch.gather(
            edited_logits.float(),
            dim=-1,
            index=baseline_cache.target_tokens.unsqueeze(-1),
        ).squeeze(-1)
        edited_token_log_probs = edited_token_logits - edited_logsumexp
        edited_mean_nll = float((-edited_token_log_probs.mean()).detach().cpu().item())
        mean_nll_shift = edited_mean_nll - baseline_cache.baseline_mean_nll
        edited_perplexity = _nll_to_perplexity(edited_mean_nll)
        perplexity_shift = edited_perplexity - baseline_cache.baseline_perplexity
        edited_top1 = edited_logits.argmax(dim=-1)
        top1_change_rate = float((baseline_cache.baseline_top1 != edited_top1).float().mean().detach().cpu().item())
    finally:
        del edited_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "kl_divergence": kl_divergence,
        "perplexity_shift": perplexity_shift,
        "mean_nll_shift": mean_nll_shift,
        "top1_change_rate": top1_change_rate,
    }


def _nll_to_perplexity(mean_nll: float) -> float:
    return float(math.exp(min(float(mean_nll), MAX_EXP_NLL)))


def _mean_token_kl_divergence(baseline_logits: torch.Tensor, edited_logits: torch.Tensor) -> float:
    baseline_logsumexp = torch.logsumexp(baseline_logits.float(), dim=-1, keepdim=True)
    edited_logsumexp = torch.logsumexp(edited_logits.float(), dim=-1, keepdim=True)
    vocab_size = baseline_logits.shape[-1]
    kl_values = torch.zeros(baseline_logits.shape[:-1], dtype=torch.float32, device=baseline_logits.device)
    for start in range(0, vocab_size, KL_CHUNK_SIZE):
        end = min(vocab_size, start + KL_CHUNK_SIZE)
        base_chunk = baseline_logits[..., start:end].float()
        edited_chunk = edited_logits[..., start:end].float()
        base_log_probs = base_chunk - baseline_logsumexp
        edited_log_probs = edited_chunk - edited_logsumexp
        base_probs = torch.exp(base_log_probs)
        kl_values = kl_values + (base_probs * (base_log_probs - edited_log_probs)).sum(dim=-1)
    return float(kl_values.mean().detach().cpu().item())


def _build_dense_basis(
    proxy: str,
    dense_residual: pd.DataFrame,
    segments: pd.DataFrame,
    rank: int,
) -> np.ndarray:
    if dense_residual.empty or rank <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    dim = int(np.asarray(dense_residual.iloc[0]["vector"]).shape[0])
    merged = dense_residual.merge(segments[["segment_id", proxy]], on="segment_id", how="left")
    train = merged.loc[merged["split"] == "train"].copy()
    if train.empty:
        return np.zeros((0, dim), dtype=np.float32)

    matrix = np.nan_to_num(
        np.vstack(train["vector"].tolist()).astype(np.float32),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    max_rank = min(rank, matrix.shape[0], matrix.shape[1])
    if max_rank <= 0:
        return np.zeros((0, dim), dtype=np.float32)

    positives = train.loc[train[proxy].fillna(False)]
    negatives = train.loc[~train[proxy].fillna(False)]
    pca = PCA(n_components=max_rank)
    pca.fit(matrix)

    pos_mean = (
        np.nan_to_num(np.vstack(positives["vector"].tolist()).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6).mean(axis=0)
        if not positives.empty
        else matrix.mean(axis=0)
    )
    neg_mean = (
        np.nan_to_num(np.vstack(negatives["vector"].tolist()).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6).mean(axis=0)
        if not negatives.empty
        else matrix.mean(axis=0)
    )
    dense_weight = pos_mean - neg_mean
    projections = np.abs(pca.components_ @ dense_weight)
    top_indices = projections.argsort()[::-1][:max_rank]
    return pca.components_[top_indices].astype(np.float32)


def build_forced_choice_prompt(text: str, proxy: str, contrast_proxy: str) -> str:
    return (
        "Passage:\n"
        f"{text}\n\n"
        "Question: Which proxy is more strongly reflected in the passage?\n"
        f"A. {PROXY_LABELS[proxy]}\n"
        f"B. {PROXY_LABELS[contrast_proxy]}\n"
        "Answer:"
    )


def margin_for_prompt(
    model_bundle: object,
    prompt: str,
    proxy: str,
    contrast_proxy: str,
    editor: object | None = None,
    layer: int | None = None,
) -> float:
    target = " " + PROXY_LABELS[proxy]
    contrast = " " + PROXY_LABELS[contrast_proxy]
    with maybe_layer_editor(model_bundle.model, layer, editor):
        target_logp = continuation_logprob(model_bundle, prompt, target)
        contrast_logp = continuation_logprob(model_bundle, prompt, contrast)
    return float(target_logp - contrast_logp)


def continuation_logprob(model_bundle: object, prompt: str, continuation: str) -> float:
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = model_bundle.device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        logits = model(full_ids).logits[:, :-1, :]
    continuation_length = full_ids.shape[1] - prompt_ids.shape[1]
    if continuation_length <= 0:
        return 0.0
    start = prompt_ids.shape[1] - 1
    end = start + continuation_length
    target_tokens = full_ids[:, prompt_ids.shape[1] : full_ids.shape[1]]
    target_logits = logits[:, start:end, :]
    log_probs = torch.log_softmax(target_logits.float(), dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().detach().cpu().item())


@contextmanager
def maybe_layer_editor(model: object, layer: int | None, editor: object | None):
    if layer is None or editor is None:
        yield
        return
    target_layer = resolve_transformer_layer(model, layer)

    def hook(_module: object, _inputs: tuple[object, ...], output: object) -> object:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = editor(hidden)
        if isinstance(edited, torch.Tensor):
            edited = edited.to(dtype=hidden.dtype, device=hidden.device)
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    handle = target_layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def sparse_ablation_editor(hidden: torch.Tensor, sae: object, feature_ids: list[int]) -> torch.Tensor:
    latents = sae.encode(hidden)
    latents[..., feature_ids] = 0.0
    return sae.decode(latents)


def dense_subspace_editor(hidden: torch.Tensor, basis: np.ndarray) -> torch.Tensor:
    if basis.size == 0:
        return hidden
    basis_tensor = torch.as_tensor(basis, dtype=hidden.dtype, device=hidden.device)
    if basis_tensor.ndim == 1:
        basis_tensor = basis_tensor.unsqueeze(0)
    projection = torch.einsum("btd,kd->btk", hidden, basis_tensor)
    reconstruction = torch.einsum("btk,kd->btd", projection, basis_tensor)
    return hidden - reconstruction


def steering_editor(hidden: torch.Tensor, direction: np.ndarray, alpha: float) -> torch.Tensor:
    if direction.size == 0 or alpha == 0.0:
        return hidden
    direction_tensor = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
    return hidden + (alpha * direction_tensor.view(1, 1, -1))


def _run_proxy_steering(
    proxy: str,
    layer: int,
    selected_features: list[int],
    segments: pd.DataFrame,
    matches: pd.DataFrame,
    dense_residual: pd.DataFrame,
    model_bundle: object,
    sae: object,
    config: ExperimentConfig,
) -> dict[str, object]:
    del sae
    if layer not in config.steering.target_layers:
        return {
            "proxy": proxy,
            "layer": layer,
            "alpha": 0.0,
            "status": "skipped_layer",
            "margin_shift": np.nan,
            "margin_shift_ci_low": np.nan,
            "margin_shift_ci_high": np.nan,
            "feature_overlap_with_direction": np.nan,
        }

    direction = _build_proxy_direction(proxy, dense_residual, segments)
    if direction.size == 0:
        return {
            "proxy": proxy,
            "layer": layer,
            "alpha": 0.0,
            "status": "missing_direction",
            "margin_shift": np.nan,
            "margin_shift_ci_low": np.nan,
            "margin_shift_ci_high": np.nan,
            "feature_overlap_with_direction": np.nan,
        }

    dev_rows = _build_evaluation_rows(segments, matches, proxy, config, eval_split="dev")
    test_rows = _build_evaluation_rows(segments, matches, proxy, config, eval_split="test")
    dev_positive_count = int(dev_rows[proxy].astype(bool).sum()) if not dev_rows.empty else 0
    alpha, status = _select_alpha_for_steering(proxy, layer, direction, dev_rows, model_bundle, config, dev_positive_count)

    pair_proxy = PAIR_MAP[proxy]
    margin_shifts: list[float] = []
    for record in test_rows.itertuples(index=False):
        prompt = build_forced_choice_prompt(str(record.text), proxy, pair_proxy)
        baseline_margin = margin_for_prompt(model_bundle, prompt, proxy, pair_proxy)
        steered_margin = margin_for_prompt(
            model_bundle,
            prompt,
            proxy,
            pair_proxy,
            editor=lambda hidden: steering_editor(hidden, direction, alpha),
            layer=layer,
        )
        margin_shifts.append(steered_margin - baseline_margin)

    margin_ci = bootstrap_ci(margin_shifts, config.ablation.bootstrap_iterations, config.ablation.ci_level)
    return {
        "proxy": proxy,
        "layer": layer,
        "alpha": alpha,
        "status": status,
        "dev_positive_count": dev_positive_count,
        "margin_shift": float(np.mean(margin_shifts)) if margin_shifts else np.nan,
        "margin_shift_ci_low": margin_ci[0],
        "margin_shift_ci_high": margin_ci[1],
        "feature_overlap_with_direction": steering_sae_overlap(direction, selected_features)
        if config.steering.report_overlap_with_sae
        else np.nan,
    }


def _build_proxy_direction(proxy: str, dense_residual: pd.DataFrame, segments: pd.DataFrame) -> np.ndarray:
    if dense_residual.empty:
        return np.zeros((0,), dtype=np.float32)
    merged = dense_residual.merge(segments[["segment_id", proxy]], on="segment_id", how="left")
    train = merged.loc[merged["split"] == "train"].copy()
    if train.empty:
        return np.zeros((0,), dtype=np.float32)
    positives = train.loc[train[proxy].fillna(False)]
    negatives = train.loc[~train[proxy].fillna(False)]
    if positives.empty or negatives.empty:
        return np.zeros((0,), dtype=np.float32)
    pos_mean = np.nan_to_num(
        np.vstack(positives["vector"].tolist()).astype(np.float32),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    ).mean(axis=0)
    neg_mean = np.nan_to_num(
        np.vstack(negatives["vector"].tolist()).astype(np.float32),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    ).mean(axis=0)
    return (pos_mean - neg_mean).astype(np.float32)


def _select_alpha_for_steering(
    proxy: str,
    layer: int,
    direction: np.ndarray,
    dev_rows: pd.DataFrame,
    model_bundle: object,
    config: ExperimentConfig,
    dev_positive_count: int,
) -> tuple[float, str]:
    if dev_rows.empty:
        return 0.0, "missing_dev_rows"
    if dev_positive_count < config.steering.min_dev_positives_for_sweep:
        return float(config.steering.default_alpha_if_small_dev), "suggestive"

    pair_proxy = PAIR_MAP[proxy]
    best_alpha = float(config.steering.default_alpha_if_small_dev)
    best_score = -np.inf
    for alpha in config.steering.alpha_grid:
        margin_shifts: list[float] = []
        for record in dev_rows.itertuples(index=False):
            prompt = build_forced_choice_prompt(str(record.text), proxy, pair_proxy)
            baseline_margin = margin_for_prompt(model_bundle, prompt, proxy, pair_proxy)
            steered_margin = margin_for_prompt(
                model_bundle,
                prompt,
                proxy,
                pair_proxy,
                editor=lambda hidden, step=float(alpha): steering_editor(hidden, direction, step),
                layer=layer,
            )
            margin_shifts.append(steered_margin - baseline_margin)
        mean_shift = float(np.mean(margin_shifts)) if margin_shifts else -np.inf
        if mean_shift > best_score:
            best_score = mean_shift
            best_alpha = float(alpha)
    return best_alpha, "supported"


def steering_sae_overlap(direction: np.ndarray, selected_features: list[int]) -> float:
    if direction.size == 0 or not selected_features:
        return np.nan
    top_indices = np.argsort(np.abs(direction))[::-1][: len(selected_features)]
    direction_set = {int(index) for index in top_indices.tolist()}
    feature_set = {int(feature_id) for feature_id in selected_features}
    union = direction_set | feature_set
    if not union:
        return np.nan
    return float(len(direction_set & feature_set) / len(union))


def bootstrap_ci(values: list[float], iterations: int, ci_level: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (np.nan, np.nan)
    if array.size == 1:
        scalar = float(array[0])
        return (scalar, scalar)
    rng = np.random.default_rng(17)
    bootstrap_means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.choice(array, size=array.size, replace=True)
        bootstrap_means[index] = sample.mean()
    lower_q = (1.0 - ci_level) / 2.0
    upper_q = 1.0 - lower_q
    return (
        float(np.quantile(bootstrap_means, lower_q)),
        float(np.quantile(bootstrap_means, upper_q)),
    )
