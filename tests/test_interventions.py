from pathlib import Path

import pandas as pd
import torch

from policy_interp.interventions import (
    _build_ablation_targets,
    _mean_metric_dicts,
    _mean_token_kl_divergence,
    bootstrap_ci,
)
from policy_interp.schemas import DatasetConfig, ExperimentConfig


def test_build_ablation_targets_prefers_stable_modules_and_probe_topk(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    stable_modules = pd.DataFrame(
        [
            {
                "stable_module_id": "layer_24_stable_001",
                "layer": 24,
                "stable": True,
                "feature_ids": [1, 2, 3],
            },
            {
                "stable_module_id": "layer_24_stable_002",
                "layer": 24,
                "stable": True,
                "feature_ids": [4, 5, 6],
            },
            {
                "stable_module_id": "layer_24_unstable_003",
                "layer": 24,
                "stable": False,
                "feature_ids": [7, 8, 9],
            },
        ]
    )
    alignment = pd.DataFrame(
        [
            {"stable_module_id": "layer_24_unstable_003", "proxy": "privacy", "dev_auc": 0.99, "test_auc": 0.99},
            {"stable_module_id": "layer_24_stable_001", "proxy": "privacy", "dev_auc": 0.72, "test_auc": 0.68},
            {"stable_module_id": "layer_24_stable_002", "proxy": "bias", "dev_auc": 0.63, "test_auc": 0.60},
        ]
    )
    sparse_feature_selection = pd.DataFrame(
        [
            {
                "proxy": "privacy",
                "selected_layer": 24,
                "selected_feature_ids": [11, 12, 13, 14],
            },
            {
                "proxy": "bias",
                "selected_layer": 24,
                "selected_feature_ids": [21, 22, 23, 24],
            },
        ]
    )

    targets = _build_ablation_targets(config, stable_modules, alignment, sparse_feature_selection)
    target_index = {target.target_id: target for target in targets}

    assert set(target_index) == {
        "privacy_module_whole",
        "privacy_individual_top3",
        "bias_individual_top3",
        "bias_module_whole",
    }
    assert target_index["privacy_module_whole"].feature_ids == [1, 2, 3]
    assert target_index["privacy_individual_top3"].feature_ids == [11, 12, 13]
    assert target_index["bias_individual_top3"].feature_ids == [21, 22, 23]
    assert target_index["bias_module_whole"].feature_ids == [4, 5, 6]


def test_build_ablation_targets_adds_autointerp_single_feature_targets(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    autointerp_dir = config.run_root / "features" / "autointerp"
    autointerp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_id": "google/gemma-2-2b",
                "sae_release": "gemma-scope-2b-pt-res-canonical",
                "layer": 12,
                "feature_id": 101,
                "primary_ranking_family": "policy_specific",
                "rank": 1,
                "best_proxy": "privacy",
                "feature_name": "Feature A",
                "simulation_accuracy": 0.8,
                "faithfulness_score": 0.8,
            },
            {
                "model_id": "google/gemma-2-2b",
                "sae_release": "gemma-scope-2b-pt-res-canonical",
                "layer": 16,
                "feature_id": 202,
                "primary_ranking_family": "layer_unique",
                "rank": 1,
                "best_proxy": "interpretability",
                "feature_name": "Feature B",
                "simulation_accuracy": 0.7,
                "faithfulness_score": 0.7,
            },
        ]
    ).to_parquet(autointerp_dir / "autointerp_feature_scores.parquet", index=False)

    stable_modules = pd.DataFrame(columns=["stable_module_id", "layer", "stable", "feature_ids"])
    alignment = pd.DataFrame(columns=["stable_module_id", "proxy", "dev_auc", "test_auc"])
    sparse_feature_selection = pd.DataFrame(columns=["proxy", "selected_layer", "selected_feature_ids"])

    targets = _build_ablation_targets(config, stable_modules, alignment, sparse_feature_selection)
    target_index = {target.target_id: target for target in targets}

    assert "autointerp_layer_12_feature_101" in target_index
    assert "autointerp_layer_16_feature_202" in target_index
    assert target_index["autointerp_layer_12_feature_101"].feature_ids == [101]
    assert target_index["autointerp_layer_12_feature_101"].ranking_family == "policy_specific"
    assert target_index["autointerp_layer_12_feature_101"].primary_proxy == "privacy"
    assert target_index["autointerp_layer_16_feature_202"].primary_proxy == "interpretability"


def test_build_ablation_targets_adds_autointerp_feature_set_targets(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    autointerp_dir = config.run_root / "features" / "autointerp"
    autointerp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_id": "google/gemma-2-2b",
                "sae_release": "gemma-scope-2b-pt-res-canonical",
                "layer": 20,
                "feature_id": 301,
                "primary_ranking_family": "policy_specific",
                "rank": 1,
                "best_proxy": "privacy",
                "feature_name": "Feature A",
                "simulation_accuracy": 0.8,
                "faithfulness_score": 0.9,
            },
            {
                "model_id": "google/gemma-2-2b",
                "sae_release": "gemma-scope-2b-pt-res-canonical",
                "layer": 20,
                "feature_id": 302,
                "primary_ranking_family": "policy_specific",
                "rank": 2,
                "best_proxy": "privacy",
                "feature_name": "Feature B",
                "simulation_accuracy": 0.75,
                "faithfulness_score": 0.85,
            },
            {
                "model_id": "google/gemma-2-2b",
                "sae_release": "gemma-scope-2b-pt-res-canonical",
                "layer": 20,
                "feature_id": 303,
                "primary_ranking_family": "layer_unique",
                "rank": 3,
                "best_proxy": "privacy",
                "feature_name": "Feature C",
                "simulation_accuracy": 0.7,
                "faithfulness_score": 0.82,
            },
            {
                "model_id": "google/gemma-2-2b",
                "sae_release": "gemma-scope-2b-pt-res-canonical",
                "layer": 20,
                "feature_id": 304,
                "primary_ranking_family": "layer_unique",
                "rank": 4,
                "best_proxy": "privacy",
                "feature_name": "Feature D",
                "simulation_accuracy": 0.68,
                "faithfulness_score": 0.81,
            },
            {
                "model_id": "google/gemma-2-2b",
                "sae_release": "gemma-scope-2b-pt-res-canonical",
                "layer": 20,
                "feature_id": 305,
                "primary_ranking_family": "layer_unique",
                "rank": 5,
                "best_proxy": "privacy",
                "feature_name": "Feature E",
                "simulation_accuracy": 0.66,
                "faithfulness_score": 0.8,
            },
        ]
    ).to_parquet(autointerp_dir / "autointerp_feature_scores.parquet", index=False)

    stable_modules = pd.DataFrame(columns=["stable_module_id", "layer", "stable", "feature_ids"])
    alignment = pd.DataFrame(columns=["stable_module_id", "proxy", "dev_auc", "test_auc"])
    sparse_feature_selection = pd.DataFrame(columns=["proxy", "selected_layer", "selected_feature_ids"])

    targets = _build_ablation_targets(config, stable_modules, alignment, sparse_feature_selection)
    target_index = {target.target_id: target for target in targets}

    assert "autointerp_layer_20_top_3" in target_index
    assert "autointerp_layer_20_top_5" in target_index
    assert target_index["autointerp_layer_20_top_3"].feature_ids == [301, 302, 303]
    assert target_index["autointerp_layer_20_top_5"].feature_ids == [301, 302, 303, 304, 305]
    assert target_index["autointerp_layer_20_top_3"].primary_proxy == "privacy"
    assert target_index["autointerp_layer_20_top_3"].target_type == "autointerp_feature_set"


def test_mean_token_kl_divergence_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[[2.0, 0.5], [1.0, 1.0]]], dtype=torch.float32)
    kl_value = _mean_token_kl_divergence(logits, logits.clone())
    assert abs(kl_value) < 1e-8


def test_mean_metric_dicts_includes_perplexity_shift() -> None:
    metrics = _mean_metric_dicts(
        [
            {
                "kl_divergence": 0.1,
                "perplexity_shift": 2.0,
                "mean_nll_shift": 0.2,
                "top1_change_rate": 0.3,
            },
            {
                "kl_divergence": 0.3,
                "perplexity_shift": 4.0,
                "mean_nll_shift": 0.4,
                "top1_change_rate": 0.5,
            },
        ]
    )
    assert abs(metrics["kl_divergence"] - 0.2) < 1e-8
    assert abs(metrics["perplexity_shift"] - 3.0) < 1e-8
    assert abs(metrics["mean_nll_shift"] - 0.3) < 1e-8
    assert abs(metrics["top1_change_rate"] - 0.4) < 1e-8


def test_bootstrap_ci_singleton_returns_same_value() -> None:
    low, high = bootstrap_ci([0.25], iterations=50, ci_level=0.95)
    assert low == 0.25
    assert high == 0.25
