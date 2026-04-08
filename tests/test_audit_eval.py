from pathlib import Path

import numpy as np
import pandas as pd

from policy_interp.audit_eval import (
    _compute_autointerp_validation,
    _compute_discriminativeness_statistics,
    _compute_failure_transparency,
    _compute_report_robustness_statistics,
    _summarize_autointerp_validation,
)
from policy_interp.schemas import DatasetConfig, ExperimentConfig


def test_compute_autointerp_validation_adds_contrastive_and_lexicality(tmp_path: Path) -> None:
    autointerp_scores = pd.DataFrame(
        [
            {
                "layer": 20,
                "feature_id": 101,
                "primary_ranking_family": "policy_specific",
                "faithfulness_score": 0.75,
                "feature_name": "High Risk Decision Systems",
                "activation_hypothesis": "Activates on high risk decision system obligations",
                "boundary_text": "high risk systems",
            }
        ]
    )
    autointerp_simulation = pd.DataFrame(
        [
            {
                "layer": 20,
                "feature_id": 101,
                "primary_ranking_family": "policy_specific",
                "gold_label": 1,
                "predicted_label": 1,
                "confidence": 0.9,
            },
            {
                "layer": 20,
                "feature_id": 101,
                "primary_ranking_family": "policy_specific",
                "gold_label": 0,
                "predicted_label": 0,
                "confidence": 0.2,
            },
        ]
    )
    exemplars = pd.DataFrame(
        [
            {
                "layer": 20,
                "feature_id": 101,
                "ranking_family": "policy_specific",
                "example_kind": "positive",
                "top_token_span_text": "high risk decision system obligations",
            }
        ]
    )

    frame = _compute_autointerp_validation(autointerp_scores, autointerp_simulation, exemplars)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert float(row["contrastive_accuracy"]) == 1.0
    assert float(row["lexicality_penalty"]) > 0.0


def test_summarize_autointerp_validation_reports_threshold_counts(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    frame = pd.DataFrame(
        [
            {
                "layer": 12,
                "feature_id": 1,
                "faithfulness_score": 0.8,
                "contrastive_accuracy": 0.9,
                "lexicality_penalty": 0.2,
            },
            {
                "layer": 12,
                "feature_id": 2,
                "faithfulness_score": 0.4,
                "contrastive_accuracy": 0.6,
                "lexicality_penalty": 0.1,
            },
        ]
    )

    summary = _summarize_autointerp_validation(frame, config)

    assert len(summary) == 1
    assert int(summary.iloc[0]["feature_count"]) == 2
    assert int(summary.iloc[0]["count_high_faithfulness"]) == 1


def test_compute_failure_transparency_summarizes_negative_findings(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    feature_summary = pd.DataFrame(
        [
            {"layer": 24, "feature_id": 1, "activation_frequency": 1.0},
            {"layer": 24, "feature_id": 2, "activation_frequency": 1.0},
        ]
    )
    baseline_comparison = pd.DataFrame(
        [
            {"proxy": "privacy", "method": "sparse_individual_feature_probe", "test_auc": 0.65},
            {"proxy": "privacy", "method": "sparse_module_probe", "test_auc": 0.45},
        ]
    )
    causal_summary = pd.DataFrame(
        [
            {
                "target_type": "autointerp_single_feature",
                "kl_divergence_delta": 0.0001,
                "paired_delta": 0.001,
            }
        ]
    )
    overlay = pd.DataFrame(
        [
            {"layer": 24, "feature_id": 10, "proxy": "privacy", "test_auc": 0.52},
            {"layer": 24, "feature_id": 10, "proxy": "bias", "test_auc": 0.51},
        ]
    )
    catalog = pd.DataFrame(
        [
            {"layer": 24, "feature_id": 1, "ranking_family": "global_dominance", "rank": 1},
            {"layer": 24, "feature_id": 2, "ranking_family": "global_dominance", "rank": 2},
            {"layer": 24, "feature_id": 10, "ranking_family": "policy_specific", "rank": 1},
        ]
    )

    summary = _compute_failure_transparency(
        config=config,
        feature_summary=feature_summary,
        baseline_comparison=baseline_comparison,
        causal_summary=causal_summary,
        overlay=overlay,
        catalog=catalog,
    )

    assert set(summary["metric_name"]) == {
        "top_overall_activation_frequency_one_rate",
        "individual_minus_module_auc_gap",
        "single_feature_near_zero_effect_rate",
        "flat_proxy_overlay_rate",
    }


def test_compute_discriminativeness_statistics_reports_ci_and_pvalues(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    config.ablation.bootstrap_iterations = 40
    config.validity.enrichment_random_trials = 40
    case_manifest = pd.DataFrame(
        [
            {"case_id": "a1", "family_id": "family_a"},
            {"case_id": "a2", "family_id": "family_a"},
            {"case_id": "b1", "family_id": "family_b"},
            {"case_id": "b2", "family_id": "family_b"},
        ]
    )
    case_scores = pd.DataFrame(
        [
            {"case_id": "a1", "ranking_family": "policy_specific", "pooled_activation": 3.0, "layer": 20, "feature_id": 101, "feature_key": "L20_F101"},
            {"case_id": "a1", "ranking_family": "global_dominance", "pooled_activation": 2.0, "layer": 12, "feature_id": 1, "feature_key": "L12_F1"},
            {"case_id": "a2", "ranking_family": "policy_specific", "pooled_activation": 2.5, "layer": 20, "feature_id": 101, "feature_key": "L20_F101"},
            {"case_id": "a2", "ranking_family": "global_dominance", "pooled_activation": 2.0, "layer": 12, "feature_id": 1, "feature_key": "L12_F1"},
            {"case_id": "b1", "ranking_family": "policy_specific", "pooled_activation": 3.0, "layer": 20, "feature_id": 202, "feature_key": "L20_F202"},
            {"case_id": "b1", "ranking_family": "global_dominance", "pooled_activation": 2.0, "layer": 12, "feature_id": 1, "feature_key": "L12_F1"},
            {"case_id": "b2", "ranking_family": "policy_specific", "pooled_activation": 2.5, "layer": 20, "feature_id": 202, "feature_key": "L20_F202"},
            {"case_id": "b2", "ranking_family": "global_dominance", "pooled_activation": 2.0, "layer": 12, "feature_id": 1, "feature_key": "L12_F1"},
        ]
    )

    stats = _compute_discriminativeness_statistics(config, case_manifest, case_scores)

    assert {"view_name", "cosine_gap_ci_low", "cosine_gap_ci_high", "retrieval_accuracy_permutation_pvalue"}.issubset(
        set(stats.columns)
    )
    policy_row = stats.loc[stats["view_name"] == "policy_specific_view"].iloc[0]
    assert float(policy_row["cosine_gap"]) > 0.0
    assert float(policy_row["retrieval_accuracy"]) >= 0.5
    assert 0.0 <= float(policy_row["retrieval_accuracy_permutation_pvalue"]) <= 1.0


def test_compute_report_robustness_statistics_handles_array_rankings(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    config.ablation.bootstrap_iterations = 40
    robustness_case_scores = pd.DataFrame(
        [
            {
                "case_id": "case_1",
                "source_case_id": "case_1",
                "perturbation": "original",
                "policy_specific_feature_set": ["L20_F1", "L20_F2"],
                "overall_feature_set": ["L12_F1"],
                "proxy_ranking": np.array(["privacy", "bias"]),
                "layer_vector": {"L20_mean": 1.0, "L20_max": 2.0},
            },
            {
                "case_id": "case_2",
                "source_case_id": "case_2",
                "perturbation": "original",
                "policy_specific_feature_set": ["L24_F3"],
                "overall_feature_set": ["L12_F1"],
                "proxy_ranking": np.array(["bias", "privacy"]),
                "layer_vector": {"L20_mean": 2.0, "L20_max": 3.0},
            },
            {
                "case_id": "case_1__heading_removal",
                "source_case_id": "case_1",
                "perturbation": "heading_removal",
                "policy_specific_feature_set": ["L20_F1"],
                "overall_feature_set": ["L12_F1"],
                "proxy_ranking": np.array(["privacy", "bias"]),
                "layer_vector": {"L20_mean": 1.0, "L20_max": 2.0},
            },
            {
                "case_id": "case_2__heading_removal",
                "source_case_id": "case_2",
                "perturbation": "heading_removal",
                "policy_specific_feature_set": ["L24_F3"],
                "overall_feature_set": ["L12_F1"],
                "proxy_ranking": np.array(["privacy", "bias"]),
                "layer_vector": {"L20_mean": 2.1, "L20_max": 3.0},
            },
        ]
    )

    stats = _compute_report_robustness_statistics(config, robustness_case_scores)

    assert len(stats) == 1
    row = stats.iloc[0]
    assert row["perturbation"] == "heading_removal"
    assert float(row["policy_specific_jaccard_ci_low"]) <= float(row["mean_policy_specific_jaccard"])
    assert float(row["proxy_rank_spearman_ci_high"]) >= float(row["mean_proxy_rank_spearman"])
