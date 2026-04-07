from pathlib import Path

import pandas as pd

from policy_interp.audit_eval import (
    _compute_autointerp_validation,
    _compute_failure_transparency,
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
