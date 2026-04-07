import pandas as pd

from policy_interp.batch_scorer import _segment_document_text
from policy_interp.feature_catalog import (
    _build_catalog_segment_scores,
    _merge_backfilled_contexts,
    _select_evidence_backfill_candidates,
)
from policy_interp.interventions import _build_feature_first_summary_frame
from policy_interp.schemas import DatasetConfig, ExperimentConfig


def test_segment_document_text_supports_paragraph_mode() -> None:
    text = "First paragraph.\nStill first.\n\nSecond paragraph.\n\nThird paragraph."
    frame = _segment_document_text(text, output_name="demo", segment_mode="paragraph")

    assert frame["segment_id"].tolist() == [
        "demo_segment_0000",
        "demo_segment_0001",
        "demo_segment_0002",
    ]
    assert frame["text"].tolist() == [
        "First paragraph. Still first.",
        "Second paragraph.",
        "Third paragraph.",
    ]
    assert frame["split"].tolist() == ["inference", "inference", "inference"]


def test_build_feature_first_summary_frame_adds_feature_first_aliases() -> None:
    summary = _build_feature_first_summary_frame(
        [
            {
                "target_id": "privacy_individual_top3",
                "target_name": "Privacy individual probe top 3",
                "target_type": "individual_probe_topk",
                "primary_proxy": "privacy",
                "layer": 24,
                "feature_ids": [11, 12, 13],
                "target_margin_drop": 0.2,
                "random_control_margin_drop": 0.1,
                "paired_delta": 0.1,
                "paired_delta_ci_low": 0.02,
                "paired_delta_ci_high": 0.18,
                "dense_margin_drop": 0.03,
                "dense_paired_delta": -0.01,
                "dense_paired_delta_ci_low": -0.05,
                "dense_paired_delta_ci_high": 0.02,
                "target_mean_kl_divergence": 0.4,
                "random_mean_kl_divergence": 0.1,
                "kl_divergence_delta": 0.3,
                "kl_divergence_delta_ci_low": 0.1,
                "kl_divergence_delta_ci_high": 0.5,
                "dense_mean_kl_divergence": 0.08,
                "dense_kl_divergence_delta": -0.02,
                "dense_kl_divergence_delta_ci_low": -0.08,
                "dense_kl_divergence_delta_ci_high": 0.04,
                "target_mean_perplexity_shift": 1.5,
                "random_mean_perplexity_shift": 0.2,
                "perplexity_shift_delta": 1.3,
                "perplexity_shift_delta_ci_low": 0.8,
                "perplexity_shift_delta_ci_high": 1.8,
                "dense_mean_perplexity_shift": 0.1,
                "dense_perplexity_shift_delta": -0.1,
                "dense_perplexity_shift_delta_ci_low": -0.3,
                "dense_perplexity_shift_delta_ci_high": 0.1,
                "target_mean_nll_shift": 0.6,
                "random_mean_nll_shift": 0.1,
                "nll_shift_delta": 0.5,
                "nll_shift_delta_ci_low": 0.2,
                "nll_shift_delta_ci_high": 0.8,
                "dense_mean_nll_shift": 0.05,
                "dense_nll_shift_delta": -0.05,
                "dense_nll_shift_delta_ci_low": -0.2,
                "dense_nll_shift_delta_ci_high": 0.05,
                "target_mean_top1_change_rate": 0.12,
                "random_mean_top1_change_rate": 0.03,
                "top1_change_rate_delta": 0.09,
                "top1_change_rate_delta_ci_low": 0.04,
                "top1_change_rate_delta_ci_high": 0.14,
                "dense_mean_top1_change_rate": 0.02,
                "dense_top1_change_rate_delta": -0.01,
                "dense_top1_change_rate_delta_ci_low": -0.03,
                "dense_top1_change_rate_delta_ci_high": 0.01,
            }
        ]
    )

    row = summary.iloc[0]
    assert row["target_kind"] == "feature_topk"
    assert row["ranking_family"] == "individual_probe"
    assert row["proxy_overlay"] == "privacy"
    assert row["causal_badge"] == "proxy_free_causal"
    assert "paired_delta" in row["ci"]
    assert "dense_kl_divergence_delta" in row["dense_controls"]


def test_select_evidence_backfill_candidates_targets_missing_early_feature(tmp_path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    config.feature_catalog.evidence_backfill_top_n_per_family = 2
    config.autointerp.enabled = True
    config.autointerp.num_train_positive = 2
    config.autointerp.num_holdout_positive = 3
    catalog = pd.DataFrame(
        [
            {"layer": 12, "feature_id": 1, "ranking_family": "policy_specific", "rank": 1},
            {"layer": 12, "feature_id": 2, "ranking_family": "policy_specific", "rank": 2},
            {"layer": 24, "feature_id": 3, "ranking_family": "policy_specific", "rank": 1},
        ]
    )
    top_feature_frame = pd.DataFrame(
        [{"layer": 24, "feature_id": 3, "segment_id": f"s{i}"} for i in range(6)]
    )
    context_frame = pd.DataFrame(
        [{"layer": 24, "feature_id": 3, "rank": i + 1} for i in range(5)]
    )

    selected = _select_evidence_backfill_candidates(catalog, top_feature_frame, context_frame, config)

    assert selected["feature_id"].tolist() == [1, 2]
    assert (selected["layer"] == 12).all()


def test_build_catalog_segment_scores_prefers_backfilled_values(tmp_path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    catalog = pd.DataFrame(
        [
            {
                "layer": 12,
                "feature_id": 7,
                "model_id": "demo",
                "sae_release": "demo_sae",
                "model_depth": 26,
                "layer_depth_fraction": 0.48,
                "layer_stage": "mid",
            }
        ]
    )
    top_feature_frame = pd.DataFrame(
        [
            {
                "layer": 12,
                "segment_id": "s1",
                "feature_id": 7,
                "pooled_activation": 0.0,
                "token_values": [0.0],
                "token_positions": [0],
                "top_token_span_text": "",
            }
        ]
    )
    segments = pd.DataFrame(
        [
            {"segment_id": "s1", "document_id": 1, "split": "train"},
            {"segment_id": "s2", "document_id": 2, "split": "test"},
        ]
    )
    backfilled_segment_scores = pd.DataFrame(
        [
            {
                "layer": 12,
                "segment_id": "s1",
                "feature_id": 7,
                "pooled_activation": 3.5,
                "peak_token_value": 1.2,
                "peak_token_position": 4,
                "top_token_span_text": "important clause",
            },
            {
                "layer": 12,
                "segment_id": "s2",
                "feature_id": 7,
                "pooled_activation": 0.8,
                "peak_token_value": 0.8,
                "peak_token_position": 2,
                "top_token_span_text": "notice text",
            },
        ]
    )

    segment_scores = _build_catalog_segment_scores(
        catalog_frame=catalog,
        top_feature_frame=top_feature_frame,
        segments=segments,
        config=config,
        backfilled_segment_scores=backfilled_segment_scores,
    )

    assert segment_scores.loc[segment_scores["segment_id"] == "s1", "pooled_activation"].iloc[0] == 3.5
    assert segment_scores.loc[segment_scores["segment_id"] == "s2", "top_token_span_text"].iloc[0] == "notice text"


def test_merge_backfilled_contexts_replaces_original_feature_rows() -> None:
    base = pd.DataFrame(
        [
            {"layer": 12, "feature_id": 1, "rank": 1, "context_text": "base"},
            {"layer": 24, "feature_id": 2, "rank": 1, "context_text": "keep"},
        ]
    )
    backfilled = pd.DataFrame(
        [
            {"layer": 12, "feature_id": 1, "rank": 1, "context_text": "backfilled"},
            {"layer": 12, "feature_id": 1, "rank": 2, "context_text": "backfilled second"},
        ]
    )

    merged = _merge_backfilled_contexts(base, backfilled)

    assert merged.loc[(merged["layer"] == 12) & (merged["feature_id"] == 1), "context_text"].tolist() == [
        "backfilled",
        "backfilled second",
    ]
    assert merged.loc[(merged["layer"] == 24) & (merged["feature_id"] == 2), "context_text"].tolist() == ["keep"]
