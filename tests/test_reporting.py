import pandas as pd

from policy_interp.reporting import _build_policy_feature_map_frame


def test_build_policy_feature_map_frame_merges_causal_and_overlay() -> None:
    top_features = pd.DataFrame(
        [
            {
                "layer": 12,
                "feature_id": 101,
                "feature_name": "Control Measures Safety",
                "primary_ranking_family": "layer_unique",
                "rank": 1,
                "best_proxy": "privacy",
                "faithfulness_score": 0.8,
                "score": 12.0,
            },
            {
                "layer": 20,
                "feature_id": 202,
                "feature_name": "High Risk Decision Systems",
                "primary_ranking_family": "policy_specific",
                "rank": 2,
                "best_proxy": "discrimination",
                "faithfulness_score": 0.7,
                "score": 20.0,
            },
        ]
    )
    causal_frame = pd.DataFrame(
        [
            {
                "target_type": "autointerp_single_feature",
                "layer": 20,
                "feature_ids": [202],
                "target_id": "autointerp_layer_20_feature_202",
                "paired_delta": 0.01,
                "paired_delta_ci_low": 0.001,
                "paired_delta_ci_high": 0.02,
                "kl_divergence_delta": 0.002,
                "kl_divergence_delta_ci_low": 0.0005,
                "kl_divergence_delta_ci_high": 0.003,
                "perplexity_shift_delta": 0.05,
                "top1_change_rate_delta": 0.01,
                "causal_badge": "proxy_free_causal",
            }
        ]
    )
    overlay_frame = pd.DataFrame(
        [
            {
                "layer": 12,
                "feature_id": 101,
                "proxy": "privacy",
                "test_auc": 0.61,
                "validated_auc": 0.58,
                "mutual_information": 0.03,
            }
        ]
    )

    frame = _build_policy_feature_map_frame(top_features, causal_frame, overlay_frame)

    assert len(frame) == 2
    row_12 = frame.loc[(frame["layer"] == 12) & (frame["feature_id"] == 101)].iloc[0]
    row_20 = frame.loc[(frame["layer"] == 20) & (frame["feature_id"] == 202)].iloc[0]

    assert row_12["overlay_proxy"] == "privacy"
    assert abs(float(row_12["overlay_test_auc"]) - 0.61) < 1e-8
    assert bool(row_12["causal_available"]) is False

    assert row_20["target_id"] == "autointerp_layer_20_feature_202"
    assert abs(float(row_20["kl_divergence_delta"]) - 0.002) < 1e-8
    assert row_20["causal_badge"] == "proxy_free_causal"
    assert bool(row_20["causal_available"]) is True
