import pandas as pd

from policy_interp.baselines import _build_feature_module_overlap_frame, _select_top_individual_feature_ids
from policy_interp.discovery import _benjamini_hochberg, _compute_pairwise_jaccard
from policy_interp.feature_matrix import build_module_score_matrix
from policy_interp.labeling import _normalize_generation_lines
from policy_interp.masking import _clip_ratio, mask_text


def test_bh_fdr_monotonicity() -> None:
    qvals = _benjamini_hochberg([0.01, 0.02, 0.20, 0.25])
    assert all(0.0 <= value <= 1.0 for value in qvals)
    assert qvals[0] <= qvals[2]


def test_pairwise_jaccard_from_top_features() -> None:
    top_features = pd.DataFrame(
        [
            {"segment_id": "a", "feature_id": 1, "pooled_activation": 0.8},
            {"segment_id": "a", "feature_id": 2, "pooled_activation": 0.7},
            {"segment_id": "b", "feature_id": 1, "pooled_activation": 0.9},
            {"segment_id": "b", "feature_id": 3, "pooled_activation": 0.5},
        ]
    )
    weights, activity = _compute_pairwise_jaccard(top_features)
    assert activity[1] == 2
    assert abs(weights[(1, 2)] - 0.5) < 1e-6


def test_mask_text_replaces_anchor_boundaries() -> None:
    masked = mask_text("privacy rules mention privacy preserving methods", ["privacy"], "[MASK]")
    assert masked.count("[MASK]") == 2


def test_clip_ratio_limits_above_one() -> None:
    assert _clip_ratio(1.2) == 1.0
    assert _clip_ratio(-0.2) == 0.0


def test_select_top_individual_feature_ids_prefers_predictive_features() -> None:
    feature_ids = [10, 20, 30]
    matrix = [
        [1.0, 0.0, 0.1],
        [1.2, 0.1, 0.2],
        [0.0, 2.0, 0.1],
        [0.1, 2.2, 0.2],
    ]
    labels = [1, 1, 0, 0]
    splits = ["train", "train", "train", "train"]
    selected = _select_top_individual_feature_ids(
        feature_ids=feature_ids,
        matrix=matrix,
        labels=labels,
        splits=splits,
        top_k=2,
    )
    assert 10 in selected
    assert 20 in selected


def test_build_feature_module_overlap_frame_reports_jaccard() -> None:
    selection = pd.DataFrame(
        [
            {
                "proxy": "privacy",
                "selected_layer": 24,
                "selected_feature_ids": [1, 2, 3],
                "selected_feature_count": 3,
            }
        ]
    )
    stable_modules = pd.DataFrame(
        [
            {
                "stable": True,
                "stable_module_id": "layer_24_stable_001",
                "layer": 24,
                "module_size": 2,
                "feature_ids": [2, 3],
            }
        ]
    )
    overlap = _build_feature_module_overlap_frame(selection, stable_modules, enabled=True)
    assert len(overlap) == 1
    assert overlap.loc[0, "overlap_count"] == 2
    assert abs(overlap.loc[0, "jaccard"] - (2 / 3)) < 1e-6


def test_normalize_generation_lines_strips_markdown_noise() -> None:
    lines = _normalize_generation_lines("```python\nName: Privacy safeguards\nRationale: Focuses on privacy language.\n```")
    assert lines == ["Privacy safeguards", "Focuses on privacy language."]


def test_build_module_score_matrix() -> None:
    stable_modules = pd.DataFrame(
        [
            {
                "stable_module_id": "layer_24_stable_001",
                "layer": 24,
                "stable": True,
                "feature_ids": [1, 2],
            }
        ]
    )
    top_features = pd.DataFrame(
        [
            {"segment_id": "s1", "feature_id": 1, "pooled_activation": 1.0},
            {"segment_id": "s1", "feature_id": 2, "pooled_activation": 0.5},
            {"segment_id": "s2", "feature_id": 2, "pooled_activation": 1.5},
        ]
    )
    segments = pd.DataFrame(
        [
            {"segment_id": "s1", "split": "train"},
            {"segment_id": "s2", "split": "test"},
        ]
    )
    scores = build_module_score_matrix(stable_modules, top_features, segments, layer=24)
    value = scores.loc[scores["segment_id"] == "s1", "layer_24_stable_001"].item()
    assert abs(value - 0.75) < 1e-6
