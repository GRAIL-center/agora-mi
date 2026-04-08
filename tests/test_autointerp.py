import pandas as pd
import torch
from transformers.tokenization_utils_base import BatchEncoding

from policy_interp.adapters.modeling import resolve_torch_dtype
from policy_interp.autointerp import (
    _build_autointerp_candidates,
    _coerce_model_inputs,
    _derive_name_from_hypothesis,
    _looks_reasonable_name,
    _parse_interpretation_response,
    _parse_simulation_response,
    _score_simulation_rows,
    _strip_xml_fragments,
)
from policy_interp.schemas import DatasetConfig, ExperimentConfig


def test_build_autointerp_candidates_deduplicates_across_families(tmp_path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    config.autointerp.top_n_per_family = 2
    catalog = pd.DataFrame(
        [
            {
                "model_id": "demo",
                "sae_release": "demo_sae",
                "layer": 12,
                "feature_id": 1,
                "model_depth": 26,
                "layer_depth_fraction": 0.48,
                "layer_stage": "mid",
                "ranking_family": "policy_specific",
                "rank": 1,
                "best_proxy": "privacy",
                "best_proxy_test_auc": 0.7,
                "best_proxy_validated_auc": 0.6,
                "catalog_key": "a",
            },
            {
                "model_id": "demo",
                "sae_release": "demo_sae",
                "layer": 12,
                "feature_id": 1,
                "model_depth": 26,
                "layer_depth_fraction": 0.48,
                "layer_stage": "mid",
                "ranking_family": "layer_unique",
                "rank": 2,
                "best_proxy": "privacy",
                "best_proxy_test_auc": 0.7,
                "best_proxy_validated_auc": 0.6,
                "catalog_key": "b",
            },
            {
                "model_id": "demo",
                "sae_release": "demo_sae",
                "layer": 12,
                "feature_id": 2,
                "model_depth": 26,
                "layer_depth_fraction": 0.48,
                "layer_stage": "mid",
                "ranking_family": "layer_unique",
                "rank": 1,
                "best_proxy": "bias",
                "best_proxy_test_auc": 0.65,
                "best_proxy_validated_auc": 0.55,
                "catalog_key": "c",
            },
        ]
    )

    candidates = _build_autointerp_candidates(catalog, config)

    assert len(candidates) == 2
    first = candidates.loc[candidates["feature_id"] == 1].iloc[0]
    assert first["primary_ranking_family"] == "policy_specific"
    assert first["ranking_families"] == ["policy_specific", "layer_unique"]


def test_score_simulation_rows_computes_faithfulness(tmp_path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    candidate = _build_autointerp_candidates(
        pd.DataFrame(
            [
                {
                    "model_id": "demo",
                    "sae_release": "demo_sae",
                    "layer": 24,
                    "feature_id": 9,
                    "model_depth": 26,
                    "layer_depth_fraction": 0.96,
                    "layer_stage": "late",
                    "ranking_family": "policy_specific",
                    "rank": 1,
                    "best_proxy": "privacy",
                    "best_proxy_test_auc": 0.7,
                    "best_proxy_validated_auc": 0.6,
                    "catalog_key": "x",
                }
            ]
        ),
        config,
    ).iloc[0]

    score = _score_simulation_rows(
        candidate=candidate,
        interpretation={
            "feature_name": "Data protection obligation",
            "activation_hypothesis": "This feature activates on direct data protection obligations.",
            "boundary_text": "It should stay low on unrelated policy passages.",
        },
        simulation_rows=[
            {"gold_label": 1, "predicted_label": 1, "confidence": 90.0},
            {"gold_label": 1, "predicted_label": 0, "confidence": 60.0},
            {"gold_label": 0, "predicted_label": 0, "confidence": 70.0},
            {"gold_label": 0, "predicted_label": 0, "confidence": 80.0},
        ],
        best_proxy="privacy",
    )

    assert abs(score["simulation_accuracy"] - 0.75) < 1e-8
    assert abs(score["simulation_balanced_accuracy"] - 0.75) < 1e-8
    assert abs(score["positive_precision"] - 1.0) < 1e-8
    assert abs(score["positive_recall"] - 0.5) < 1e-8
    assert abs(score["faithfulness_score"] - 0.75) < 1e-8


def test_parse_interpretation_response_reads_xml_fields() -> None:
    parsed = _parse_interpretation_response(
        "<name>Data sharing obligation</name>\n"
        "<hypothesis>This feature activates on clauses that require organizations to share personal data.</hypothesis>\n"
        "<boundary>It should stay low on passages that mention data but do not impose a sharing requirement.</boundary>"
    )

    assert parsed is not None
    assert parsed["feature_name"] == "Data sharing obligation"
    assert "share personal data" in parsed["activation_hypothesis"]


def test_parse_simulation_response_reads_xml_fields() -> None:
    parsed = _parse_simulation_response(
        "<label>activates</label>\n<confidence>82</confidence>\n<reason>The candidate contains a direct compliance requirement.</reason>"
    )

    assert parsed is not None
    assert parsed["predicted_label"] == 1
    assert parsed["confidence"] == 82.0


def test_looks_reasonable_name_rejects_prompt_echo() -> None:
    assert not _looks_reasonable_name("Based on the provided context, what causes the feature to activate?")
    assert _looks_reasonable_name("Data retention clause")


def test_strip_xml_fragments_removes_tags() -> None:
    assert _strip_xml_fragments("<hypothesis>National security AI systems</hypothesis>") == "National security AI systems"


def test_derive_name_from_hypothesis_builds_compact_label() -> None:
    derived = _derive_name_from_hypothesis(
        "The feature activates when the text discusses the development or use of AI systems for national security purposes.",
        "Layer 24 feature 11802",
    )
    assert derived == "Development Systems National Security"


def test_coerce_model_inputs_supports_batch_encoding_style_dict() -> None:
    encoded = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    model_inputs = _coerce_model_inputs(encoded, "cpu")

    assert torch.equal(model_inputs["input_ids"], encoded["input_ids"])
    assert torch.equal(model_inputs["attention_mask"], encoded["attention_mask"])


def test_coerce_model_inputs_builds_attention_mask_for_tensor() -> None:
    encoded = torch.tensor([[4, 5]], dtype=torch.long)

    model_inputs = _coerce_model_inputs(encoded, "cpu")

    assert torch.equal(model_inputs["input_ids"], encoded)
    assert torch.equal(model_inputs["attention_mask"], torch.ones_like(encoded))


def test_coerce_model_inputs_supports_batch_encoding_object() -> None:
    encoded = BatchEncoding(
        data={
            "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )

    model_inputs = _coerce_model_inputs(encoded, "cpu")

    assert torch.equal(model_inputs["input_ids"], torch.tensor([[7, 8, 9]], dtype=torch.long))
    assert torch.equal(model_inputs["attention_mask"], torch.tensor([[1, 1, 1]], dtype=torch.long))


def test_resolve_torch_dtype_supports_bfloat16() -> None:
    assert resolve_torch_dtype("bfloat16") == torch.bfloat16
