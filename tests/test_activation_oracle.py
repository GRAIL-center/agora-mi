from pathlib import Path

import pandas as pd

from policy_interp.activation_oracle import (
    AO_REAL_CONDITION,
    build_batch_oracle_report_summary,
    build_oracle_requests,
    build_scaffold_manifest,
    build_shuffled_activation_controls,
    parse_structured_oracle_response,
    summarize_oracle_predictions,
    activation_oracle_is_compatible,
)
from policy_interp.schemas import DatasetConfig, ExperimentConfig


def test_parse_structured_oracle_response_normalizes_labels() -> None:
    parsed = parse_structured_oracle_response(
        """
        {"concept_summary":"Focuses on privacy controls.","specificity_label":"Policy Specific",
         "obligation_family":"Privacy","regulatory_family":"privacy and data protection",
         "confidence":88,"rationale_short":"Privacy language dominates."}
        """,
        ["privacy_and_data_protection", "high_risk_ai_governance"],
    )

    assert parsed is not None
    assert parsed["specificity_label"] == "policy_specific"
    assert parsed["obligation_family"] == "privacy"
    assert parsed["regulatory_family"] == "privacy_and_data_protection"


def test_build_shuffled_activation_controls_uses_different_family() -> None:
    real = pd.DataFrame(
        [
            {
                "request_id": "a",
                "segment_id": "seg_a",
                "source_case_id": "case_a",
                "family_id": "family_a",
                "scaffold_frame": "raw_excerpt",
                "condition": AO_REAL_CONDITION,
                "unit_type": "feature_bundle",
                "activation_evidence": "privacy evidence",
                "request_status": "ready",
            },
            {
                "request_id": "b",
                "segment_id": "seg_b",
                "source_case_id": "case_b",
                "family_id": "family_b",
                "scaffold_frame": "raw_excerpt",
                "condition": AO_REAL_CONDITION,
                "unit_type": "feature_bundle",
                "activation_evidence": "bias evidence",
                "request_status": "ready",
            },
        ]
    )

    shuffled = build_shuffled_activation_controls(real, seed=17)

    assert len(shuffled) == 2
    assert set(shuffled["activation_donor_family_id"]) == {"family_a", "family_b"}
    assert set(shuffled["condition"]) == {"ao_shuffled_activation"}


def test_activation_oracle_compatibility_checks_model_family(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    assert not activation_oracle_is_compatible(config)
    config.backbone.model_name = "google/gemma-2-9b"
    assert activation_oracle_is_compatible(config)


def test_build_scaffold_manifest_expands_fixed_frames(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    source = pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "family_id": "family_a",
                "family_label": "Family A",
                "text": "A statutory excerpt.",
            }
        ]
    )

    manifest = build_scaffold_manifest(source, config)

    assert set(manifest["scaffold_frame"]) == {
        "raw_excerpt",
        "analyst_question",
        "compliance_memo",
        "neutral_restatement",
        "adversarial_bland",
    }
    assert manifest["source_case_id"].nunique() == 1


def test_build_batch_oracle_report_summary_marks_agreement() -> None:
    bundle = pd.DataFrame(
        [
            {
                "segment_id": "seg_a",
                "source_case_id": "seg_a",
                "policy_signal_total": 4.0,
                "concept_summary": "Privacy bundle",
                "specificity_label": "policy_specific",
                "obligation_family": "privacy",
                "regulatory_family": "privacy_and_data_protection",
                "confidence": 90,
                "rationale_short": "Privacy spans dominate.",
            }
        ]
    )
    window = pd.DataFrame(
        [
            {
                "segment_id": "seg_a",
                "source_case_id": "seg_a",
                "concept_summary": "Privacy window",
                "specificity_label": "policy_specific",
                "obligation_family": "privacy",
                "regulatory_family": "privacy_and_data_protection",
                "confidence": 84,
                "rationale_short": "Same signal.",
            }
        ]
    )

    summary = build_batch_oracle_report_summary(bundle, window)

    assert len(summary) == 1
    assert summary.iloc[0]["agreement_status"] == "agree"


def test_build_oracle_requests_adds_controls_and_bundle_selection(tmp_path: Path) -> None:
    config = ExperimentConfig(name="unit_test", dataset=DatasetConfig(base_dir=tmp_path))
    config.activation_oracle.enabled = True
    manifest = pd.DataFrame(
        [
            {
                "segment_id": "seg_a",
                "source_case_id": "case_a",
                "family_id": "family_a",
                "family_label": "Family A",
                "scaffold_frame": "raw_excerpt",
                "text": "Privacy and data governance obligations apply.",
            },
            {
                "segment_id": "seg_b",
                "source_case_id": "case_b",
                "family_id": "family_b",
                "family_label": "Family B",
                "scaffold_frame": "raw_excerpt",
                "text": "Bias and transparency obligations apply.",
            },
        ]
    )
    scored = pd.DataFrame(
        [
            {"segment_id": "seg_a", "text": "Privacy and data governance obligations apply.", "ranking_family": "policy_specific", "layer": 20, "feature_id": 1, "pooled_activation": 3.0, "best_proxy": "privacy", "generated_name": "Privacy Controls", "top_token_span_text": "privacy and data"},
            {"segment_id": "seg_a", "text": "Privacy and data governance obligations apply.", "ranking_family": "policy_specific", "layer": 20, "feature_id": 2, "pooled_activation": 2.5, "best_proxy": "privacy", "generated_name": "Data Governance", "top_token_span_text": "data governance"},
            {"segment_id": "seg_a", "text": "Privacy and data governance obligations apply.", "ranking_family": "policy_specific", "layer": 24, "feature_id": 3, "pooled_activation": 2.0, "best_proxy": "privacy", "generated_name": "Sensitive Data", "top_token_span_text": "sensitive data"},
            {"segment_id": "seg_b", "text": "Bias and transparency obligations apply.", "ranking_family": "policy_specific", "layer": 20, "feature_id": 4, "pooled_activation": 3.0, "best_proxy": "bias", "generated_name": "Bias Controls", "top_token_span_text": "bias controls"},
            {"segment_id": "seg_b", "text": "Bias and transparency obligations apply.", "ranking_family": "policy_specific", "layer": 20, "feature_id": 5, "pooled_activation": 2.0, "best_proxy": "transparency", "generated_name": "Transparency Notes", "top_token_span_text": "transparency notes"},
            {"segment_id": "seg_b", "text": "Bias and transparency obligations apply.", "ranking_family": "policy_specific", "layer": 24, "feature_id": 6, "pooled_activation": 1.8, "best_proxy": "bias", "generated_name": "Fairness Review", "top_token_span_text": "fairness review"},
        ]
    )

    requests = build_oracle_requests(scored, manifest, config, include_controls=True)

    assert {"ao_real", "ao_text_only", "ao_shuffled_activation"} == set(requests["condition"])
    assert {"feature_bundle", "activation_window"} == set(requests["unit_type"])
    assert (requests.loc[requests["condition"] == "ao_real", "request_status"] == "ready").all()


def test_summarize_oracle_predictions_computes_retention_and_margin() -> None:
    predictions = pd.DataFrame(
        [
            {"source_case_id": "case_a", "condition": "ao_real", "unit_type": "feature_bundle", "scaffold_frame": "raw_excerpt", "regulatory_family": "family_a", "obligation_family": "privacy", "specificity_label": "policy_specific", "dominant_proxy": "privacy"},
            {"source_case_id": "case_a", "condition": "ao_real", "unit_type": "feature_bundle", "scaffold_frame": "analyst_question", "regulatory_family": "family_a", "obligation_family": "privacy", "specificity_label": "policy_specific", "dominant_proxy": "privacy"},
            {"source_case_id": "case_a", "condition": "ao_text_only", "unit_type": "feature_bundle", "scaffold_frame": "raw_excerpt", "regulatory_family": "family_a", "obligation_family": "privacy", "specificity_label": "generic_legalese", "dominant_proxy": "privacy"},
            {"source_case_id": "case_a", "condition": "ao_text_only", "unit_type": "feature_bundle", "scaffold_frame": "analyst_question", "regulatory_family": "family_b", "obligation_family": "governance_other", "specificity_label": "generic_legalese", "dominant_proxy": "privacy"},
        ]
    )
    gold = pd.DataFrame(
        [
            {"source_case_id": "case_a", "regulatory_family": "family_a", "primary_obligation_family": "privacy", "specificity_label": "policy_specific"}
        ]
    )

    condition_summary, scaffold_summary = summarize_oracle_predictions(predictions, gold)

    real_row = condition_summary.loc[condition_summary["condition"] == "ao_real"].iloc[0]
    assert float(real_row["policy_specificity_margin"]) > 0.0
    assert float(real_row["best_proxy_agreement_rate"]) == 1.0
    retention_row = scaffold_summary.loc[scaffold_summary["condition"] == "ao_real"].iloc[0]
    assert float(retention_row["regulatory_family_retention_rate"]) == 1.0
