"""Verify the included audit evidence reproducibility artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "audit_evidence_eval"


EXPECTED_CONDITIONS = [
    "C0_passage_only",
    "C1_blackbox_surface",
    "C2_sae_only",
    "C3_logit_lens_only",
    "C4_steering_only",
    "C5_activation_oracle_only",
    "C6_full_whitebox",
    "C7_hybrid_blackbox_whitebox",
    "C8_raw_autointerp_sae",
    "C9_shuffled_whitebox_control",
]


EXPECTED_HUMAN_MEANS = {
    "C1_blackbox_surface": {
        "report_correctness_1_5_avg": 4.30,
        "span_grounding_1_5_avg": 3.75,
        "diagnostic_usefulness_1_5_avg": 4.30,
        "evidence_misuse_1_5_avg": 1.00,
    },
    "C6_full_whitebox": {
        "report_correctness_1_5_avg": 3.85,
        "span_grounding_1_5_avg": 2.85,
        "diagnostic_usefulness_1_5_avg": 3.85,
        "evidence_misuse_1_5_avg": 2.55,
    },
    "C7_hybrid_blackbox_whitebox": {
        "report_correctness_1_5_avg": 4.55,
        "span_grounding_1_5_avg": 3.80,
        "diagnostic_usefulness_1_5_avg": 4.60,
        "evidence_misuse_1_5_avg": 1.55,
    },
    "C9_shuffled_whitebox_control": {
        "report_correctness_1_5_avg": 3.55,
        "span_grounding_1_5_avg": 2.60,
        "diagnostic_usefulness_1_5_avg": 3.55,
        "evidence_misuse_1_5_avg": 4.35,
    },
}


EXPECTED_CITATION_UPTAKE = {
    "C1_blackbox_surface": 3.8666666667,
    "C6_full_whitebox": 10.1333333333,
    "C7_hybrid_blackbox_whitebox": 7.5166666667,
    "C9_shuffled_whitebox_control": 9.65,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def assert_close(actual: float, expected: float, *, label: str, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")


def verify_cases() -> None:
    cases = pd.read_csv(require(ARTIFACT_ROOT / "cases" / "case_manifest.csv"))
    if len(cases) != 60:
        raise AssertionError(f"Expected 60 cases, observed {len(cases)}")
    if cases["case_id"].nunique() != 60:
        raise AssertionError("Case ids are not unique.")


def verify_packages() -> None:
    package_dir = require(ARTIFACT_ROOT / "packages")
    for condition in EXPECTED_CONDITIONS:
        records = read_jsonl(require(package_dir / f"{condition}.jsonl"))
        if len(records) != 60:
            raise AssertionError(f"{condition}: expected 60 packages, observed {len(records)}")
        observed = {record.get("condition_id") for record in records}
        if observed != {condition}:
            raise AssertionError(f"{condition}: unexpected condition ids {sorted(observed)}")


def verify_reports() -> None:
    reports = read_jsonl(
        require(ARTIFACT_ROOT / "reports" / "audit_reports_qwen25_7b_all_conditions_compact_final.jsonl")
    )
    if len(reports) != 600:
        raise AssertionError(f"Expected 600 reports, observed {len(reports)}")
    keys = {(record.get("case_id"), record.get("condition_id")) for record in reports}
    if len(keys) != 600:
        raise AssertionError("Report case and condition keys are not unique.")
    parse_statuses = {record.get("parse_status") for record in reports}
    if parse_statuses != {"parsed"}:
        raise AssertionError(f"Unexpected parse statuses: {sorted(parse_statuses)}")


def verify_structural_summary() -> None:
    summary = pd.read_csv(
        require(ARTIFACT_ROOT / "analysis" / "all_conditions_qwen25_7b_final" / "report_quality_summary.csv")
    ).set_index("condition_id")
    missing = set(EXPECTED_CONDITIONS) - set(summary.index)
    if missing:
        raise AssertionError(f"Missing structural summary conditions: {sorted(missing)}")
    if not (summary["citation_validity_mean"].astype(float) == 1.0).all():
        raise AssertionError("Not all citation validity means are 1.0.")
    for condition, expected in EXPECTED_CITATION_UPTAKE.items():
        actual = float(summary.loc[condition, "cited_evidence_id_count_mean"])
        assert_close(actual, expected, label=f"{condition} cited evidence id mean", tol=1e-5)


def verify_human_review() -> None:
    summary = json.loads(
        require(ARTIFACT_ROOT / "human_review" / "policy_audit_human_review_summary.json").read_text(encoding="utf-8")
    )
    if int(summary.get("gold_rows", 0)) != 20:
        raise AssertionError("Human review summary does not contain 20 gold rows.")
    if int(summary.get("diagnostic_rows", 0)) != 80:
        raise AssertionError("Human review summary does not contain 80 diagnostic rows.")
    by_condition = {row["condition_id"]: row for row in summary["condition_summary"]}
    for condition, expected_values in EXPECTED_HUMAN_MEANS.items():
        row = by_condition[condition]
        if int(row["n"]) != 20:
            raise AssertionError(f"{condition}: expected n=20, observed {row['n']}")
        for metric, expected in expected_values.items():
            assert_close(float(row[metric]), expected, label=f"{condition} {metric}")

    diagnostic = pd.read_csv(
        require(ARTIFACT_ROOT / "human_review" / "diagnostic_review_sheet_after_gold_human_filled.csv")
    )
    c9 = diagnostic[diagnostic["condition_id"] == "C9_shuffled_whitebox_control"].copy()
    high_correct_high_misuse = c9[
        (pd.to_numeric(c9["report_correctness_1_5"]) >= 4)
        & (pd.to_numeric(c9["evidence_misuse_1_5"]) >= 4)
    ]
    if len(high_correct_high_misuse) != 11:
        raise AssertionError(f"Expected 11 high correctness and high misuse C9 cases, observed {len(high_correct_high_misuse)}")


def main() -> None:
    checks = [
        verify_cases,
        verify_packages,
        verify_reports,
        verify_structural_summary,
        verify_human_review,
    ]
    for check in checks:
        check()
        print(f"ok: {check.__name__}")
    print("All reproducibility checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise
