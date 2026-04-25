"""Quality checks for generated audit reports before human gold scoring."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_interp.io import read_jsonl  # noqa: E402
from policy_interp.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "reports" / "audit_reports_qwen25_7b_primary_C1_C6_final.jsonl",
    )
    parser.add_argument(
        "--packages",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "packages",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "analysis" / "primary_qwen25_7b",
    )
    return parser.parse_args()


def load_packages(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("C*.jsonl"))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for file_path in files:
        for package in read_jsonl(file_path):
            index[(str(package.get("case_id")), str(package.get("condition_id")))] = package
    return index


def report_cited_ids(report: dict[str, Any]) -> list[str]:
    cited: list[str] = []
    for finding in report.get("issue_findings", []) or []:
        if isinstance(finding, dict):
            raw_ids = finding.get("evidence_ids") or []
            if isinstance(raw_ids, list):
                cited.extend(str(item) for item in raw_ids if str(item).strip())
    internal = report.get("internal_evidence_used") or []
    if isinstance(internal, list):
        cited.extend(str(item) for item in internal if str(item).strip())
    return cited


def tool_from_evidence_id(package: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("evidence_id")): str(item.get("tool"))
        for item in package.get("evidence_items", [])
        if item.get("evidence_id")
    }


def collect_rows(reports: list[dict[str, Any]], packages: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in reports:
        case_id = str(record.get("case_id"))
        condition_id = str(record.get("condition_id"))
        package = packages.get((case_id, condition_id), {})
        report = record.get("report") if isinstance(record.get("report"), dict) else {}
        findings = report.get("issue_findings", []) if isinstance(report, dict) else []
        if not isinstance(findings, list):
            findings = []
        cited = report_cited_ids(report)
        evidence_tool = tool_from_evidence_id(package)
        valid_cited = [item for item in cited if item in evidence_tool]
        tool_counts: dict[str, int] = {}
        for evidence_id in valid_cited:
            tool = evidence_tool[evidence_id]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        rows.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "parse_status": record.get("parse_status"),
                "repair_note_count": len(record.get("repair_notes") or []),
                "repair_notes": ";".join(record.get("repair_notes") or []),
                "raw_output_chars": len(str(record.get("raw_output", ""))),
                "finding_count": len(findings),
                "supporting_span_count": sum(
                    len(finding.get("supporting_policy_spans") or [])
                    for finding in findings
                    if isinstance(finding, dict)
                ),
                "finding_evidence_id_count": sum(
                    len(finding.get("evidence_ids") or [])
                    for finding in findings
                    if isinstance(finding, dict)
                ),
                "internal_evidence_used_count": len(report.get("internal_evidence_used") or []),
                "cited_evidence_id_count": len(cited),
                "valid_cited_evidence_id_count": len(valid_cited),
                "citation_validity": len(valid_cited) / len(cited) if cited else 1.0,
                "unsupported_claim_count": len(report.get("unsupported_or_low_confidence_claims") or []),
                "overall_confidence": report.get("overall_confidence"),
                "package_evidence_item_count": len(package.get("evidence_items", [])),
                "blackbox_id_count": tool_counts.get("blackbox", 0),
                "sae_id_count": tool_counts.get("sae", 0),
                "logit_lens_id_count": tool_counts.get("logit_lens", 0),
                "steering_vector_id_count": tool_counts.get("steering_vector", 0),
                "activation_oracle_surrogate_id_count": tool_counts.get("activation_oracle_surrogate", 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    reports = read_jsonl(args.reports)
    packages = load_packages(args.packages)
    detail = collect_rows(reports, packages)
    detail_path = output_dir / "report_quality_detail.csv"
    detail.to_csv(detail_path, index=False)

    numeric_cols = [
        "repair_note_count",
        "raw_output_chars",
        "finding_count",
        "supporting_span_count",
        "finding_evidence_id_count",
        "internal_evidence_used_count",
        "cited_evidence_id_count",
        "citation_validity",
        "unsupported_claim_count",
        "overall_confidence",
        "package_evidence_item_count",
        "blackbox_id_count",
        "sae_id_count",
        "logit_lens_id_count",
        "steering_vector_id_count",
        "activation_oracle_surrogate_id_count",
    ]
    summary = (
        detail.groupby("condition_id", dropna=False)[numeric_cols]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    summary_path = output_dir / "report_quality_summary.csv"
    summary.to_csv(summary_path, index=False)

    parse_summary = (
        detail.groupby(["condition_id", "parse_status"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    parse_path = output_dir / "parse_status_summary.csv"
    parse_summary.to_csv(parse_path, index=False)

    repair_rows = []
    for _, row in detail.iterrows():
        notes = str(row.get("repair_notes") or "")
        if not notes:
            continue
        for note in notes.split(";"):
            if note:
                repair_rows.append({"condition_id": row["condition_id"], "repair_note": note})
    repair_summary = pd.DataFrame(repair_rows)
    if not repair_summary.empty:
        repair_summary = (
            repair_summary.groupby(["condition_id", "repair_note"], dropna=False)
            .size()
            .reset_index(name="count")
        )
    repair_path = output_dir / "repair_note_summary.csv"
    repair_summary.to_csv(repair_path, index=False)

    note_path = output_dir / "quality_check_readme.md"
    note_path.write_text(
        "# Audit Report Quality Checks\n\n"
        "These checks evaluate generated report structure before human gold scoring. "
        "They do not measure audit accuracy.\n\n"
        f"Detail table: `{detail_path}`\n\n"
        f"Condition summary: `{summary_path}`\n\n"
        f"Parse summary: `{parse_path}`\n\n"
        f"Repair summary: `{repair_path}`\n",
        encoding="utf-8",
    )
    print(f"Wrote detail to {detail_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote parse summary to {parse_path}")
    print(f"Wrote repair summary to {repair_path}")


if __name__ == "__main__":
    main()
