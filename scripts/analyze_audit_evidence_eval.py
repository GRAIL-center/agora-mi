"""Analyze audit evidence evaluation scores and package coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_interp.audit_evidence_suite import summarize_scores  # noqa: E402
from policy_interp.io import read_jsonl  # noqa: E402
from policy_interp.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "scores" / "audit_report_scores.jsonl",
        help="Score JSONL from scripts/score_audit_reports.py.",
    )
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "packages" / "package_manifest.csv",
        help="Package manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval" / "analysis",
        help="Output directory for analysis tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    scores = read_jsonl(args.scores) if args.scores.exists() else []
    score_summary = summarize_scores(scores)
    score_summary_path = output_dir / "score_summary_by_condition.csv"
    score_summary.to_csv(score_summary_path, index=False)

    coverage_path = output_dir / "package_coverage_by_condition.csv"
    if args.package_manifest.exists():
        manifest = pd.read_csv(args.package_manifest)
        coverage = (
            manifest.groupby(["condition_id", "condition_name"], dropna=False)
            .agg(
                package_count=("case_id", "count"),
                case_count=("case_id", "nunique"),
                mean_evidence_items=("evidence_item_count", "mean"),
                min_evidence_items=("evidence_item_count", "min"),
                max_evidence_items=("evidence_item_count", "max"),
            )
            .reset_index()
        )
    else:
        coverage = pd.DataFrame()
    coverage.to_csv(coverage_path, index=False)

    report_path = output_dir / "analysis_readme.md"
    report_path.write_text(
        "# Audit Evidence Evaluation Analysis\n\n"
        f"Score summary: `{score_summary_path}`\n\n"
        f"Package coverage: `{coverage_path}`\n\n"
        "If the score summary contains only `needs_gold`, fill and review "
        "`gold/gold_briefs_template.jsonl`, then rerun the scorer.\n",
        encoding="utf-8",
    )
    print(f"Wrote score summary to {score_summary_path}")
    print(f"Wrote package coverage to {coverage_path}")
    print(f"Wrote analysis note to {report_path}")


if __name__ == "__main__":
    main()
