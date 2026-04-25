"""Build proxy free audit evidence packages for the revised experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_interp.audit_evidence_suite import (  # noqa: E402
    build_evidence_packages,
    load_audit_evidence_inputs,
    write_evidence_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "artifacts" / "lambda_full_swap_2b_16k",
        help="Root containing taigr_revision_eval and paper_exports artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "audit_evidence_eval",
        help="Directory for generated evaluation packages.",
    )
    parser.add_argument("--case-manifest", type=Path, default=None, help="Optional case manifest CSV override.")
    parser.add_argument("--case-scores", type=Path, default=None, help="Optional case scores CSV override.")
    parser.add_argument("--manual-review", type=Path, default=None, help="Optional manual review CSV override.")
    parser.add_argument("--blackbox-observations", type=Path, default=None, help="Optional black box evidence JSONL or CSV.")
    parser.add_argument("--logit-lens", type=Path, default=None, help="Optional logit lens evidence JSONL or CSV.")
    parser.add_argument("--steering", type=Path, default=None, help="Optional steering evidence JSONL or CSV.")
    parser.add_argument("--activation-oracle", type=Path, default=None, help="Optional activation oracle evidence JSONL or CSV.")
    parser.add_argument("--case-limit", type=int, default=0, help="Deterministic subset size. Use 0 for all cases.")
    parser.add_argument("--pilot-case-count", type=int, default=12, help="Number of cases in the pilot manifest.")
    parser.add_argument("--max-sae-items", type=int, default=5, help="Maximum SAE evidence items per case.")
    parser.add_argument("--max-surface-items", type=int, default=5, help="Maximum black box surface items per case.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic case selection seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_audit_evidence_inputs(
        args.source_root,
        case_manifest_path=args.case_manifest,
        case_scores_path=args.case_scores,
        manual_review_path=args.manual_review,
        blackbox_observations_path=args.blackbox_observations,
        logit_lens_path=args.logit_lens,
        steering_path=args.steering,
        activation_oracle_path=args.activation_oracle,
    )
    packages = build_evidence_packages(
        inputs,
        case_limit=args.case_limit,
        max_sae_items=args.max_sae_items,
        max_surface_items=args.max_surface_items,
        seed=args.seed,
    )
    result = write_evidence_bundle(packages, args.output_root, pilot_case_count=args.pilot_case_count)
    print(f"Wrote audit evidence bundle to {result.output_root}")
    print(f"Case manifest: {result.case_manifest_path}")
    print(f"Gold template: {result.gold_template_path}")
    print(f"Package manifest: {result.package_manifest_path}")
    for condition_id, path in result.package_paths.items():
        print(f"{condition_id}: {path}")


if __name__ == "__main__":
    main()
