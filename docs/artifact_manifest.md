# Artifact Manifest

This manifest lists what is included and why.

## Code

`src/policy_interp/audit_evidence_suite.py` contains the core evidence package definitions, prompt rendering, report normalization, and scoring helpers.

`scripts/` contains only scripts needed for the audit evidence package experiment:

1. `build_audit_evidence_eval.py`
2. `run_local_audit_reports.py`
3. `audit_report_quality_checks.py`
4. `analyze_audit_evidence_eval.py`
5. `build_logit_lens_evidence.py`
6. `build_steering_vector_evidence.py`
7. `build_activation_explanation_surrogate.py`
8. `score_audit_reports.py`
9. `verify_reproducibility.py`

## Data and Results

`artifacts/audit_evidence_eval/cases/` contains the 60 case manifest and the pilot case manifest.

`artifacts/audit_evidence_eval/packages/` contains final C0 through C9 evidence packages.

`artifacts/audit_evidence_eval/reports/` contains the final 600 report JSONL.

`artifacts/audit_evidence_eval/analysis/` contains structural summaries used by the paper.

`artifacts/audit_evidence_eval/human_review/` contains the filled human review summary and diagnostic tables used for pilot quality results.

`artifacts/audit_evidence_eval/tool_evidence/` contains compact logit lens, steering, and activation hypothesis evidence files.

## Paper

`paper/policy_audit_evidence_icml2026.tex` is the manuscript source. The figure folder contains only the audit evidence package figures used by this manuscript.

## Excluded

The branch excludes previous broad public values data, unrelated proxy experiments, raw model caches, local environment files, LaTeX build byproducts, retry outputs, smoke outputs, and model weights.
