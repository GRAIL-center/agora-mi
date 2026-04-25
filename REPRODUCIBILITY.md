# Reproducibility Notes

This repository supports three levels of reproduction.

## Level 1: Artifact Verification

Run:

```bash
python scripts/verify_reproducibility.py
```

This checks that the included artifacts match the main reported quantities:

1. 60 policy cases.
2. 10 evidence conditions.
3. 600 final auditor reports.
4. Citation validity of 1.0 for every condition in the structural summary.
5. Human review means for C1, C6, C7, and C9.
6. 11 C9 reports with both correctness at least 4 and evidence misuse at least 4.

This level does not require downloading model weights.

## Level 2: Analysis and Figure Reproduction

Regenerate structural quality summaries from final reports:

```bash
python scripts/audit_report_quality_checks.py \
  --reports artifacts/audit_evidence_eval/reports/audit_reports_qwen25_7b_all_conditions_compact_final.jsonl \
  --packages artifacts/audit_evidence_eval/packages \
  --output-dir artifacts/audit_evidence_eval/analysis/all_conditions_qwen25_7b_final
```

Regenerate paper figures:

```bash
python paper/figures_src/build_audit_evidence_package_figures.py
```

The transformer style Figure 1 asset is included as a static manuscript figure. The remaining summary figures are generated from the included CSV and JSON artifacts.

## Level 3: Model Rerun

The final reports can be regenerated with a local Hugging Face model:

```bash
python scripts/run_local_audit_reports.py \
  --packages artifacts/audit_evidence_eval/packages \
  --output artifacts/audit_evidence_eval/reports/audit_reports_qwen25_7b_all_conditions_compact_final.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-new-tokens 900
```

White box tool evidence can be regenerated with:

```bash
python scripts/build_logit_lens_evidence.py
python scripts/build_steering_vector_evidence.py
python scripts/build_activation_explanation_surrogate.py --dry-run
```

These commands require local compute and access to the relevant Hugging Face model repositories. The included artifacts are the reference outputs used for the manuscript.

## Claim Boundary

The repository is designed to reproduce the paper's evidence interface claim. It does not claim that full white box evidence improves audit quality. The supported result is that white box evidence changes evidence use, can improve traceability when anchored by surface passage evidence, and can increase evidence misuse when relevance is not checked.
