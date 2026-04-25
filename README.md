# White Box Evidence Packages for Policy Audit Reports

This branch contains a curated reproducibility package for the policy audit evidence interface experiment. It replaces the earlier broad repository contents with the files needed to inspect, reproduce, and rerun the audit evidence package study.

The study asks how the same policy passage and audit rubric produce different auditor reports when the auditor receives black box surface evidence, white box evidence, hybrid evidence, or shuffled white box control evidence.

## Included

The repository includes:

1. Source code for evidence package construction, local auditor runs, structural quality checks, human review analysis, and figure generation.
2. Final evidence packages for conditions C0 through C9.
3. Final local Qwen 2.5 7B auditor reports for all 60 cases and 10 conditions.
4. Human gold review files for the 20 case pilot over C1, C6, C7, and C9.
5. Structural summaries and paper figures used in the manuscript.
6. The ICML format paper source and bibliography.

The repository intentionally excludes model weights, caches, local environment files, smoke runs, retry files, logs, previous broad data products, and unrelated analysis scripts.

## Quick Start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify the included artifacts:

```bash
python scripts/verify_reproducibility.py
```

Regenerate the deterministic paper figures:

```bash
python paper/figures_src/build_audit_evidence_package_figures.py
```

Compile the paper:

```bash
cd paper
pdflatex -interaction=nonstopmode policy_audit_evidence_icml2026.tex
bibtex policy_audit_evidence_icml2026
pdflatex -interaction=nonstopmode policy_audit_evidence_icml2026.tex
pdflatex -interaction=nonstopmode policy_audit_evidence_icml2026.tex
```

## Main Artifacts

The central files are:

1. `artifacts/audit_evidence_eval/packages/`
2. `artifacts/audit_evidence_eval/reports/audit_reports_qwen25_7b_all_conditions_compact_final.jsonl`
3. `artifacts/audit_evidence_eval/analysis/all_conditions_qwen25_7b_final/report_quality_summary.csv`
4. `artifacts/audit_evidence_eval/human_review/policy_audit_human_review_summary.json`
5. `paper/policy_audit_evidence_icml2026.tex`

See `REPRODUCIBILITY.md` for the full reproduction path and claim boundary.
