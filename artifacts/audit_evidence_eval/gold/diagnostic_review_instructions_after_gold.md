# Diagnostic Review Instructions After Gold Briefs

## Purpose

This second pass evaluates what the generated audit reports and evidence packages reveal after the blind gold briefs are complete.

Do not open this diagnostic sheet until the blind gold review has been completed for the same cases.

## File To Fill

`artifacts/audit_evidence_eval/gold/diagnostic_review_sheet_after_gold.csv`

This file includes four conditions for each pilot case:

1. `C1_blackbox_surface`
2. `C6_full_whitebox`
3. `C7_hybrid_blackbox_whitebox`
4. `C9_shuffled_whitebox_control`

## Rating Columns

### report_correctness_1_5

How correct is the report relative to the passage and your gold brief?

1 means mostly wrong or misleading.

3 means partially correct but incomplete or over broad.

5 means substantively correct.

### span_grounding_1_5

How well are the report claims grounded in exact policy spans?

1 means weak or unsupported grounding.

3 means some span support but with missing or loose citations.

5 means claims are clearly grounded in the passage.

### diagnostic_usefulness_1_5

How useful is this report for a policy auditor trying to find obligations, risks, gaps, or ambiguities?

1 means not useful.

3 means somewhat useful but not enough to trust alone.

5 means highly useful.

### evidence_misuse_1_5

How much does the report misuse the provided evidence?

1 means no obvious misuse.

3 means some over interpretation.

5 means severe misuse, such as treating weak or irrelevant evidence as proof.

For `C9_shuffled_whitebox_control`, mark evidence misuse high if the report relies on white box evidence that does not match the passage.

## Text Columns

### whitebox_adds_nonblackbox_information

Answer `yes`, `no`, or `unclear`.

Use `yes` only when white box evidence reveals something useful that the black box condition does not show, such as:

1. A model internal concept that aligns with a policy obligation.
2. A hidden over reliance on generic risk language.
3. A mismatch between internal evidence and passage level claim.
4. A traceable failure mode that black box output alone would hide.

### whitebox_failure_mode

Describe the main white box failure mode, if any.

Examples:

1. `over cites logit lens token evidence`
2. `uses shuffled evidence as if it were relevant`
3. `SAE label is too broad`
4. `steering evidence is suggestive but not passage specific`
5. `no clear failure`

### reviewer_notes

Optional short notes for adjudication.

## How To Use This For The Paper

The goal is not only to ask whether white box evidence increases audit quality.

The stronger governance question is:

Does white box evidence expose diagnostic information, limitations, or misuse risks that a black box audit would hide?

The most important comparison is:

`C6_full_whitebox` and `C7_hybrid_blackbox_whitebox` should outperform `C9_shuffled_whitebox_control` on correctness and evidence misuse.

If they do not, that is still a meaningful result. It means white box evidence needs validation controls before it can be trusted in policy audit.
