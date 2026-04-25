# Gold Brief Review Instructions

## Purpose

Create human gold briefs for policy audit scoring. These gold briefs are the reference answers used to evaluate whether generated audit reports are accurate, grounded, and diagnostically useful.

## Blind Review Rule

Use only the policy passage and source metadata in the review sheet.

Do not inspect model reports, evidence packages, SAE labels, logit lens outputs, steering outputs, activation explanation outputs, or shuffled controls while filling gold briefs.

This avoids leaking treatment information into the gold labels.

## File To Fill

Primary pilot review sheet:

`artifacts/audit_evidence_eval/gold/gold_briefs_pilot_review_sheet.csv`

Each row is one policy passage.

## Columns To Fill

### gold_issue_tags

Short labels for the audit issues that a good auditor should identify.

Use semicolon separated values.

Example:

`notice requirement; opt out right; complaint handling gap`

### gold_actors

Actors who carry obligations, permissions, or risks in the passage.

Use semicolon separated values.

Example:

`algorithmic recommendation service providers; users; regulatory departments`

### gold_obligations

Concrete obligations, rights, risks, gaps, or audit relevant claims supported by the passage.

Use semicolon separated values.

Example:

`providers must notify users about recommendation mechanisms; users must be offered a non targeted option`

### gold_support_spans

Exact short spans copied from the passage. These should support the gold issues and obligations.

Use semicolon separated values.

Rules:

1. Copy exact text from the passage.
2. Keep each span short, usually one clause or sentence.
3. Include enough text to identify the obligation or risk.
4. Do not paraphrase in this field.

Example:

`providers shall notify users in a clear manner; provide users with options that are not specific to their personal characteristics`

### known_confounds

Alternative readings, limitations, or missing context that a careful auditor should mention.

Use semicolon separated values.

Example:

`the passage does not specify enforcement timeline; exact technical standard is not defined`

### unsupported_claims_to_penalize

Claims that a model should not make from this passage alone.

Use semicolon separated values.

Example:

`specific penalty amount; exact audit frequency; claim that the rule applies outside the covered service`

### reviewer_notes

Optional notes for later adjudication.

### needs_review

Set to `FALSE` when the row is complete.

Leave as `TRUE` only when the passage should not be scored yet.

## Review Standard

Prefer conservative gold labels.

A good gold brief should capture what is clearly supported by the passage, not every possible policy interpretation.

Include:

1. Explicit obligations.
2. Explicit rights or options.
3. Responsible actors.
4. Important risk controls.
5. Major gaps or ambiguities visible in the passage.

Do not include:

1. Background knowledge not visible in the passage.
2. Mechanistic claims about the model.
3. Evidence tool labels.
4. Broad policy family labels unless directly relevant.
5. Speculative harms not grounded in the text.

## Minimum Completion Target

For a quick pilot, complete 12 rows.

For a stronger pilot, complete 20 rows.

After review, convert the CSV to JSONL before scoring.
