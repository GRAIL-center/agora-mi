# Qwen 2.5 7B Audit Evidence Experiment Result Review

## Run Status

The Qwen 2.5 7B auditor run completed for all 600 condition case pairs.

Run file:

`artifacts/audit_evidence_eval/reports/audit_reports_qwen25_7b_all_conditions_compact_final.jsonl`

Quality check outputs:

`artifacts/audit_evidence_eval/analysis/all_conditions_qwen25_7b_final/report_quality_summary.csv`

`artifacts/audit_evidence_eval/analysis/all_conditions_qwen25_7b_final/repair_note_summary.csv`

Gold scoring output:

`artifacts/audit_evidence_eval/scores/audit_report_scores_qwen25_7b_all_conditions.jsonl`

All 600 records parsed successfully after targeted retry of failed long white box outputs. There are no duplicate condition case keys and no empty finding lists.

## What Can Be Concluded Now

This run validates the experimental pipeline and reveals how the auditor model uses each evidence condition. It does not yet validate audit accuracy, because the gold brief file is still a human review template and all score records are correctly marked `needs_gold`.

Therefore, the current result supports operational claims only:

1. The evidence package format is runnable across all planned conditions.
2. Qwen 2.5 7B can produce structurally valid audit reports from every condition.
3. White box evidence increases tool citation and evidence uptake, but also increases formatting burden and citation repair.
4. Tool uptake is not the same as correctness, especially because the shuffled white box control also receives substantial citation uptake.

## Primary Comparison

Primary planned comparison:

`C6_full_whitebox` versus `C1_blackbox_surface`

Observed structural differences:

| Metric | C1 black box | C6 full white box | Interpretation |
| --- | ---: | ---: | --- |
| Parsed reports | 60 / 60 | 60 / 60 | Both are runnable after retry and validation. |
| Mean findings | 1.43 | 1.35 | White box evidence did not increase finding count. |
| Mean support spans | 1.82 | 1.47 | White box reports cited slightly fewer policy spans. |
| Mean cited evidence ids | 3.87 | 10.13 | White box reports cited far more tool evidence. |
| Mean repair notes | 0.03 | 0.75 | White box packages caused more format and citation cleanup. |
| Mean confidence | 0.85 | 0.83 | Confidence is similar, slightly lower for full white box. |
| Mean unsupported claims | 0.93 | 0.92 | Unsupported claim count is similar. |

Interpretation:

Full white box evidence clearly changes report behavior by increasing internal evidence uptake. However, without gold scoring, there is no basis yet to claim that it improves audit accuracy or groundedness. It may also shift attention away from direct policy span grounding.

## Ablation Pattern

Single tool and control conditions show important risks:

| Condition | Mean cited evidence ids | Mean repair notes | Main observation |
| --- | ---: | ---: | --- |
| C2 SAE only | 1.97 | 0.33 | SAE labels are usable but sparse coverage is limited. |
| C3 logit lens only | 9.23 | 0.17 | The auditor cites logit lens evidence heavily, even though it is only token direction evidence. |
| C4 steering only | 6.57 | 0.07 | Steering evidence is format stable and cited often. |
| C5 activation explanation surrogate | 1.85 | 0.68 | Surrogate explanations are less stable and trigger invalid evidence id repairs. |
| C8 raw AutoInterp SAE | 9.20 | 0.00 | Raw labels are easy for the auditor to cite, but this should not be interpreted as label correctness. |
| C9 shuffled white box control | 9.65 | 0.63 | The auditor still uses irrelevant white box evidence when supplied, so evidence uptake alone is not validity. |

The strongest methodological warning is C9. A report can look evidence rich while using evidence from the wrong case. This makes the human gold brief and traceability scoring essential.

## Hybrid Condition

`C7_hybrid_blackbox_whitebox` shows a different pattern:

Mean findings are higher than C1 and C6. It uses both text observations and some internal evidence, but with much lower repair burden than C6. It cites black box evidence most often, then SAE and logit lens evidence.

This suggests the hybrid package may be the most practical auditor interface, even if the main causal comparison remains C6 versus C1.

## Quality Problems Found And Fixed

Several issues were found during execution and corrected before the final Qwen result file:

1. Logit lens initially failed due PyTorch inference tensor behavior. Hidden states are now cloned before unembedding.
2. Logit lens tokens initially included non ASCII and vocabulary artifacts. Token filtering was added.
3. Full white box outputs initially truncated when the model listed too many evidence ids. Prompt limits and report normalization were added.
4. Long runs initially lost partial output on interruption. Report generation now streams one JSONL record at a time and supports resume.
5. Final Qwen all condition output had four parse failures before retry. Those exact condition case pairs were rerun with a larger token budget and replaced.

Final Qwen output status:

| Check | Result |
| --- | --- |
| Total records | 600 |
| Unique condition case keys | 600 |
| Parsed records | 600 |
| Empty findings | 0 |
| Bad confidence values | 0 |
| Final citation validity after repair | 1.0 in every condition |

## Current Scientific Reading

The current result does not yet prove that white box evidence improves audit quality. It proves that the experiment is executable and that white box evidence materially changes the auditor model evidence use pattern.

The most defensible interim claim is:

When given the same policy passages and a fixed audit schema, Qwen 2.5 7B can generate valid audit reports under black box and white box evidence conditions. Full white box packages increase internal evidence uptake, but also increase citation management burden. Because shuffled white box evidence is also heavily cited, correctness must be evaluated against human gold briefs rather than evidence uptake alone.

## Next Required Step

Fill human gold briefs for a pilot subset before making any accuracy claim.

Recommended pilot:

1. Select 12 to 20 cases.
2. Fill `gold_issue_tags`, `gold_actors`, `gold_obligations`, `gold_support_spans`, and `known_confounds`.
3. Rerun `score_audit_reports.py`.
4. Compare `C1_blackbox_surface`, `C6_full_whitebox`, `C7_hybrid_blackbox_whitebox`, and `C9_shuffled_whitebox_control`.

The key decision rule should be:

White box is useful only if C6 or C7 improves grounded audit quality over C1 and also beats C9. If C6 does not beat C9, the evidence package is causing citation uptake without valid diagnostic value.
