Activation Oracle and Scaffolded Analyst Robustness Experiment Plan

Activation Oracle integration is supplementary validation for the 9B audit run.

Objectives
1. Test whether activation conditioned explanations improve policy audit interpretability over text only controls.
2. Test whether policy audit judgments remain stable under analyst style prompt scaffolds.
3. Preserve the main 2B paper spine by treating AO outputs as supplementary 9B evidence.

Primary configuration
1. Use `experiments/lambda_9b_robustness_partial.yaml`.
2. Keep `activation_oracle.enabled: true`.
3. Keep `audit.enabled: true` and `audit.scaffold_eval.enabled: true`.

Execution order
1. `policy-interp run-oracle-eval experiments/lambda_9b_robustness_partial.yaml`
2. `policy-interp export-reports experiments/lambda_9b_robustness_partial.yaml`

Generated artifacts
1. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_case_manifest.parquet`
2. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_scaffold_manifest.parquet`
3. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_predictions.parquet`
4. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_condition_summary.parquet`
5. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_scaffold_summary.parquet`
6. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_gold_labels.csv`
7. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_human_eval_sheet.csv`
8. `artifacts/lambda_9b_robustness_partial/oracle_eval/oracle_human_eval_summary.parquet`

Automatic evaluation
1. Compare `ao_real`, `ao_text_only`, and `ao_shuffled_activation`.
2. Report `regulatory_family_accuracy`, `primary_obligation_macro_f1`, `specificity_accuracy`, `best_proxy_agreement_rate`, and `policy_specificity_margin`.
3. Report scaffold retention on regulatory family, obligation family, and specificity.

Human evaluation
1. Sample `30` cases with family stratification.
2. Use conditions `base_audit_only`, `base_plus_ao`, and `ao_only`.
3. Ask annotators for regulatory family, primary obligation family, specificity label, usefulness, confidence, and time.

Paper integration
1. Add one compact AO condition summary table in the 9B supplementary validation subsection.
2. Add one scaffold retention figure based on `oracle_scaffold_summary.parquet`.
3. Add one appendix table from `oracle_human_eval_summary.parquet`.

Interpretation rule
1. AO should be described as an activation window and feature bundle explainer.
2. AO should not be described as a feature label replacement.
3. Weak AO results are still reportable as failure transparent supplementary evidence.
