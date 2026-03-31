# ICML Revision Lambda Runbook

## Scope

This runbook defines the locked experiment package for the ICML revision of the policy proxy SAE paper.

1. The paper source of truth is a locked Gemma 2 2B package.
2. The replication package is a sparse Gemma 2 9B run on the two primary pairs only.
3. Large experiments run on Lambda, not on local hardware.
4. If the locked Lambda package weakens causal evidence, the manuscript must downgrade to localization and stability first.

## Locked 2B Benchmark Package

Run the full benchmark with the reproducibility and export extensions enabled.

```bash
cd ~/agora-mi && python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark_locked_2b.yaml --output_root results/policy_feature_benchmark_locked_2b && python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_2b.yaml --output_root results/policy_feature_benchmark_locked_2b
```

Aggregate again after the full run so all summary exports are fresh.

```bash
cd ~/agora-mi && python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_2b.yaml --output_root results/policy_feature_benchmark_locked_2b
```

## Dense Intervention Control

Run the dense subspace control on the two headline causal candidates after the sparse benchmark finishes.

```bash
cd ~/agora-mi && python scripts/run_dense_subspace_control_eval.py --config configs/policy_mech_interp.yaml --manifest_root data/processed/public_values --family equality_neutrality --proxy_slug risk_factors_bias --paired_proxy_slug harms_discrimination --layer 20 --site resid_post --split test --max_samples 100 --random_sets 100 --k 3 && python scripts/run_dense_subspace_control_eval.py --config configs/policy_mech_interp.yaml --manifest_root data/processed/public_values --family individual_rights --proxy_slug risk_factors_privacy --paired_proxy_slug harms_violation_of_civil_or_human_rights_including_privacy --layer 24 --site resid_post --split test --max_samples 100 --random_sets 100 --k 3
```

## Locked 2B Assistant Package

Run the downstream assistant layer after the benchmark package is complete.

```bash
cd ~/agora-mi && python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_locked_2b.yaml --output_root results/policy_analysis_assistant_locked_2b
```

Optional prototype card export:

```bash
cd ~/agora-mi && python scripts/run_policy_document_analysis.py --input_path sample_policy_doc.txt --config configs/policy_analysis_assistant_locked_2b.yaml --output_path results/policy_document_analysis_locked_2b.json --document_id lambda_doc_1 --title "Lambda Document" --source_type lambda_text
```

## 9B Replication Package

Run the sparse mechanistic package on the two primary pairs only.

```bash
cd ~/agora-mi && python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark_locked_9b_primary.yaml --output_root results/policy_feature_benchmark_locked_9b_primary && python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_9b_primary.yaml --output_root results/policy_feature_benchmark_locked_9b_primary && python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_9b_primary.yaml --output_root results/policy_feature_benchmark_locked_9b_primary
```

## Required 2B Benchmark Artifacts

The locked 2B package is incomplete unless the following files exist.

1. `results/policy_feature_benchmark_locked_2b/summary/table_main_benchmark_results.csv`
2. `results/policy_feature_benchmark_locked_2b/summary/table_proxy_mechanistic_evidence.csv`
3. `results/policy_feature_benchmark_locked_2b/summary/table_pair_transfer_and_causality.csv`
4. `results/policy_feature_benchmark_locked_2b/summary/proxy_feature_summary.csv`
5. `results/policy_feature_benchmark_locked_2b/summary/proxy_causal_summary.csv`
6. `results/policy_feature_benchmark_locked_2b/summary/proxy_causal_samples.csv`
7. `results/policy_feature_benchmark_locked_2b/summary/proxy_causal_random_controls.csv`
8. `results/policy_feature_benchmark_locked_2b/summary/proxy_off_target_effects.csv`
9. `results/policy_feature_benchmark_locked_2b/summary/negative_matching_diagnostics.csv`
10. `results/policy_feature_benchmark_locked_2b/summary/feature_concept_cards.jsonl`

## Required Assistant Artifacts

1. `results/policy_analysis_assistant_locked_2b/summary/assistant_leaderboard.json`
2. `results/policy_analysis_assistant_locked_2b/summary/assistant_feature_cards.jsonl`
3. `results/policy_analysis_assistant_locked_2b/summary/assistant_feature_usage.csv`
4. `results/policy_analysis_assistant_locked_2b/summary/assistant_card_dossier_links.jsonl`

## Required 9B Replication Artifacts

1. `results/policy_feature_benchmark_locked_9b_primary/summary/table_proxy_mechanistic_evidence.csv`
2. `results/policy_feature_benchmark_locked_9b_primary/summary/table_pair_transfer_and_causality.csv`
3. `results/policy_feature_benchmark_locked_9b_primary/summary/proxy_feature_summary.csv`
4. `results/policy_feature_benchmark_locked_9b_primary/summary/proxy_causal_summary.csv`

## Acceptance Rules

Promote evidence to the main paper only if the locked package satisfies all of the following conditions.

1. At least two proxies in the primary pairs remain positive on selective causal support in 2B.
2. At least one directed transfer in the primary pairs remains positive in the locked 2B run.
3. At least one of the two headline causal cases replicates directionally in 9B.
4. Dense intervention control does not erase the sparse selectivity argument.

If these conditions fail, downgrade the paper to localization and stability first and move causal claims to exploratory analysis.
