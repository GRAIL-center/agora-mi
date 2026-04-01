# Reviewer Hardening Runbook

This runbook collects the new controls added after the weak reject review.

## 1. Build null SAE dictionaries

Random matched norm dictionary for 2B Layer 24 `resid_post`:

```bash
python scripts/build_null_sae_npz.py \
  --config configs/policy_mech_interp.yaml \
  --layer 24 \
  --site resid_post \
  --mode random_dictionary \
  --output_path results/sae_nulls/gemma2_2b_random_dictionary/layer_24_resid_post.npz
```

Decoder tied null for 2B Layer 24 `resid_post`:

```bash
python scripts/build_null_sae_npz.py \
  --config configs/policy_mech_interp.yaml \
  --layer 24 \
  --site resid_post \
  --mode tied_encoder_null \
  --output_path results/sae_nulls/gemma2_2b_tied_encoder_null/layer_24_resid_post.npz
```

Soft frozen decoder null for 2B Layer 24 `resid_post`:

```bash
python scripts/build_null_sae_npz.py \
  --config configs/policy_mech_interp.yaml \
  --layer 24 \
  --site resid_post \
  --mode soft_frozen_decoder \
  --encoder_blend_alpha 0.5 \
  --output_path results/sae_nulls/gemma2_2b_soft_frozen_decoder/layer_24_resid_post.npz
```

Repeat for Layers `12,16,20,24` and any additional site supported by available SAEs.

## 2. Run null benchmark baselines

Random dictionary benchmark:

```bash
python scripts/run_policy_feature_benchmark.py \
  --config configs/policy_feature_benchmark_null_random_2b.yaml \
  --output_root results/policy_feature_benchmark_null_random_2b
```

Decoder tied null benchmark:

```bash
python scripts/run_policy_feature_benchmark.py \
  --config configs/policy_feature_benchmark_null_tied_2b.yaml \
  --output_root results/policy_feature_benchmark_null_tied_2b
```

Soft frozen decoder benchmark:

```bash
python scripts/run_policy_feature_benchmark.py \
  --config configs/policy_feature_benchmark_null_soft_frozen_2b.yaml \
  --output_root results/policy_feature_benchmark_null_soft_frozen_2b
```

## 3. Run sparse causal sensitivity

Privacy example:

```bash
python scripts/run_proxy_causal_sensitivity_suite.py \
  --config configs/policy_mech_interp.yaml \
  --prompt_config configs/policy_text_prompt_sensitivity.yaml \
  --label_variant_config configs/policy_label_variants.yaml \
  --manifest_root data/processed/public_values \
  --family individual_rights \
  --proxy_slug risk_factors_privacy \
  --paired_proxy_slug harms_violation_of_civil_or_human_rights_including_privacy \
  --layer 24 \
  --site resid_post \
  --feature_ids 245,11529,261,6280,9460 \
  --k_values 1,3,5 \
  --prompt_template_keys proxy_forced_choice_template,proxy_forced_choice_template_compact,proxy_forced_choice_template_evidence_first \
  --label_normalizations sum,mean_token,mean_char \
  --output_dir results/reviewer_hardening/privacy_sensitivity_2b
```

Bias example:

```bash
python scripts/run_proxy_causal_sensitivity_suite.py \
  --config configs/policy_mech_interp.yaml \
  --prompt_config configs/policy_text_prompt_sensitivity.yaml \
  --label_variant_config configs/policy_label_variants.yaml \
  --manifest_root data/processed/public_values \
  --family equality_neutrality \
  --proxy_slug risk_factors_bias \
  --paired_proxy_slug harms_discrimination \
  --layer 24 \
  --site resid_post \
  --feature_ids 5081,261,4387,6631,8450 \
  --k_values 1,3,5 \
  --prompt_template_keys proxy_forced_choice_template,proxy_forced_choice_template_compact,proxy_forced_choice_template_evidence_first \
  --label_normalizations sum,mean_token,mean_char \
  --output_dir results/reviewer_hardening/bias_sensitivity_2b
```

## 4. Run dense controls with stronger calibration

Energy weighted dense control:

```bash
python scripts/run_dense_subspace_control_eval.py \
  --config configs/policy_mech_interp.yaml \
  --manifest_root data/processed/public_values \
  --family individual_rights \
  --proxy_slug risk_factors_privacy \
  --paired_proxy_slug harms_violation_of_civil_or_human_rights_including_privacy \
  --layer 24 \
  --site resid_post \
  --split test \
  --max_samples 100 \
  --random_sets 100 \
  --k 3 \
  --selection_mode topk_energy_weighted \
  --prompt_config configs/policy_text_prompt_sensitivity.yaml \
  --label_normalization mean_token
```

Target energy calibrated dense control:

```bash
python scripts/run_dense_subspace_control_eval.py \
  --config configs/policy_mech_interp.yaml \
  --manifest_root data/processed/public_values \
  --family individual_rights \
  --proxy_slug risk_factors_privacy \
  --paired_proxy_slug harms_violation_of_civil_or_human_rights_including_privacy \
  --layer 24 \
  --site resid_post \
  --split test \
  --max_samples 100 \
  --random_sets 100 \
  --k 3 \
  --selection_mode target_energy \
  --target_ablation_energy 0.25
```

## 5. Stronger masking controls

The benchmark now supports:

1. `keyword_mask`
2. `expanded_keyword_mask`
3. `char_mask`
4. `expanded_char_mask`

Switch the benchmark masking block, for example:

```yaml
masking:
  keyword_source: proxy
  strategy: expanded_keyword_mask
  additional_strategies: [char_mask]
```

The registry will expand proxy terms with simple morphology and proxy specific synonyms.

You can also run a full masking suite:

```bash
python scripts/run_masking_sensitivity_suite.py \
  --benchmark_config configs/policy_feature_benchmark_locked_2b.yaml \
  --output_root results/reviewer_hardening/masking_suite_2b \
  --suite keyword_mask expanded_keyword_mask expanded_char_mask:char_mask
```

## 6. Build qualitative and stable-proxy diagnostics

```bash
python scripts/analyze_locked_proxy_diagnostics.py \
  --bundle_root /path/to/icml_all_results_20260401 \
  --manifest_root data/processed/public_values \
  --output_root results/reviewer_hardening/locked_proxy_diagnostics
```

This writes:

1. `feature_semantic_inventory.csv`
2. `feature_absorption_diagnostics.csv`
3. `stable_noncausal_proxy_diagnostics.csv`
4. `feature_failure_cases.csv`
5. `README.md`

## 7. Prepare a small human evaluation pack for assistant claims

```bash
python scripts/build_assistant_human_eval_pack.py \
  --bundle_root /path/to/icml_all_results_20260401 \
  --output_root results/reviewer_hardening/assistant_human_eval_pack \
  --dossier_method sparse_sae_feature_bank \
  --baseline_method finetuned_encoder_multilabel
```

This produces a blinded annotation pack with randomized A B presentations.
## 8. Run site sensitivity checks

```bash
python scripts/run_proxy_site_sensitivity_suite.py \
  --config configs/policy_mech_interp.yaml \
  --manifest_root data/processed/public_values \
  --family individual_rights \
  --proxy_slug risk_factors_privacy \
  --paired_proxy_slug harms_violation_of_civil_or_human_rights_including_privacy \
  --layer 24 \
  --feature_ids 245,11529,261 \
  --sites resid_post,resid_pre,mlp_out \
  --k 3 \
  --prompt_config configs/policy_text_prompt_sensitivity.yaml \
  --label_normalization mean_token \
  --output_dir results/reviewer_hardening/privacy_site_sensitivity
```

This writes per site logs plus `site_sensitivity_summary.csv`.

## 9. Build exploratory appendix evidence gradients

Run the broader 2B exploratory matrix over the four primary proxies:

```bash
python scripts/run_proxy_causal_sensitivity_matrix.py \
  --matrix_config configs/proxy_causal_exploratory_matrix_2b.yaml \
  --output_root results/reviewer_hardening/exploratory_matrix_2b
```

Run the mirrored 9B exploratory matrix:

```bash
python scripts/run_proxy_causal_sensitivity_matrix.py \
  --matrix_config configs/proxy_causal_exploratory_matrix_9b.yaml \
  --output_root results/reviewer_hardening/exploratory_matrix_9b
```

Summarize strict versus exploratory evidence tiers for appendix reporting:

```bash
python scripts/summarize_exploratory_proxy_evidence.py \
  --bundle_root /path/to/icml_all_results_20260401 \
  --sensitivity_root results/reviewer_hardening \
  --output_root results/reviewer_hardening/exploratory_evidence_summary
```

Use these outputs only for appendix or discussion-level evidence gradients.
Keep the main-paper claims anchored to the original strict positive-causality criterion.
