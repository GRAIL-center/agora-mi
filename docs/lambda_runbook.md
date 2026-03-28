# Lambda Runbook

This document provides copy and paste one line commands for running the full policy feature benchmark and the policy analysis assistant on Lambda.

## Assumptions

1. The repository has already been cloned on the Lambda machine.
2. The tracked processed manifests under `data/processed/public_values` are present after cloning the repository.
3. You have accepted the Hugging Face access terms for `google/gemma-2-2b` and can authenticate with a token.
4. Commands are run from the repository root.

## Step 1. Update the repository

```bash
git checkout main && git pull --ff-only
```

## Step 2. Create the environment and install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt
```

## Step 3. Authenticate to Hugging Face

```bash
export HF_TOKEN="YOUR_HF_TOKEN" && python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
```

## Step 4. Verify processed manifests

```bash
python -c "from pathlib import Path; p=Path('data/processed/public_values'); print('exists', p.exists()); print('summary', (p/'summary.json').exists())"
```

## Step 5. Run benchmark preflight

```bash
python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 6. Run benchmark cheap baselines

```bash
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_cheap.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 7. Run the benchmark dense residual stage

```bash
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_dense.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 8. Run the benchmark sparse SAE stage and causal qualification

```bash
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_sae.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 9. Aggregate benchmark outputs

```bash
python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 10. Run the assistant cheap stage

```bash
python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_cheap.yaml --output_root results/policy_analysis_assistant_lambda
```

## Step 11. Run the assistant dense stage

```bash
python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_dense.yaml --output_root results/policy_analysis_assistant_lambda
```

## Step 12. Run the assistant sparse stage

```bash
python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_sae.yaml --output_root results/policy_analysis_assistant_lambda
```

## Step 13. Optional single document analysis

```bash
python scripts/run_policy_document_analysis.py --input_path path/to/document.txt --config configs/policy_analysis_assistant_sae.yaml --output_path results/policy_document_analysis_lambda.json --document_id lambda_doc_1 --title "Lambda Document" --source_type lambda_text
```

## Benchmark outputs to inspect

1. `results/policy_feature_benchmark_lambda/summary/preflight_report.json`
2. `results/policy_feature_benchmark_lambda/summary/core_leaderboard.csv`
3. `results/policy_feature_benchmark_lambda/summary/main_table.csv`
4. `results/policy_feature_benchmark_lambda/summary/mechanistic_qualification.csv`
5. `results/policy_feature_benchmark_lambda/summary/appendix_task_inventory.csv`
6. `results/policy_feature_benchmark_lambda/summary/benchmark_report.json`

## Assistant outputs to inspect

1. `results/policy_analysis_assistant_lambda/summary/assistant_leaderboard.json`
2. `results/policy_analysis_assistant_lambda/summary/highlighting_summary.json`
3. `results/policy_analysis_assistant_lambda/summary/retrieval_summary.json`
4. `results/policy_analysis_assistant_lambda/summary/triage_summary.json`
5. `results/policy_analysis_assistant_lambda/summary/trust_bundle.json`
6. `results/policy_analysis_assistant_lambda/summary/assistant_report.json`
7. `results/policy_document_analysis_lambda.json`

## Optional sanity checks

Check processed manifests:

```bash
python -c "from pathlib import Path; p = Path('data/processed/public_values'); print('exists', p.exists()); print('summary', (p/'summary.json').exists())"
```

Check GPU visibility:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('devices', torch.cuda.device_count())"
```
