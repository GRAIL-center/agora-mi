# Lambda Runbook

This document provides copy and paste one line commands for running the expensive policy representation experiments on Lambda.

## Assumptions

1. The repository has already been cloned on the Lambda machine.
2. AGORA raw files are available locally under `agora/agora/` or `data/raw/agora/`.
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

## Step 4. Build proxy manifests

```bash
python scripts/build_public_value_corpus.py
```

## Step 5. Build matched negatives

```bash
python scripts/build_matched_negatives.py
```

## Step 6. Run benchmark preflight

```bash
python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 7. Run cheap baselines

```bash
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_cheap.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 8. Run the dense residual baseline

```bash
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_dense.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 9. Run the sparse SAE benchmark and causal qualification

```bash
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_sae.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 10. Aggregate all staged outputs into final tables

```bash
python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark.yaml --output_root results/policy_feature_benchmark_lambda
```

## Step 11. Inspect the main outputs

```bash
python -c "from pathlib import Path; root = Path('results/policy_feature_benchmark_lambda/summary'); print((root / 'core_leaderboard.csv').read_text(encoding='utf-8')); print('\n---\n'); print((root / 'benchmark_report.json').read_text(encoding='utf-8')[:4000])"
```

## Useful output paths

1. `results/policy_feature_benchmark_lambda/summary/preflight_report.json`
2. `results/policy_feature_benchmark_lambda/summary/core_leaderboard.csv`
3. `results/policy_feature_benchmark_lambda/summary/main_table.csv`
4. `results/policy_feature_benchmark_lambda/summary/mechanistic_qualification.csv`
5. `results/policy_feature_benchmark_lambda/summary/appendix_task_inventory.csv`
6. `results/policy_feature_benchmark_lambda/summary/benchmark_report.json`

## Optional sanity checks

Check AGORA detection:

```bash
python -c "from pathlib import Path; paths = [Path('agora/agora'), Path('data/raw/agora')]; print({str(p): p.exists() for p in paths})"
```

Check GPU visibility:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('devices', torch.cuda.device_count())"
```
