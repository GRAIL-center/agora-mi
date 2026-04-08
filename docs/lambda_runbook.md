# Lambda Runbook for Full Swap and 9B Robustness

This document collects copy-paste terminal commands for running the feature-first audit pipeline on a Linux Lambda or cloud GPU machine. The commands assume the `feature-bottom` branch of the `agora-mi` repository.

## 1. Bootstrap

```bash
git clone --branch feature-bottom https://github.com/GRAIL-center/agora-mi.git
cd agora-mi

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.utils.collect_env > torch_env.txt
nvidia-smi > nvidia_smi.txt
```

## 2. Full Swap on Gemma 2 2B

The goal of this run is to close the layer-wise causal concentration gap with the current four-layer 2B setup.

### 2.1 Validate config

```bash
policy-interp validate-config experiments/lambda_full_swap_2b_16k.yaml
```

### 2.2 Prepare corpus tables and matched negatives

```bash
policy-interp prepare-agora experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_prepare.log
policy-interp build-matches experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_matches.log
```

### 2.3 Run the four-layer feature pipeline

```bash
policy-interp extract-activations experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_extract.log
policy-interp build-feature-catalog experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_catalog.log
policy-interp run-autointerp experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_autointerp.log
policy-interp run-baselines experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_baselines.log
policy-interp run-interventions experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_interventions.log
policy-interp export-reports experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_reports.log
policy-interp hash-artifacts experiments/lambda_full_swap_2b_16k.yaml | tee lambda_full_swap_hashes.log
```

### 2.4 Primary outputs

Look under:

`artifacts/lambda_full_swap_2b_16k`

Important files:

1. `interventions/feature_causal_summary.parquet`
2. `interventions/feature_causal_per_segment.parquet`
3. `paper_exports/feature_causal_summary.csv`
4. `paper_exports/autointerp_causal_targets.csv`
5. `paper_exports/autointerp_causal_concentration.png`

## 3. 9B Partial Robustness

The goal of this run is a cheaper robustness pass on Gemma 2 9B for layers 20 and 24 only.

### 3.1 Validate config

```bash
policy-interp validate-config experiments/lambda_9b_robustness_partial.yaml
```

### 3.2 Prepare tables

```bash
policy-interp prepare-agora experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_prepare.log
policy-interp build-matches experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_matches.log
```

### 3.3 Run extraction, cataloging, and AutoInterp

```bash
policy-interp extract-activations experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_extract.log
policy-interp build-feature-catalog experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_catalog.log
policy-interp run-autointerp experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_autointerp.log
policy-interp export-reports experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_reports.log
policy-interp hash-artifacts experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_hashes.log
```

### 3.4 Optional causal add-on

This step is more expensive and can be skipped if the first goal is overlap and feature-faithfulness robustness only.

```bash
policy-interp run-baselines experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_baselines.log
policy-interp run-interventions experiments/lambda_9b_robustness_partial.yaml | tee lambda_9b_interventions.log
```

### 3.5 Primary outputs

Look under:

`artifacts/lambda_9b_robustness_partial`

Important files:

1. `features/feature_catalog.parquet`
2. `features/autointerp/autointerp_feature_scores.parquet`
3. `paper_exports/feature_catalog_table.csv`
4. `paper_exports/feature_catalog_heatmap.png`
5. `interventions/feature_causal_summary.parquet` if the optional causal add-on is run

## 4. Suggested local smoke checks before Lambda

The next local step can stay cheap. Use config validation and small table generation first.

```bash
policy-interp validate-config experiments/lambda_full_swap_2b_16k.yaml
policy-interp validate-config experiments/lambda_9b_robustness_partial.yaml
python -m pytest -q tests
```

If a local GPU smoke run is desired, keep it on the existing pilot configuration rather than the new Lambda runs:

```bash
policy-interp prepare-agora experiments/pilot_200_layer24.yaml
policy-interp build-matches experiments/pilot_200_layer24.yaml
```

## 5. Notes

1. The runbook assumes raw AGORA files are already present at `AGORA Data`.
2. The 9B config uses `bfloat16` and smaller shard sizes to reduce memory pressure.
3. The full swap config keeps the full four-layer 2B setup and adds the AutoInterp feature-set intervention sweep path.
4. The run names are unique, so Lambda outputs will not overwrite the existing local artifact directories.
