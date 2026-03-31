# Policy Proxy Sparse Feature Localization

This repository contains the code, configs, tutorials, and benchmark wrappers for a mechanistic NLP project on policy text.

The central question is whether sparse autoencoder features aligned to a pretrained language model can localize policy-relevant proxies, remain stable under stress, show selective causal support in some cases, and then be rendered into analyst-facing policy support objects.

## What this project claims

1. Sparse SAE features can be used as **proxy-specific evidence objects** rather than only as opaque model internals.
2. Mechanistic evidence should be evaluated in stages: localization, stability, selective causal support, pair-linked transfer, and assistant rendering.
3. Sparse features do not need to win the aggregate benchmark to still be useful scientifically.

## What this project does not claim

1. It does not claim that sparse SAE is the best overall predictor across all baselines.
2. It does not claim a universal family-level shared mechanism.
3. It does not present the assistant as an end-to-end policy judge.

## Project scope

The benchmark uses six proxy tasks organized into three pairs.

1. Bias and Discrimination
2. Privacy and Rights violation
3. Transparency and Interpretability

The first two pairs are the main headline pairs. Transparency and Interpretability is kept as appendix-level evidence because the available sample size is much smaller.

## Start here

If you are new to the project, start with the tutorial notebook.

`docs/tutorials/policy_representation_tutorial.ipynb`

The notebook explains the full pipeline at a high level and includes the main formulas used in the benchmark and assistant summaries.

It walks through:

1. AGORA segments and grouped splits
2. Proxy manifests and matched negatives
3. Dense representations and SAE features
4. Proxy-specific feature discovery
5. Stability and masking
6. Pair-linked transfer
7. Targeted proxy feature-set ablation
8. Benchmark aggregation
9. Assistant highlighting, retrieval, and triage
10. Locked Lambda rerun commands and saved outputs

## Repository layout

```text
configs/      Benchmark, assistant, and model configs
caches/       Optional local model caches
data/         Processed manifests, matched negatives, and summaries
docs/         Tutorials, figures, and runbooks
results/      Benchmark outputs, assistant outputs, and analysis artifacts
scripts/      Runnable entry points for manifests, benchmark runs, and controls
src/          Library code for data IO, matching, model hooks, SAE handling, and evaluation
tests/        Unit tests for benchmark and assistant code paths
```

## High-level pipeline

The current pipeline has five stages.

1. **Proxy manifests and matched negatives**
   Build positive and negative segment sets for each proxy while preserving document-grouped splits.
2. **Proxy-specific feature discovery**
   Select a layer, fit proxy scoring functions, and build a sparse feature dossier for each proxy.
3. **Stability and transfer diagnostics**
   Measure bootstrap stability, masking retention, feature overlap across reseeds, and pair-linked transfer.
4. **Targeted causal intervention**
   Ablate top sparse feature sets and compare the resulting margin drop against matched random controls.
5. **Assistant rendering**
   Surface feature-backed cards for highlighting, retrieval, and triage.

## Core benchmark quantities

The tutorial notebook gives the full derivations. The main benchmark uses the following high-level quantities.

1. **Coverage**
   Mean held-out AUROC across proxy tasks.
2. **Robustness**
   Mean clipped masking retention across proxy tasks.
3. **Consistency**
   Mean directed transfer gap relative to cross-family controls.
4. **CoreScore**
   Equal-weight average of Coverage, Robustness, and Consistency.
5. **CausalSelectivity**
   Mean per-example improvement of target feature-set ablation over matched random controls.

Sparse-only causal evidence is reported separately from the mixed all-method comparison table.

## Local setup

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data expectations

The repository is designed to work from processed manifests under `data/processed/public_values/`.

If raw AGORA data is available locally, some helper scripts can rebuild manifests and negatives. The main benchmark and assistant runs, however, are expected to start from the processed manifests.

## Locked ICML run package

The paper numbers should come from the locked Lambda package, not from ad hoc local rehearsal runs.

### Locked 2B benchmark and assistant

1. Benchmark config
   `configs/policy_feature_benchmark_locked_2b.yaml`
2. Assistant config
   `configs/policy_analysis_assistant_locked_2b.yaml`
3. Dense intervention control script
   `scripts/run_dense_subspace_control_eval.py`

### Locked 9B primary replication

1. Benchmark config
   `configs/policy_feature_benchmark_locked_9b_primary.yaml`

## Lambda quickstart

Gemma access is gated on Hugging Face. Make sure the account used by your token already has access to the Gemma model pages.

### One-line login

```bash
export HF_TOKEN='PASTE_YOUR_TOKEN_HERE' && python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN']); print('HF login complete')"
```

### One-line Gemma access check

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-2-2b'); print('Gemma access OK')"
```

### Locked 2B full package

```bash
cd ~/agora-mi && source .venv/bin/activate && python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark_locked_2b.yaml --output_root results/policy_feature_benchmark_locked_2b && python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_2b.yaml --output_root results/policy_feature_benchmark_locked_2b && python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_2b.yaml --output_root results/policy_feature_benchmark_locked_2b && python scripts/run_dense_subspace_control_eval.py --config configs/policy_mech_interp.yaml --manifest_root data/processed/public_values --family equality_neutrality --proxy_slug risk_factors_bias --paired_proxy_slug harms_discrimination --layer 20 --site resid_post --split test --max_samples 100 --random_sets 100 --k 3 && python scripts/run_dense_subspace_control_eval.py --config configs/policy_mech_interp.yaml --manifest_root data/processed/public_values --family individual_rights --proxy_slug risk_factors_privacy --paired_proxy_slug harms_violation_of_civil_or_human_rights_including_privacy --layer 24 --site resid_post --split test --max_samples 100 --random_sets 100 --k 3 && python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_locked_2b.yaml --output_root results/policy_analysis_assistant_locked_2b && printf '%s
' 'The agency shall publish an annual transparency report describing system deployment, audit procedures, and material incidents. Any system processing personal data must include retention limits, access controls, and independent review. Operators should assess whether deployment could create discriminatory effects for protected groups and document mitigation steps before rollout.' > sample_policy_doc.txt && python scripts/run_policy_document_analysis.py --input_path sample_policy_doc.txt --config configs/policy_analysis_assistant_locked_2b.yaml --output_path results/policy_document_analysis_locked_2b.json --document_id lambda_doc_1 --title 'Lambda Document' --source_type lambda_text
```

### Locked 9B primary replication

```bash
cd ~/agora-mi && source .venv/bin/activate && python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark_locked_9b_primary.yaml --output_root results/policy_feature_benchmark_locked_9b_primary && python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_9b_primary.yaml --output_root results/policy_feature_benchmark_locked_9b_primary && python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark_locked_9b_primary.yaml --output_root results/policy_feature_benchmark_locked_9b_primary
```

### Bundle the locked outputs

```bash
cd ~/agora-mi && tar -czf icml_locked_lambda_bundle.tar.gz results/policy_feature_benchmark_locked_2b results/policy_analysis_assistant_locked_2b results/policy_feature_benchmark_locked_9b_primary results/policy_document_analysis_locked_2b.json
```

## Main saved artifacts

After a locked benchmark run, the most important summary files are:

1. `results/policy_feature_benchmark_locked_2b/summary/table_main_benchmark_results.csv`
2. `results/policy_feature_benchmark_locked_2b/summary/table_proxy_mechanistic_evidence.csv`
3. `results/policy_feature_benchmark_locked_2b/summary/table_pair_transfer_and_causality.csv`
4. `results/policy_feature_benchmark_locked_2b/summary/proxy_feature_summary.csv`
5. `results/policy_feature_benchmark_locked_2b/summary/proxy_causal_summary.csv`
6. `results/policy_feature_benchmark_locked_2b/summary/proxy_causal_samples.csv`
7. `results/policy_feature_benchmark_locked_2b/summary/proxy_causal_random_controls.csv`
8. `results/policy_feature_benchmark_locked_2b/summary/proxy_off_target_effects.csv`
9. `results/policy_feature_benchmark_locked_2b/summary/feature_concept_cards.jsonl`

After the locked assistant run, the most important summary files are:

1. `results/policy_analysis_assistant_locked_2b/summary/assistant_leaderboard.json`
2. `results/policy_analysis_assistant_locked_2b/summary/assistant_feature_cards.jsonl`
3. `results/policy_analysis_assistant_locked_2b/summary/assistant_feature_usage.csv`
4. `results/policy_analysis_assistant_locked_2b/summary/assistant_card_dossier_links.jsonl`
5. `results/policy_document_analysis_locked_2b.json`

## Reading results in the right order

A good reading order is:

1. `table_main_benchmark_results.csv`
2. `table_proxy_mechanistic_evidence.csv`
3. `proxy_causal_summary.csv`
4. `table_pair_transfer_and_causality.csv`
5. `assistant_leaderboard.json`
6. `assistant_feature_cards.jsonl`
7. `feature_concept_cards.jsonl`

This order mirrors the intended interpretation ladder.

1. localization
2. stability
3. selective causal support
4. pair-linked support
5. assistant rendering

## Testing

Run the benchmark and assistant unit tests with:

```bash
pytest tests/test_policy_feature_benchmark.py -q
pytest tests/test_policy_analysis_assistant.py -q
```

## Notes on security and reproducibility

1. Gemma downloads require authenticated Hugging Face access.
2. If a token was pasted into a shell or chat session, rotate it after the run is finished.
3. The locked Lambda package should be treated as the paper source of truth.
4. Local runs remain useful for rehearsal, debugging, and figure prototyping, but not for final paper numbers.
