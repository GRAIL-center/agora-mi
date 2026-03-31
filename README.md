# Mechanistic Policy Feature Analysis with Sparse SAE Features

This repository accompanies a mechanistic NLP project on how language models represent policy relevant text.

The current repository direction is feature centered rather than family centered. The empirical core asks which sparse SAE features localize specific policy proxies in held out policy text, how stable those features remain under resampling and keyword masking, whether targeted feature set ablations matter more than matched random controls, and how the resulting evidence can be rendered into analyst facing policy support outputs.

## Current Research Questions

1. Which sparse SAE features reliably localize each policy proxy in held out policy text
2. Are those proxy specific features stable across bootstrap resampling and robust under keyword masking rather than only reflecting shallow lexical cues
3. Do selected proxy specific feature sets causally support proxy scoring more than matched random feature sets at the same layer and cardinality
4. Can those localized and stress tested features improve analyst facing support tasks such as highlighting, retrieval, and triage

## Empirical Scope

The benchmark keeps three proxy pairs.

1. Bias and Discrimination
2. Privacy and Rights violation
3. Transparency and Interpretability

The first two pairs are the primary headline pairs. Transparency and Interpretability remains a caution pair because the current sample size is too small for equal weight mechanistic claims.

## High Level Workflow

The v2 workflow has four stages.

1. Proxy feature discovery
   Select a held out layer, build sparse feature banks, and rank proxy aligned feature ids.
2. Stability and robustness
   Record bootstrap stability, activation distributions, keyword masking retention, and reseeded feature overlap.
3. Targeted causal intervention
   Ablate top proxy feature sets and compare their score drop against matched random feature sets from the same layer and cardinality.
4. Assistant rendering
   Expose feature backed evidence in segment highlighting, retrieval, and triage outputs.

## Repository Layout

```text
configs/      Experiment and benchmark configuration
docs/         Tutorials and runbooks
scripts/      Data preparation and experiment entry points
src/          Reusable library code for data, models, SAE handling, analysis, and benchmarking
tests/        Unit tests for benchmark and assistant code paths
```

## Tutorial Notebook

Start with the tutorial notebook if you want a guided walkthrough.

`docs/tutorials/policy_representation_tutorial.ipynb`

The notebook now follows the v2 feature dossier workflow.

1. policy proxy setup
2. sparse feature discovery
3. stability and masking
4. targeted proxy ablation
5. assistant feature cards

## Setup

```bash
pip install -r requirements.txt
```

This repository tracks `data/processed/public_values`, so the standard benchmark and assistant runs can start from processed manifests without rebuilding them from raw AGORA inputs.

## Main Commands

Run benchmark preflight.

```bash
python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark.yaml
```

Run the benchmark variants.

```bash
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_cheap.yaml
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_dense.yaml
python scripts/run_policy_feature_benchmark.py --config configs/policy_feature_benchmark_sae.yaml
```

Aggregate benchmark outputs.

```bash
python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark.yaml
```

Run the assistant variants.

```bash
python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_cheap.yaml
python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_dense.yaml
python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant_sae.yaml
```

Analyze one document with the sparse assistant.

```bash
python scripts/run_policy_document_analysis.py --input_path path/to/document.txt --config configs/policy_analysis_assistant_sae.yaml --output_path results/policy_document_analysis.json
```

## Three Line Lambda Run

If local tests are already complete, the full Lambda rerun can be launched in three shell lines.

```bash
cd ~/agora-mi && python scripts/run_policy_feature_benchmark.py --preflight_only --config configs/policy_feature_benchmark.yaml --output_root results/policy_feature_benchmark_lambda_v2 && for cfg in configs/policy_feature_benchmark_cheap.yaml configs/policy_feature_benchmark_dense.yaml configs/policy_feature_benchmark_sae.yaml; do python scripts/run_policy_feature_benchmark.py --config "$cfg" --output_root results/policy_feature_benchmark_lambda_v2 || exit 1; done && python scripts/aggregate_policy_feature_benchmark.py --config configs/policy_feature_benchmark.yaml --output_root results/policy_feature_benchmark_lambda_v2
cd ~/agora-mi && for cfg in configs/policy_analysis_assistant_cheap.yaml configs/policy_analysis_assistant_dense.yaml configs/policy_analysis_assistant_sae.yaml; do python scripts/run_policy_analysis_experiments.py --config "$cfg" --output_root results/policy_analysis_assistant_lambda_v2 || exit 1; done
cd ~/agora-mi && printf '%s\n' 'The agency shall publish an annual transparency report describing system deployment, audit procedures, and material incidents. Any system processing personal data must include retention limits, access controls, and independent review. Operators should assess whether deployment could create discriminatory effects for protected groups and document mitigation steps before rollout.' > sample_policy_doc.txt && python scripts/run_policy_document_analysis.py --input_path sample_policy_doc.txt --config configs/policy_analysis_assistant_sae.yaml --output_path results/policy_document_analysis_lambda_v2.json --document_id lambda_doc_1 --title 'Lambda Document' --source_type lambda_text && tar -czf lambda_results_bundle_v2.tar.gz results/policy_feature_benchmark_lambda_v2 results/policy_analysis_assistant_lambda_v2 results/policy_document_analysis_lambda_v2.json
```

For a fuller explanation of the run order, see `docs/lambda_runbook.md`.

## Main Outputs

The benchmark summary now emphasizes proxy and pair level mechanistic evidence.

1. `results/policy_feature_benchmark/summary/table_main_benchmark_results.csv`
2. `results/policy_feature_benchmark/summary/table_proxy_mechanistic_evidence.csv`
3. `results/policy_feature_benchmark/summary/table_pair_transfer_and_causality.csv`
4. `results/policy_feature_benchmark/summary/feature_dossiers.jsonl`
5. `results/policy_feature_benchmark/summary/proxy_feature_summary.csv`
6. `results/policy_feature_benchmark/summary/proxy_causal_summary.csv`

The assistant summary exposes feature evidence directly.

1. `results/policy_analysis_assistant/summary/assistant_leaderboard.json`
2. `results/policy_analysis_assistant/summary/assistant_report.json`
3. `results/policy_analysis_assistant/summary/assistant_feature_usage.csv`
4. `results/policy_analysis_assistant/summary/assistant_feature_cards.jsonl`
5. `results/policy_document_analysis.json`

## Interpretation Guidance

The benchmark is the scientific core.

1. Coverage asks whether a method separates proxy positives from matched negatives.
2. Stability asks whether the selected sparse features survive resampling and lexical masking.
3. Pair transfer asks whether a proxy aligned feature bank helps on its paired proxy more than on unrelated controls.
4. Causality asks whether ablating selected proxy feature sets matters more than ablating matched random feature sets.

The assistant is the downstream application layer. It uses the benchmark artifacts to surface segments, retrieve related passages, and prioritize review order while exposing feature ids, weights, stability, and causal badges on sparse cards.