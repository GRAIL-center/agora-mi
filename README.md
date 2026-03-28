# How Do Language Models Represent Policy Text?

This repository accompanies a research paper project on the internal representation of policy text in language models.

The project studies proxy grounded mechanistic analysis of policy features using AGORA policy segments, dense residual representations, sparse SAE features, matched negatives, held out transfer, keyword masking, and causal intervention.

The same mechanistic pipeline also supports an applied policy analysis assistant layer for segment highlighting, cross document retrieval, and review triage. In this repository, that assistant layer is treated as a downstream use of the mechanistic evidence rather than as a replacement for it.

## Paper Focus

The central research questions are:

1. How do language models internally represent policy text and policy relevant features
2. Which policy features produce distinct and robust internal representations
3. Do related policy features share internal structure that generalizes across proxies
4. Are these discovered policy features causally involved in held out policy judgments

Public values are treated as a secondary interpretive lens rather than as the primary empirical target.

## Core Empirical Scope

The current v1 empirical core focuses on three related concept pairs:

1. Bias and Discrimination
2. Privacy and Rights violation
3. Transparency and Interpretability

These pairs support a compact evaluation protocol for policy feature discovery, transfer, robustness, and causal qualification.

## Policy Analysis Assistant Extension

The assistant layer is built on top of the mechanistic core workflow.

1. Highlight policy relevant segments using proxy and family aligned internal signals
2. Retrieve segments from the corpus that share similar policy logic
3. Triage which document segments deserve closer human review first

The assistant does not replace the scientific core. Discovery, transfer, masking, and intervention remain the main evidence chain. Assistant facing scores and grounded natural language notes are presentation layers over that evidence.

## Repository Layout

```text
configs/      Experiment and benchmark configuration
docs/         Tutorials, figures, and lightweight project notes
scripts/      Data preparation, discovery, transfer, and benchmark utilities
src/          Reusable library code for data, models, SAE handling, analysis, and benchmarking
tests/        Unit and smoke tests for the current pipeline
```

## Tutorial Notebook

For a guided walkthrough from data processing to mechanistic interpretability, start with:

`docs/tutorials/policy_representation_tutorial.ipynb`

The notebook is written for readers who are new to mechanistic interpretability and explains both the scientific logic and the repository workflow step by step. It now ends with a transition from mechanistic evidence to policy analysis support objects such as `segment_card`, `document_brief`, retrieval examples, and grounded natural language notes.

For the full Lambda run sequence for benchmark and assistant experiments, see `docs/lambda_runbook.md`.

## Method Overview

The current workflow is:

1. Filter AGORA to AI related, operative, annotated segments
2. Build proxy specific manifests from existing AGORA tag annotations
3. Construct metadata matched negatives
4. Extract dense residual and sparse SAE representations from Gemma 2 2B PT
5. Perform train only feature discovery
6. Evaluate held out transfer across related policy features
7. Stress test with keyword masking
8. Consolidate repeated sparse features into family core banks
9. Run intervention based causal evaluation on held out policy judgments

## Evaluation Protocol

The `Policy Feature Governance Audit Benchmark` is the standardized evaluation wrapper around the methodology above.

Its main axes are:

1. Coverage
2. Consistency
3. Robustness
4. Causality

Only the first three axes define the main method comparison table. Causality is reported separately as a mechanistic qualification for intervention capable methods.

## Setup

```bash
pip install -r requirements.txt
```

The main code expects local access to AGORA files and model artifacts. These large local resources are intentionally not tracked in the repository.

## Useful Commands

Build proxy manifests:

```bash
python scripts/build_public_value_corpus.py
```

Build matched negatives:

```bash
python scripts/build_matched_negatives.py
```

Run the benchmark preflight:

```bash
python scripts/run_policy_feature_benchmark.py --preflight_only
```

Run the full benchmark:

```bash
python scripts/run_policy_feature_benchmark.py
```

Aggregate benchmark outputs:

```bash
python scripts/aggregate_policy_feature_benchmark.py
```

Run the policy analysis assistant experiment suite:

```bash
python scripts/run_policy_analysis_experiments.py --config configs/policy_analysis_assistant.yaml
```

Analyze a single document with the sparse assistant:

```bash
python scripts/run_policy_document_analysis.py --input_path path/to/document.txt --output_path results/policy_document_analysis.json
```

Main assistant outputs:

1. `results/policy_analysis_assistant/summary/assistant_leaderboard.json`
2. `results/policy_analysis_assistant/summary/assistant_report.json`
3. `results/policy_analysis_assistant/summary/trust_bundle.json`
4. `results/policy_document_analysis.json`

## Figures and Tutorials

Versioned figures and tutorial assets live under `docs/figures/` and `docs/tutorials/`.

## Notes on Local Dependencies

This repository may coexist locally with external tool clones such as PaperBanana or circuit-tracer. Those external repositories are treated as optional local dependencies and are not part of the main tracked project state here.
