# AI Forge

This repository studies how language models internally represent and process policy text.

The current project direction is a proxy grounded mechanistic analysis of policy features using AGORA policy segments, dense residual representations, sparse SAE features, matched negatives, held out transfer, keyword masking, and causal intervention.

## Research Focus

The current project asks four questions:

1. How do language models internally represent policy text and policy relevant features
2. Which policy features produce distinct and robust internal representations
3. Do related policy features share internal structure that generalizes across proxies
4. Are these discovered policy features causally involved in held out policy judgments

Public values are treated as a downstream interpretive lens rather than a primary empirical target.

## Current Core Policy Features

The v1 empirical core uses three concept pairs:

1. Bias and Discrimination
2. Privacy and Rights violation
3. Transparency and Interpretability

These pairs are evaluated through a compact audit suite, but the benchmark is a supporting evaluation protocol rather than the main identity of the project.

## Repository Layout

```text
configs/      Experiment and benchmark configuration
docs/         Tutorials, figures, and lightweight project notes
scripts/      Data preparation, discovery, transfer, and benchmark utilities
src/          Reusable library code for data, models, SAE handling, analysis, and benchmarking
tests/        Unit and smoke tests for the current pipeline
```

## Tutorial Notebook

If you want a guided walkthrough from data processing to mechanistic interpretability, start with:

`docs/tutorials/policy_representation_tutorial.ipynb`

The notebook is written for readers who are new to mechanistic interpretability and explains both the scientific logic and the repository workflow step by step.

## Main Pipeline

The current main pipeline is:

1. Filter AGORA to AI related, operative, annotated segments
2. Build proxy specific manifests from existing AGORA tag annotations
3. Construct metadata matched negatives
4. Extract dense residual and sparse SAE representations from Gemma 2 2B PT
5. Perform train only feature discovery
6. Evaluate held out transfer across related policy features
7. Stress test with keyword masking
8. Consolidate repeated sparse features into family core banks
9. Run intervention based causal evaluation on held out policy judgments

## Benchmark Scope

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

## Figures and Tutorials

Versioned figures and tutorial assets live under `docs/figures/` and `docs/tutorials/`.

## Notes on Local Dependencies

This repository may coexist locally with external tool clones such as PaperBanana or circuit-tracer. Those external repositories are treated as optional local dependencies and are not part of the main tracked project state here.
