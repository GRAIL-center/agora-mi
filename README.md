# Auditing Policy Representations in Language Models via Sparse Features

This repository contains a feature first mechanistic auditing pipeline for policy relevant model behavior in language models. The current codebase focuses on Gemma 2 2B with Gemma Scope sparse autoencoders and the AGORA policy corpus. The project treats sparse features as an audit surface for inspecting how a model processes statutory and governance text, rather than as a competitive classifier.

## Repository layout

`src/policy_interp`
Core pipeline code for data preparation, activation extraction, feature catalog construction, AutoInterp, audit evaluation, causal interventions, and report export.

`experiments`
Versioned experiment configurations for pilot runs, full layer runs, causal runs, and the revised four layer audit setup.

`scripts`
Utility scripts for rebuilding paper figures.

`tests`
Unit tests for the pipeline.

`paper`
ICML 2026 workshop style draft, bibliography, template files, and exported figures.

`references.md`
Working reference inventory used during paper development.

`Main Plan.md`
Research planning document that motivated the current feature first direction.

## Main ideas

The pipeline is organized around five questions.

1. Which sparse features activate on policy relevant text
2. Which token spans trigger those features
3. Which features look policy specific rather than generic
4. How do those features vary across layers
5. Which features or small sparse sets have measurable causal influence on model behavior

The current paper framing is:

`Auditing Policy Representations in Language Models via Sparse Features`

## Installation

Use Python 3.12 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Core CLI commands

Validate an experiment configuration:

```powershell
policy-interp validate-config experiments\revised_mvp_2b_16k.yaml
```

Prepare AGORA tables and split manifests:

```powershell
policy-interp prepare-agora experiments\revised_mvp_2b_16k.yaml
policy-interp build-matches experiments\revised_mvp_2b_16k.yaml
```

Run extraction and audit stages:

```powershell
policy-interp extract-activations experiments\revised_mvp_2b_16k.yaml
policy-interp build-feature-catalog experiments\revised_mvp_2b_16k.yaml
policy-interp run-autointerp experiments\revised_mvp_2b_16k.yaml
policy-interp run-audit-eval experiments\revised_mvp_2b_16k.yaml
policy-interp export-reports experiments\revised_mvp_2b_16k.yaml
```

Generate an audit record for a new text file:

```powershell
policy-interp run-batch-scorer experiments\revised_mvp_2b_16k.yaml --input-path path\to\document.txt --output-name audit_demo
```

## Data and artifacts

The branch includes the `AGORA Data` directory required for preparation and matching stages.

Generated experiment outputs are still excluded from version control.

Expected local output directory:

1. `artifacts` for experiment outputs

The code assumes experiment outputs live under the run root specified by each YAML configuration.

## Current scope

The repository reflects a medium local setup used for the current paper draft.

1. Gemma 2 2B
2. Gemma Scope residual SAEs
3. Layers 12, 16, 20, and 24
4. Feature first audit reports, cross layer maps, AutoInterp labels, and causal diagnostics

Module discovery remains in the codebase as a secondary analysis path, but the main paper and pipeline now prioritize feature level auditing.
