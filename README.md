# Safety vs Innovation Circuits (Runbook-Aligned Repo)

This repository follows the runbook layout for:

1. AGORA schema inspection and contrastive corpus construction (`Dsafe` / `Dinnov`)
2. Gemma residual activation capture at answer position
3. Gemma Scope SAE feature encoding
4. Contrastive polarization analysis
5. Pilot ablation with random-feature control
6. Sanity checks and summary reporting

## Install

```bash
pip install -r requirements.txt
```

## Expected Data Location

Place AGORA files in `data/raw/agora/`:

- `documents.csv`
- `segments.csv`
- (optional) `authorities.csv`, `collections.csv`, `fulltext/`

## Command Sequence

1. Inspect schema:

```bash
python scripts/inspect_agora_schema.py --input_dir data/raw/agora --out logs/agora_schema_summary.json
```

2. Build contrastive corpus:

```bash
python scripts/build_contrastive_corpus.py --config configs/label_map.yaml --input_dir data/raw/agora --out_dir data/processed
```

3. Smoke tests:

```bash
python scripts/smoke_test_model.py --config configs/run.yaml
python scripts/smoke_test_sae.py --config configs/run.yaml --layer 16
```

4. Compute polarization:

```bash
python scripts/compute_polarization.py --config configs/run.yaml --layer 16
```

5. Run sanity checks:

```bash
python scripts/sanity_checks.py --config configs/run.yaml --layer 16
```

6. Pilot ablation:

```bash
python scripts/pilot_ablation.py --config configs/run.yaml --layer 16
```

7. Stats summary:

```bash
python scripts/stats_report.py --config configs/run.yaml --layer 16
```

8. Circuit graph extraction (requires `circuit-tracer`):

```bash
python scripts/graph_extract_circuit_tracer.py --config configs/run.yaml --layer 24 --max_prompts 20
```

9. Topology metrics:

```bash
python scripts/graph_metrics.py --config configs/run.yaml --layer 24
```

10. Generate paper figures:

```bash
python scripts/generate_paper_figures.py --config configs/run.yaml --layer 24
```

## Outputs

- `data/processed/dsafe_{train,dev,test}.jsonl`
- `data/processed/dinnov_{train,dev,test}.jsonl`
- `results/polarization/layer{L}_delta.csv`
- `results/polarization/layer{L}_topk_safe.json`
- `results/polarization/layer{L}_topk_innov.json`
- `results/pilot_ablation/effect_summary.json`
- `results/sanity/layer{L}_sanity.json`
- `results/graphs/circuit_safe_layer{L}.json`
- `results/graphs/circuit_innov_layer{L}.json`
- `results/graphs/topology_layer{L}.json`
- `results/figures/paper/fig1_ablation_heatmap.{png,pdf}`
- `results/figures/paper/fig2_dose_response.{png,pdf}`
- `results/figures/paper/fig3_volcano.{png,pdf}`
- `results/figures/paper/fig4_circuit_comparison.{png,pdf}`
- `results/stats/summary.md`

## Future Multi-Model Testing Plan

Planned model expansion for robustness/generalization checks:

- `google/gemma-2-2b-it`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b`
- `google/gemma-3-1b-it`
- `google/gemma-3-4b-it`
- `google/gemma-3-12b-it`

Recommended rollout order:

1. `Gemma 2 2B Instruct` (alignment effect vs base 2B)
2. `Gemma 2 9B` (scale effect in same family)
3. `Llama 3 8B Instruct` (cross-family generalization)
4. `Gemma 3 1B/4B/12B` (next-gen family sweep)

Execution notes:

- Keep the same data split and evaluation scripts to ensure comparability.
- Add a dedicated `configs/run_<model>.yaml` per model.
- For Gemma 3 models, verify SAE availability (`sae_release` / `sae_id_template`) and circuit-tracer compatibility before full runs.
- Current `run_all_models.sh` covers Gemma 2 2B, Gemma 2 2B-IT, Gemma 2 9B, and Llama 3 8B Instruct.

## Notes

- Scripts use `argparse` and log to `logs/<script>_<timestamp>.log`.
- Prompting uses `configs/prompt_templates.yaml` and label-token next-token logprob.
- SAE load defaults to `sae-lens` with `sae_release` + `sae_id_template`; optional `--sae_npz` is supported for testing.
- Polarization now computes per-feature permutation p-values and applies Benjamini-Hochberg FDR correction.
- Circuit graph extraction requires `circuit-tracer` (`pip install .` after cloning).
