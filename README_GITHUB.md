# GitHub Upload Notes

This folder is a cleaned research package prepared from the original workspace.

## Included
- Core code: `src/`, `scripts/`, `configs/`
- Data used in experiments: `data/raw/agora/`, `data/processed/`
- Research outputs: `results/`
- Small test artifact: `artifacts/`
- Paper/source materials and run scripts at repo root

## Excluded as non-essential/garbage
- Caches/build metadata: `__pycache__/`, `*.pyc`, `*.egg-info`, `build/`
- Local editor/runtime noise: `.vscode/`, `.env`, `logs/`, temp dirs
- External vendored repos not required to version here: `paperbanana/`, `circuit-tracer/`
- Duplicate/raw archive leftovers such as `agora/` and `agora.zip`

## Dependency note
- `circuit-tracer` is required only for graph extraction scripts.
- Install externally when needed:
  - `pip install git+https://github.com/safety-research/circuit-tracer.git`

## Quick push
```bash
git init
git add .
git commit -m "Initial cleaned research repo"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
