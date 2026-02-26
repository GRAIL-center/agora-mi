#!/bin/bash
# Gilbreth submission script.
# Submit with:
#   sbatch -A <YOUR_ALLOCATION> -p gpu submit_gautschi.sh [CONFIG_PATH]

#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -J aiforge_pipeline
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/run_gemma2b_it.yaml}"

mkdir -p logs

echo "=========================================================="
echo "Starting AI Forge pipeline on Gilbreth"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Config: ${CONFIG_PATH}"
echo "=========================================================="

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional dependency for circuit extraction stage.
if ! python -c "import circuit_tracer" >/dev/null 2>&1; then
  python -m pip install git+https://github.com/decoderesearch/circuit-tracer.git
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Warning: HF_TOKEN is not set. Model download may fail for gated repos."
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)"
export TF_ENABLE_ONEDNN_OPTS=0

bash run_pipeline.sh "${CONFIG_PATH}"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
