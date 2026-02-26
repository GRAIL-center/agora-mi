#!/bin/bash
set -euo pipefail

# Usage: bash run_pipeline.sh configs/run_gemma2b_it.yaml
CONFIG="${1:-}"

if [ -z "$CONFIG" ]; then
  echo "Usage: bash run_pipeline.sh <path_to_config_yaml>"
  echo "Example: bash run_pipeline.sh configs/run_gemma2b_it.yaml"
  exit 1
fi

echo "==========================================="
echo "Starting Full AI Forge Pipeline with: $CONFIG"
echo "==========================================="

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)"

LAYERS_RAW=$(grep "^layers:" "$CONFIG" | sed 's/layers:[[:space:]]*\[//;s/\]//g' | tr ',' ' ')
LAYERS=($LAYERS_RAW)
if [ "${#LAYERS[@]}" -eq 0 ]; then
  echo "No layers found in config: $CONFIG"
  exit 1
fi
LAST_LAYER=${LAYERS[${#LAYERS[@]}-1]}

RESULTS_DIR=$(python - "$CONFIG" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(cfg.get("results_dir", "results"))
PY
)

echo "Detected Layers: ${LAYERS[*]}"
echo "Target Layer for Deep Extraction (Steps 3,4,6): $LAST_LAYER"
echo "Results directory: $RESULTS_DIR"

echo ""
echo "[1/6] Extracting Features & Computing P-Values (Polarization)..."
for layer in "${LAYERS[@]}"; do
  echo "Running Layer $layer..."
  python scripts/compute_polarization.py --config "$CONFIG" --layer "$layer"
done

echo ""
echo "[2/6] Applying Global FDR (Benjamini-Hochberg)..."
python scripts/verify_global_fdr.py --results_dir "$RESULTS_DIR/polarization" --layers "${LAYERS[@]}" --tag ""

echo ""
echo "[3/6] Testing Causal Dose-Response (Interference Clamping)..."
python scripts/interference_clamp.py --config "$CONFIG" --layer "$LAST_LAYER"

echo ""
echo "[4/6] Extracting Causal Subgraphs (Circuit Tracer)..."
python scripts/graph_extract_circuit_tracer.py --config "$CONFIG" --layer "$LAST_LAYER" --max_feature_nodes 1024

mv "$RESULTS_DIR/graphs/circuit_safe_layer${LAST_LAYER}_off0.json" "$RESULTS_DIR/graphs/circuit_safe_layer${LAST_LAYER}.json" || true
mv "$RESULTS_DIR/graphs/circuit_innov_layer${LAST_LAYER}_off0.json" "$RESULTS_DIR/graphs/circuit_innov_layer${LAST_LAYER}.json" || true

echo ""
echo "[5/6] Calculating Topological Metrics (Depth & Density)..."
python scripts/graph_metrics.py --config "$CONFIG" --layer "$LAST_LAYER"

echo ""
echo "[6/6] Evaluating Circuit Faithfulness (CPR & CMD Patching)..."
python scripts/circuit_faithfulness.py --config "$CONFIG" --layer "$LAST_LAYER"

echo ""
echo "==========================================="
echo "Pipeline Completed Successfully!"
echo "Check the '$RESULTS_DIR' directory for CSV logs and JSON graphs."
echo "==========================================="
