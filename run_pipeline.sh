#!/bin/bash
set -e

# Usage: bash run_pipeline.sh configs/run_gemma2b_it.yaml

CONFIG=$1

if [ -z "$CONFIG" ]; then
  echo "Usage: bash run_pipeline.sh <path_to_config_yaml>"
  echo "Example: bash run_pipeline.sh configs/run_gemma2b_it.yaml"
  exit 1
fi

echo "==========================================="
echo "Starting Full AI Forge Pipeline with: $CONFIG"
echo "==========================================="

# Ensure Python can find modules inside 'src/'
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)

# Parse layers list from config (e.g., layers: [12, 16, 20, 24])
LAYERS_RAW=$(grep "^layers:" "$CONFIG" | sed 's/layers:[[:space:]]*\[//;s/\]//g' | tr ',' ' ')
LAYERS=($LAYERS_RAW)
LAST_LAYER=${LAYERS[${#LAYERS[@]}-1]}

echo "Detected Layers: ${LAYERS[@]}"
echo "Target Layer for Deep Extraction (Steps 3,4,6): $LAST_LAYER"

echo ""
echo "[1/6] Extracting Features & Computing P-Values (Polarization)..."
for layer in "${LAYERS[@]}"; do
  echo "Running Layer $layer..."
  python scripts/compute_polarization.py --config $CONFIG --layer $layer
done

echo ""
echo "[2/6] Applying Global FDR (Benjamini-Hochberg)..."
# Pool hypotheses to isolate core safety dimensions
python scripts/verify_global_fdr.py --layers ${LAYERS[@]}

echo ""
echo "[3/6] Testing Causal Dose-Response (Interference Clamping)..."
# Clamp active nodes to perturb behavior
python scripts/interference_clamp.py --config $CONFIG --layer $LAST_LAYER

echo ""
echo "[4/6] Extracting Causal Subgraphs (Circuit Tracer)..."
# Find topology origins (limit to top 1024 features to prevent OOM)
python scripts/graph_extract_circuit_tracer.py --config $CONFIG --layer $LAST_LAYER --max_feature_nodes 1024

# Rename graph outputs to strip _off0 suffix for Step 5 & 6
mv results/graphs/circuit_safe_layer${LAST_LAYER}_off0.json results/graphs/circuit_safe_layer${LAST_LAYER}.json || true
mv results/graphs/circuit_innov_layer${LAST_LAYER}_off0.json results/graphs/circuit_innov_layer${LAST_LAYER}.json || true

echo ""
echo "[5/6] Calculating Topological Metrics (Depth & Density)..."
# Prove "Brake vs Accelerator" structural differences
python scripts/graph_metrics.py --config $CONFIG

echo ""
echo "[6/6] Evaluating Circuit Faithfulness (CPR & CMD Patching)..."
# Systematically patch circuit nodes to measure necessity and sufficiency
python scripts/circuit_faithfulness.py --config $CONFIG --layer $LAST_LAYER

echo ""
echo "==========================================="
echo "Pipeline Completed Successfully!"
echo "Check the 'results/' directory for CSV logs and JSON graphs."
echo "==========================================="
