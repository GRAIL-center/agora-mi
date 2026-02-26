#!/bin/bash
set -e

echo "=========================================================="
echo "    AI Forge - Multi-Model Generalization Suite (A100)    "
echo "=========================================================="
echo ""
echo "This script will sequentially execute the full feature tracking,"
echo "FDR, clamping, and topology pipeline across 4 distinct models."
echo "If any pipeline fails, the script will halt immediately."
echo ""

echo "▶▶▶ 1️⃣ Executing: Gemma-2-2B (Baseline Replication ─ Control)"
bash run_pipeline.sh configs/run.yaml
echo "✅ Gemma-2-2B Pipeline Completed!"
echo ""

echo "▶▶▶ 2️⃣ Executing: Gemma-2-2B-IT (Alignment/Instruct Effect)"
bash run_pipeline.sh configs/run_gemma2b_it.yaml
echo "✅ Gemma-2-2B-IT Pipeline Completed!"
echo ""

echo "▶▶▶ 3️⃣ Executing: Gemma-2-9B (Parameter Scale Effect)"
bash run_pipeline.sh configs/run_gemma9b.yaml
echo "✅ Gemma-2-9B Pipeline Completed!"
echo ""

echo "▶▶▶ 4️⃣ Executing: Llama-3-8B-Instruct (Cross-Architecture Generalization)"
bash run_pipeline.sh configs/run_llama3.yaml
echo "✅ Llama-3-8B-Instruct Pipeline Completed!"
echo ""

echo "=========================================================="
echo "🏆 ALL 4 MODELS SUCCESSFULLY EVALUATED! 🏆"
echo "Please check the individual 'results/' subdirectories for metrics."
echo "=========================================================="
