# AI Forge: A100 Multi-Model Execution Plan (Phase 5)

## 📌 Objective
To rigorously validate the "Brake vs. Accelerator" hypothesis across multiple State-of-the-Art (SOTA) Large Language Models. This plan addresses critical reviewer feedback by demonstrating that the topological differences (depth and sparsity) between safety and innovation circuits are not specific to a single base model, but are a general property of aligned LLMs. 

Additionally, we introduce rigorous **Necessity and Sufficiency Patching** to formally prove the causal faithfulness of the extracted circuits.

---

## 🚀 1. Target Models for Generalization
We will sequentially execute the complete analysis pipeline on four strategically selected models on the A100 (40GB/80GB) instance:

1. **`google/gemma-2-2b` (Control / Baseline Replication)**
2. **`google/gemma-2-2b-it` (Alignment Effect Verification)**
3. **`google/gemma-2-9b` (Scale Effect Verification)**
4. **`meta-llama/Meta-Llama-3-8B-Instruct` (Cross-Architecture Generalization)**

---

## 💾 2. Checkpointing & Data Persistence
To handle the long execution times safely, the pipeline is designed to **save state immediately to disk** at the end of every step.
- **Step 1 (Activation/Polarization):** Saves intermediate raw feature activations and $p$-value statistics to `results/features/`.
- **Step 2 (FDR):** Subsets the feature dimensions and immediately writes the final target indices to `results/fdr_features.csv`.
- **Step 3 (Steering/Clamping):** Saves logit Dose-Response curves to `results/steering/`.
- **Step 4 (Graph Extraction):** Extracted node/edge topology is immediately written out as `results/graphs/circuit_{label}_layer{L}.json`.
- **Step 6 (Patching):** Frame-by-frame probing results are written directly to `results/circuit_faithfulness_layer{L}.csv`.

---

## ⚙️ 3. Execution Pipeline & Realistic Time Estimations

> ⚠️ **Resource Note (A100 80GB):** Total end-to-end execution for all 4 models is estimated at **15~24 hours**.

### [Phase A] Feature Discovery & Causal Verification
*Estimated Time: 1~2 hours (2B models), 2~3 hours (8B/9B models).*

- **Step 1. Polarization:** Extract SAE features across deep layers (L12, L16, L20, L24).
- **Step 2. Global FDR:** Apply Benjamini-Hochberg correction ($q < 0.05$) to isolate strictly significant features.
- **Step 3. Dose-Response (Clamping):** Clamp FDR-surviving features up/down to prove causal logit shifting.

### [Phase B] Graph Extraction (The Bottleneck)
*Estimated Time: 1.5~2 hours (2B models), 3~6 hours (8B/9B models).*

- **Step 4. Circuit Extraction:** Utilize `circuit-tracer` (node_threshold=0.8, edge_threshold=0.98, $n=20$ prompts). 
  - *Note:* Backpropagating through 42 layers (for the 9B model) over 20 prompts is computationally extreme. Disk-offloading and tiny batch sizes are enforced.
- **Step 5. Topology Metrics:** Calculate Depth/Density on the JSON graphs.

### [Phase C] Circuit Faithfulness Evaluation
*Estimated Time: 0.5~1 hour per model.*

**Step 6. CPR & CMD Patching**
To prove the Autoencoder-extracted nodes hold genuine causal fidelity, we evaluate:

1. **Sufficiency (Circuit Performance Ratio - CPR):**
   - **Protocol:** Run the *Innovative* prompt (Corrupted baseline). Use `TransformerLens` Hooks to selectively inject/cache the exact activation values of the $G_{safe}$ circuit nodes obtained from a *Safe* prompt run.
   - **Risk Control (Off-Manifold):** Since we are injecting exact cached values into a different context stream, there is a risk of off-manifold hallucination. We mitigate this by only intervening strictly on the FDR-verified nodes at the final output layers.
   - **Measurement:** We measure the recovery of the relative Logit Probability of the refusal/restriction tokens vs. capability tokens.

2. **Necessity (Circuit Masking Difference - CMD):**
   - **Protocol:** Run the *Safe* prompt (Clean baseline). Zero-ablate (mask to 0) strictly the nodes within $G_{safe}$.
   - **Measurement:** We track the exact performance drop in the probability space: $P(\text{Refusal} \mid \text{Clean}) - P(\text{Refusal} \mid \text{Ablated})$. A total collapse in $P(\text{Refusal})$ proves the circuit was necessary for the safety behavior.
