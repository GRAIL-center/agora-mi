"""B4: Compare SAE features to the refusal direction (Arditi et al. 2024).

Computes the mean difference-of-means refusal direction in residual stream space,
then projects each of the 42 FDR-surviving SAE decoder vectors onto this direction
to measure alignment.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, setup_logging
from data.io import read_jsonl
from model.hooks import capture_residual_stream
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt
from sae.load_sae import load_sae_for_layer


def _get_residuals(model, tokenizer, device, prompts, layer, batch_size, max_length):
    """Extract raw residual stream vectors (before SAE)."""
    all_h = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = capture_residual_stream(model, layer=layer, token_index="last", inputs=enc)
        all_h.append(h.cpu().to(torch.float32).numpy())
    return np.concatenate(all_h, axis=0)


def main():
    setup_logging("compare_refusal_direction")
    cfg = read_config("configs/run.yaml")
    layer = 24

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])

    safe_rows = read_jsonl("data/processed/dsafe_train.jsonl")
    innov_rows = read_jsonl("data/processed/dinnov_train.jsonl")
    safe_prompts = [render_prompt(template, str(r["text"])) for r in safe_rows]
    innov_prompts = [render_prompt(template, str(r["text"])) for r in innov_rows]

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    sae = load_sae_for_layer(cfg, layer=layer, device=device)
    batch_size = int(cfg.get("batch_size", 4))
    max_length = int(cfg.get("max_length", 1024))

    logging.info("Extracting residuals for safe prompts...")
    h_safe = _get_residuals(model, tokenizer, device, safe_prompts, layer, batch_size, max_length)
    logging.info("Extracting residuals for innov prompts...")
    h_innov = _get_residuals(model, tokenizer, device, innov_prompts, layer, batch_size, max_length)

    # Compute refusal direction (Arditi et al.): difference of means
    refusal_dir = h_safe.mean(axis=0) - h_innov.mean(axis=0)
    refusal_dir = refusal_dir / (np.linalg.norm(refusal_dir) + 1e-12)
    logging.info("Refusal direction computed (norm=1, dim=%d)", refusal_dir.shape[0])

    # Get SAE decoder weights for the 42 FDR features
    topk_path = Path("results/polarization/layer24_train_topk_safe.json")
    with open(topk_path, "r") as f:
        fdr_ids = json.load(f)["feature_ids"][:42]

    # Extract decoder vectors (W_dec rows)
    inner_sae = sae.sae if hasattr(sae, "sae") else sae
    if hasattr(inner_sae, "W_dec"):
        W_dec = inner_sae.W_dec.data.cpu().to(torch.float32).numpy()
    elif hasattr(inner_sae, "decoder"):
        W_dec = inner_sae.decoder.weight.data.cpu().to(torch.float32).numpy()
    else:
        logging.error("Cannot find decoder weights in SAE object (dir: %s)", dir(inner_sae))
        return

    # Some SAEs have shape (d_sae, d_model) and some (d_model, d_sae)
    if W_dec.shape[0] == refusal_dir.shape[0]:
        # shape is (d_model, d_sae), need to transpose
        W_dec = W_dec.T  # Now (d_sae, d_model)

    results = []
    for fid in fdr_ids:
        dec_vec = W_dec[fid]
        dec_vec_norm = dec_vec / (np.linalg.norm(dec_vec) + 1e-12)
        cos_sim = float(np.dot(dec_vec_norm, refusal_dir))
        results.append({"feature_id": int(fid), "cos_sim_refusal_dir": cos_sim})

    # Sort by absolute alignment
    results.sort(key=lambda x: abs(x["cos_sim_refusal_dir"]), reverse=True)

    cos_vals = [r["cos_sim_refusal_dir"] for r in results]
    mean_abs_cos = float(np.mean(np.abs(cos_vals)))
    n_aligned = sum(1 for c in cos_vals if abs(c) > 0.3)
    n_orthogonal = sum(1 for c in cos_vals if abs(c) < 0.1)

    print("\n=== Refusal Direction Comparison (Arditi et al.) ===")
    print(f"Number of FDR features: {len(results)}")
    print(f"Mean |cos(feature, refusal_dir)|: {mean_abs_cos:.4f}")
    print(f"Strongly aligned (|cos|>0.3): {n_aligned}/{len(results)}")
    print(f"Orthogonal (|cos|<0.1): {n_orthogonal}/{len(results)}")
    print(f"\nTop 10 most aligned features:")
    for r in results[:10]:
        print(f"  Feature {r['feature_id']}: cos_sim = {r['cos_sim_refusal_dir']:.4f}")

    if n_orthogonal > len(results) // 2:
        print("\n✅ Majority of features are ORTHOGONAL to the single refusal direction.")
        print("This indicates our circuit-level analysis captures richer structure than a single direction.")
    else:
        print(f"\nℹ️ {n_aligned} features align with the refusal direction; {n_orthogonal} are orthogonal.")
        print("Our features partially overlap with Arditi's refusal direction but also capture additional structure.")

    out_dir = ensure_dir(Path("results/sensitivity"))
    payload = {
        "mean_abs_cos": mean_abs_cos,
        "n_aligned_gt03": n_aligned,
        "n_orthogonal_lt01": n_orthogonal,
        "features": results,
    }
    with open(out_dir / "refusal_direction_comparison.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_dir / 'refusal_direction_comparison.json'}")


if __name__ == "__main__":
    main()
