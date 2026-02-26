"""B2: Negative control — label-shuffled FDR.

Randomly shuffles the safe/innov labels and re-runs polarization + FDR.
Under the null hypothesis, 0 features should survive FDR.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, setup_logging
from analysis.fdr import benjamini_hochberg
from analysis.polarization import polarization_table
from data.io import read_jsonl
from model.hooks import capture_residual_stream
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt
from sae.encode import encode_features
from sae.load_sae import load_sae_for_layer


def _extract_features(*, model, tokenizer, device, sae, prompts, layer, batch_size, max_length):
    all_feats = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = capture_residual_stream(model, layer=layer, token_index="last", inputs=enc)
            f = encode_features(sae, h.to(torch.float32))
        all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def main():
    setup_logging("negative_control_fdr")
    cfg = read_config("configs/run.yaml")
    layer = 24

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])

    safe_rows = read_jsonl("data/processed/dsafe_train.jsonl")
    innov_rows = read_jsonl("data/processed/dinnov_train.jsonl")

    # Combine all prompts
    all_rows = safe_rows + innov_rows
    all_prompts = [render_prompt(template, str(r["text"])) for r in all_rows]
    n_safe = len(safe_rows)

    logging.info("Total prompts: %d (safe=%d, innov=%d)", len(all_rows), n_safe, len(innov_rows))

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    sae = load_sae_for_layer(cfg, layer=layer, device=device)
    batch_size = int(cfg.get("batch_size", 4))
    max_length = int(cfg.get("max_length", 1024))

    logging.info("Extracting features for all prompts...")
    all_feats = _extract_features(
        model=model, tokenizer=tokenizer, device=device, sae=sae,
        prompts=all_prompts, layer=layer, batch_size=batch_size, max_length=max_length,
    )

    # Shuffle labels
    np.random.seed(42)
    n_total = len(all_feats)
    perm = np.random.permutation(n_total)
    shuffled_safe = all_feats[perm[:n_safe]]
    shuffled_innov = all_feats[perm[n_safe:]]

    logging.info("Running polarization with shuffled labels...")
    perm_N = int(cfg.get("perm_N", 10000))
    delta_df = polarization_table(shuffled_safe, shuffled_innov, perm_N=perm_N, seed=42)

    fdr_q = float(cfg.get("fdr_q", 0.05))
    if "p_value" in delta_df.columns:
        fdr_result = benjamini_hochberg(delta_df["p_value"].values, q=fdr_q)
        delta_df["q_value"] = fdr_result["q_values"]
        delta_df["fdr_reject"] = fdr_result["reject"]
        n_survivors = int(delta_df["fdr_reject"].sum())
    else:
        n_survivors = -1

    print("\n=== Negative Control (Label-Shuffled FDR) ===")
    print(f"Total features tested: {len(delta_df)}")
    print(f"FDR survivors under null: {n_survivors}")
    if n_survivors == 0:
        print("✅ PASS: Zero features survive FDR under label shuffle, confirming our 42 are not spurious.")
    else:
        print(f"⚠️ WARNING: {n_survivors} features survived under null. Expected 0.")

    out_dir = ensure_dir(Path("results/sensitivity"))
    payload = {"n_survivors_null": n_survivors, "fdr_q": fdr_q, "perm_N": perm_N}
    with open(out_dir / "negative_control_fdr.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {out_dir / 'negative_control_fdr.json'}")


if __name__ == "__main__":
    main()
