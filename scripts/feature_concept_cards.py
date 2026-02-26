"""B3: Feature concept cards — top-activating contexts for each FDR survivor.

For each of the 42 FDR-surviving features at Layer 24, finds the top-5
most-activating provision texts from the training corpus.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from _common import ensure_dir, read_config, setup_logging


def main():
    setup_logging("feature_concept_cards")
    cfg = read_config("configs/run.yaml")
    layer = 24

    # Load pre-computed features
    feats_path = Path("results/polarization/layer24_train_features.npz")
    topk_path = Path("results/polarization/layer24_train_topk_safe.json")

    if not feats_path.exists() or not topk_path.exists():
        logging.error("Missing required files. Run compute_polarization.py first.")
        return

    arr = np.load(feats_path, allow_pickle=True)
    safe_feats = arr["safe_features"]   # (n_safe, d_sae)
    innov_feats = arr["innov_features"] # (n_innov, d_sae)
    safe_texts = arr["safe_text"]       # (n_safe,) string array
    innov_texts = arr["innov_text"]     # (n_innov,) string array

    with open(topk_path, "r") as f:
        safe_ids = json.load(f)["feature_ids"]

    # Take the 42 FDR survivors
    fdr_ids = safe_ids[:42]

    all_feats = np.concatenate([safe_feats, innov_feats], axis=0)
    all_texts = np.concatenate([safe_texts, innov_texts], axis=0)
    all_labels = (["SAFE"] * len(safe_texts)) + (["INNOV"] * len(innov_texts))

    cards = []
    for fid in fdr_ids:
        activations = all_feats[:, fid]
        top5_idx = np.argsort(activations)[::-1][:5]
        
        examples = []
        for idx in top5_idx:
            examples.append({
                "text": str(all_texts[idx])[:200],
                "activation": float(activations[idx]),
                "label": all_labels[idx],
            })
        
        cards.append({
            "feature_id": int(fid),
            "mean_safe": float(safe_feats[:, fid].mean()),
            "mean_innov": float(innov_feats[:, fid].mean()),
            "delta": float(safe_feats[:, fid].mean() - innov_feats[:, fid].mean()),
            "top5_contexts": examples,
        })

    print(f"\n=== Feature Concept Cards ({len(cards)} features) ===\n")
    for card in cards[:5]:  # Print first 5 as preview
        print(f"Feature {card['feature_id']}  (Δ={card['delta']:.4f})")
        for ex in card["top5_contexts"][:2]:
            print(f"  [{ex['label']}] act={ex['activation']:.4f}: {ex['text'][:100]}...")
        print()

    out_dir = ensure_dir(Path("results/sensitivity"))
    out_path = out_dir / "feature_concept_cards.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(cards)} concept cards to {out_path}")


if __name__ == "__main__":
    main()
