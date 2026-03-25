"""Post-process existing polarization features to add FDR correction.

Reads the pre-computed features.npz files and re-computes the delta table
with per-feature permutation p-values and BH FDR correction.
Does NOT require model loading or GPU.

Usage:
    python scripts/apply_fdr.py --config configs/run.yaml --layer 24 --tag train
    python scripts/apply_fdr.py --config configs/run.yaml --layers 1,12,16,20,24 --tag train
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from _common import ensure_dir, read_config, setup_logging
from analysis.fdr import benjamini_hochberg
from analysis.polarization import polarization_table, topk_feature_lists


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def process_layer(cfg: dict, layer: int, tag: str, perm_N: int, seed: int) -> None:
    """Load existing features and re-compute delta table with FDR."""
    results_dir = Path(cfg.get("results_dir", "results"))
    pol_dir = results_dir / "polarization"
    suffix = f"_{tag}" if tag else ""

    feats_path = pol_dir / f"layer{layer}{suffix}_features.npz"
    if not feats_path.exists():
        logging.warning("Skipping layer %d: %s not found", layer, feats_path)
        return

    logging.info("Loading features from %s ...", feats_path)
    arr = np.load(feats_path, allow_pickle=True)
    feats_safe = np.asarray(arr["safe_features"], dtype=np.float32)
    feats_innov = np.asarray(arr["innov_features"], dtype=np.float32)
    logging.info("  safe: %s, innov: %s", feats_safe.shape, feats_innov.shape)

    # Compute polarization table with permutation p-values.
    delta_df = polarization_table(feats_safe, feats_innov, perm_N=perm_N, seed=seed)

    # Apply FDR correction.
    fdr_q = float(cfg.get("fdr_q", 0.05))
    if "p_value" in delta_df.columns:
        fdr_result = benjamini_hochberg(delta_df["p_value"].values, q=fdr_q)
        delta_df["q_value"] = fdr_result["q_values"]
        delta_df["fdr_reject"] = fdr_result["reject"]
        n_survivors = int(delta_df["fdr_reject"].sum())
        logging.info("Layer %d: FDR correction (q<%.2f): %d / %d features survive",
                     layer, fdr_q, n_survivors, len(delta_df))

    # Compute top-k feature lists.
    topk = int(cfg.get("topk", 64))
    delta_thresh = cfg.get("delta_thresh")
    safe_ids, innov_ids = topk_feature_lists(delta_df, topk=topk, delta_thresh=delta_thresh)

    # Save updated delta CSV (overwrite).
    delta_path = pol_dir / f"layer{layer}{suffix}_delta.csv"
    delta_df.to_csv(delta_path, index=False)
    logging.info("Saved: %s", delta_path)

    # Save updated top-k files.
    safe_path = pol_dir / f"layer{layer}{suffix}_topk_safe.json"
    innov_path = pol_dir / f"layer{layer}{suffix}_topk_innov.json"
    with safe_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_ids": safe_ids}, f, ensure_ascii=False, indent=2)
    with innov_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_ids": innov_ids}, f, ensure_ascii=False, indent=2)
    logging.info("Saved: %s, %s", safe_path, innov_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-process existing features to add FDR correction."
    )
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, default=None, help="Single layer to process.")
    parser.add_argument("--layers", default=None, help="Comma-separated layers, e.g. 1,12,16,20,24")
    parser.add_argument("--tag", default="train", help="Tag suffix for feature files.")
    parser.add_argument("--perm_N", type=int, default=None, help="Overrides config perm_N.")
    args = parser.parse_args()

    setup_logging("apply_fdr")
    cfg = read_config(args.config)

    perm_N = args.perm_N if args.perm_N is not None else int(cfg.get("perm_N", 10000))
    seed = int(cfg.get("seed", 0))

    if args.layers:
        layers = _parse_int_list(args.layers)
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = cfg.get("layers", [16])

    logging.info("Processing layers %s with perm_N=%d, seed=%d", layers, perm_N, seed)

    for layer in layers:
        process_layer(cfg, layer, args.tag, perm_N, seed)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
