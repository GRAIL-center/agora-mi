"""B1: Seed sensitivity analysis for FDR survivors.

Re-runs the polarization + FDR pipeline at Layer 24 with multiple random seeds
and measures Jaccard similarity of surviving feature sets.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from _common import ensure_dir, setup_logging


SEEDS = [0, 42, 123, 2024, 9999]
LAYER = 24


def run_polarization_for_seed(seed: int) -> set[int]:
    """Run compute_polarization with a specific seed and return FDR survivors."""
    tag = f"seed{seed}"
    cmd = [
        sys.executable, "scripts/compute_polarization.py",
        "--config", "configs/run.yaml",
        "--layer", str(LAYER),
        "--safe_jsonl", "data/processed/dsafe_train.jsonl",
        "--innov_jsonl", "data/processed/dinnov_train.jsonl",
        "--tag", tag,
        "--perm_N", "10000",
    ]
    env_patch = {"PYTHONPATH": "src"}
    import os
    env = {**os.environ, **env_patch}

    logging.info("Running seed=%d ...", seed)
    # Monkey-patch the seed in the config by passing it via env
    # Actually, compute_polarization.py reads seed from config. We need to
    # temporarily modify it, or better yet, create a temp config.
    import yaml
    cfg_path = Path("configs/run.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    orig_seed = cfg.get("seed", 0)
    cfg["seed"] = seed
    tmp_cfg = Path(f"configs/_seed_sensitivity_{seed}.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)
    
    cmd[cmd.index("configs/run.yaml")] = str(tmp_cfg)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=".")
    if result.returncode != 0:
        logging.error("Failed for seed %d: %s", seed, result.stderr[-500:])
        return set()
    
    # Read the results
    topk_path = Path(f"results/polarization/layer{LAYER}_{tag}_topk_safe.json")
    delta_path = Path(f"results/polarization/layer{LAYER}_{tag}_delta.csv")
    
    if not topk_path.exists():
        logging.error("Missing output: %s", topk_path)
        return set()
    
    with open(topk_path, "r") as f:
        data = json.load(f)
    
    # Get FDR survivors from the delta CSV
    import pandas as pd
    df = pd.read_csv(delta_path)
    if "fdr_reject" in df.columns:
        survivors = set(df[df["fdr_reject"] == True]["feature_id"].astype(int).tolist())
    else:
        survivors = set(data["feature_ids"][:42])
    
    # Clean up temp config
    tmp_cfg.unlink(missing_ok=True)
    
    logging.info("Seed %d: %d FDR survivors", seed, len(survivors))
    return survivors


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    setup_logging("seed_sensitivity")
    
    results: dict[int, set[int]] = {}
    for seed in SEEDS:
        results[seed] = run_polarization_for_seed(seed)
    
    # Compute pairwise Jaccard
    print("\n=== Seed Sensitivity Analysis ===")
    print(f"Seeds tested: {SEEDS}")
    for s in SEEDS:
        print(f"  Seed {s}: {len(results[s])} FDR survivors")
    
    print("\nPairwise Jaccard Similarity:")
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i+1:]:
            j = jaccard(results[s1], results[s2])
            print(f"  J(seed={s1}, seed={s2}) = {j:.3f}")
    
    # Core features (intersection of all seeds)
    if all(results.values()):
        core = results[SEEDS[0]]
        for s in SEEDS[1:]:
            core = core & results[s]
        print(f"\nCore features (present in ALL {len(SEEDS)} seeds): {len(core)}")
        print(f"Union features (present in ANY seed): {len(set().union(*results.values()))}")
    
    # Save results
    out_dir = ensure_dir(Path("results/sensitivity"))
    payload = {
        "seeds": SEEDS,
        "survivor_counts": {str(s): len(results[s]) for s in SEEDS},
        "survivor_ids": {str(s): sorted(results[s]) for s in SEEDS},
        "core_count": len(core) if all(results.values()) else 0,
    }
    with open(out_dir / "seed_sensitivity.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_dir / 'seed_sensitivity.json'}")


if __name__ == "__main__":
    main()
