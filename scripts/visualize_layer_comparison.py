from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ensure_dir, read_config, setup_logging


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    # SVD-based PCA (no sklearn dependency)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    coords = x_centered @ vt[:2].T
    var_ratio = (s**2) / np.sum(s**2)
    return coords[:, 0], coords[:, 1], var_ratio[:2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    setup_logging("visualize_layer_comparison")
    cfg = read_config(args.config)
    results_dir = Path(cfg.get("results_dir", "results"))

    pol_csv = results_dir / "polarization" / f"layer{args.layer}_delta.csv"
    feat_npz = results_dir / "polarization" / f"layer{args.layer}_dev_features.npz"
    if not pol_csv.exists():
        raise FileNotFoundError(f"Missing polarization CSV: {pol_csv}")
    if not feat_npz.exists():
        raise FileNotFoundError(f"Missing feature NPZ: {feat_npz}")

    delta_df = pd.read_csv(pol_csv)
    arr = np.load(feat_npz, allow_pickle=True)
    safe_features = np.asarray(arr["safe_features"], dtype=np.float32)
    innov_features = np.asarray(arr["innov_features"], dtype=np.float32)

    topk = min(args.topk, len(delta_df) // 2)
    top_safe = delta_df.nlargest(topk, "delta").copy()
    top_innov = delta_df.nsmallest(topk, "delta").copy()

    out_dir = ensure_dir(args.out_dir)
    combined_path = out_dir / f"layer{args.layer}_safety_vs_innovation_overview.png"
    bars_path = out_dir / f"layer{args.layer}_top_features.png"
    pca_path = out_dir / f"layer{args.layer}_pca.png"

    # 1) Top features bar chart
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    splot = top_safe.sort_values("delta", ascending=True)
    ax1.barh(splot["feature_id"].astype(str), splot["delta"], color="#1f77b4")
    ax1.set_title(f"Top +{topk} Safety Features (Layer {args.layer})")
    ax1.set_xlabel("delta = mean_safe - mean_innov")

    ax2 = fig.add_subplot(1, 2, 2)
    iplot = top_innov.sort_values("delta", ascending=True)
    ax2.barh(iplot["feature_id"].astype(str), iplot["delta"], color="#ff7f0e")
    ax2.set_title(f"Top -{topk} Innovation Features (Layer {args.layer})")
    ax2.set_xlabel("delta = mean_safe - mean_innov")
    fig.tight_layout()
    fig.savefig(bars_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) PCA scatter
    x = np.concatenate([safe_features, innov_features], axis=0)
    labels = np.array(["safe"] * len(safe_features) + ["innov"] * len(innov_features))
    pc1, pc2, vr = _pca_2d(x)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    safe_mask = labels == "safe"
    innov_mask = labels == "innov"
    ax.scatter(pc1[safe_mask], pc2[safe_mask], s=16, alpha=0.65, label="Dsafe", c="#1f77b4")
    ax.scatter(pc1[innov_mask], pc2[innov_mask], s=16, alpha=0.65, label="Dinnov", c="#ff7f0e")
    ax.set_title(
        f"PCA of SAE Features (Layer {args.layer})\n"
        f"PC1={vr[0]*100:.1f}% var, PC2={vr[1]*100:.1f}% var"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pca_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) Combined overview panel
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

    # Delta histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(delta_df["delta"], bins=80, color="#4c78a8", alpha=0.9)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Delta Distribution")
    ax.set_xlabel("delta")
    ax.set_ylabel("count")

    # Mean-safe vs mean-innov scatter
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(delta_df["mean_innov"], delta_df["mean_safe"], s=7, alpha=0.35, c="#6c6c6c")
    mins = min(delta_df["mean_innov"].min(), delta_df["mean_safe"].min())
    maxs = max(delta_df["mean_innov"].max(), delta_df["mean_safe"].max())
    ax.plot([mins, maxs], [mins, maxs], "k--", lw=1)
    ax.set_title("Feature Means: Innovation vs Safety")
    ax.set_xlabel("mean_innov")
    ax.set_ylabel("mean_safe")

    # Top-k bars (safe positive)
    ax = fig.add_subplot(gs[1, 0])
    splot = top_safe.sort_values("delta", ascending=True)
    ax.barh(splot["feature_id"].astype(str), splot["delta"], color="#1f77b4")
    ax.set_title(f"Top +{topk} Safety Features")
    ax.set_xlabel("delta")

    # PCA
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(pc1[safe_mask], pc2[safe_mask], s=12, alpha=0.6, label="Dsafe", c="#1f77b4")
    ax.scatter(pc1[innov_mask], pc2[innov_mask], s=12, alpha=0.6, label="Dinnov", c="#ff7f0e")
    ax.set_title(f"PCA Separation (Layer {args.layer})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    fig.suptitle(f"Safety vs Innovation Overview (Layer {args.layer})", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved figure: %s", combined_path)
    logging.info("Saved figure: %s", bars_path)
    logging.info("Saved figure: %s", pca_path)
    print(combined_path)
    print(bars_path)
    print(pca_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
