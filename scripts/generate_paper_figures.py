"""Generate publication-quality figures for the Safety vs Innovation paper.

Usage:
    python scripts/generate_paper_figures.py --config configs/run.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from _common import ensure_dir, read_config, setup_logging

# ── Style constants ──────────────────────────────────────────────────────────
FONT_FAMILY = "sans-serif"
FONT_SIZE = 11
TITLE_SIZE = 13
SAFE_COLOR = "#2563EB"   # Blue
INNOV_COLOR = "#F97316"  # Orange
RANDOM_COLOR = "#9CA3AF" # Gray
FIG_DPI = 300
CAP_COLOR = "#10B981"    # Teal
SCALE_COLOR = "#8B5CF6"  # Purple

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "figure.dpi": 120,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
})


def _fig1_ablation_heatmap(sweep_csv: Path, out_dir: Path) -> Path:
    """Cross-layer ablation effect heatmap (layers × k-values)."""
    df = pd.read_csv(sweep_csv)
    if "mean_delta_safe_minus_random" not in df.columns:
        logging.warning("Fig1: sweep CSV missing required column.")
        return out_dir / "fig1_ablation_heatmap.png"

    pivot = df.pivot_table(
        index="layer", columns="k", values="mean_delta_safe_minus_random", aggfunc="first"
    )
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-max(abs(pivot.values.min()), abs(pivot.values.max())),
                   vmax=max(abs(pivot.values.min()), abs(pivot.values.max())))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"k={int(c)}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"L{int(r)}" for r in pivot.index])

    # Annotate cells.
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if abs(val) > 0.5 * abs(pivot.values).max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ(safe − random) logprob shift")
    ax.set_title("Causal Ablation Effect: Layer × k Sweep")
    ax.set_xlabel("Top-k features ablated")
    ax.set_ylabel("Residual stream layer")

    out_path = out_dir / "fig1_ablation_heatmap.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logging.info("Saved Fig1: %s", out_path)
    return out_path


def _fig2_dose_response(clamp_csv: Path, out_dir: Path) -> Path:
    """Clamp dose-response curve for Layer 24 (cap vs scale modes)."""
    df = pd.read_csv(clamp_csv)
    required_cols = {"mode", "alpha", "mean_delta_safe_minus_random"}
    if not required_cols.issubset(df.columns):
        logging.warning("Fig2: clamp CSV missing required columns (%s)", required_cols - set(df.columns))
        return out_dir / "fig2_dose_response.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, color, marker, label in [
        ("cap", CAP_COLOR, "o", "Cap mode"),
        ("scale", SCALE_COLOR, "s", "Scale mode"),
    ]:
        sub = df[df["mode"] == mode].sort_values("alpha")
        if sub.empty:
            continue
        ax.plot(sub["alpha"], sub["mean_delta_safe_minus_random"], f"-{marker}", color=color, label=label,
                linewidth=2, markersize=6, alpha=0.9)

        # Error bars from CI columns if available.
        if "ci_low" in df.columns and "ci_high" in df.columns:
            ax.fill_between(sub["alpha"], sub["ci_low"], sub["ci_high"],
                           alpha=0.15, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5, label="α=1.0 (baseline)")
    ax.set_xlabel("Alpha (intervention strength)")
    ax.set_ylabel("Δ(safe − random) on P(increase)")
    ax.set_title("Interference Clamp Dose-Response — Layer 24")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "fig2_dose_response.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logging.info("Saved Fig2: %s", out_path)
    return out_path


def _fig3_volcano(delta_csv: Path, out_dir: Path) -> Path:
    """FDR volcano plot: delta vs -log10(q_value)."""
    df = pd.read_csv(delta_csv)
    if "q_value" not in df.columns:
        logging.warning("Fig3: delta CSV missing q_value column. Run with FDR first.")
        return out_dir / "fig3_volcano.png"

    df["neg_log10_q"] = -np.log10(np.clip(df["q_value"].values, 1e-300, 1.0))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Non-significant features.
    ns = df[~df["fdr_reject"]]
    sig = df[df["fdr_reject"]]
    sig_safe = sig[sig["delta"] > 0]
    sig_innov = sig[sig["delta"] < 0]

    ax.scatter(ns["delta"], ns["neg_log10_q"], s=6, alpha=0.25, c="#D1D5DB", label="Not significant", zorder=1)
    ax.scatter(sig_safe["delta"], sig_safe["neg_log10_q"], s=12, alpha=0.7, c=SAFE_COLOR,
               label=f"Safety (n={len(sig_safe)})", zorder=2)
    ax.scatter(sig_innov["delta"], sig_innov["neg_log10_q"], s=12, alpha=0.7, c=INNOV_COLOR,
               label=f"Innovation (n={len(sig_innov)})", zorder=2)

    # Significance threshold line.
    q_thresh = -np.log10(0.05)
    ax.axhline(q_thresh, color="red", linewidth=1, linestyle="--", alpha=0.6, label="q = 0.05")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Delta (mean_safe − mean_innov)")
    ax.set_ylabel("−log₁₀(q-value)")
    ax.set_title("Volcano Plot: Feature Polarization with FDR Correction")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper right")
    ax.grid(True, alpha=0.2)

    out_path = out_dir / "fig3_volcano.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logging.info("Saved Fig3: %s", out_path)
    return out_path


def _fig4_circuit_viz(graphs_dir: Path, layer: int, out_dir: Path) -> Path:
    """Circuit graph visualization comparing Safety vs Innovation."""
    safe_path = graphs_dir / f"circuit_safe_layer{layer}.json"
    innov_path = graphs_dir / f"circuit_innov_layer{layer}.json"
    out_path = out_dir / "fig4_circuit_comparison.png"

    if not safe_path.exists() or not innov_path.exists():
        logging.warning("Fig4: circuit graph JSONs not found. Skipping.")
        return out_path

    try:
        import networkx as nx
    except ImportError:
        logging.warning("Fig4: networkx not installed. Skipping circuit viz.")
        return out_path

    from analysis.circuit_graph import CircuitGraph

    safe_g = CircuitGraph.load_json(safe_path)
    innov_g = CircuitGraph.load_json(innov_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, graph, title, color in [
        (axes[0], safe_g, "Safety Circuit", SAFE_COLOR),
        (axes[1], innov_g, "Innovation Circuit", INNOV_COLOR),
    ]:
        G = nx.DiGraph()
        for nid, node in graph.nodes.items():
            G.add_node(nid, layer=node.layer or 0)
        for edge in graph.edges:
            G.add_edge(edge.src, edge.dst, weight=abs(edge.weight))

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No nodes", transform=ax.transAxes, ha="center", fontsize=14)
            ax.set_title(title)
            continue

        # Layout by layer for hierarchical structure.
        pos = {}
        layer_groups: dict[int, list[str]] = {}
        for nid, data in G.nodes(data=True):
            l = data.get("layer", 0)
            layer_groups.setdefault(l, []).append(nid)

        for l, nids in sorted(layer_groups.items()):
            for idx, nid in enumerate(nids):
                pos[nid] = (idx - len(nids) / 2, -l)

        edge_weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
        max_w = max(edge_weights) if edge_weights else 1.0
        edge_widths = [0.5 + 2.5 * (w / max_w) for w in edge_weights]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color=color, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.4,
                              edge_color=color, arrows=True, arrowsize=8)
        ax.set_title(f"{title}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        ax.axis("off")

    fig.suptitle(f"Circuit Comparison — Layer {layer}", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logging.info("Saved Fig4: %s", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures."
    )
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, default=24, help="Primary layer for figures.")
    args = parser.parse_args()

    setup_logging("generate_paper_figures")
    cfg = read_config(args.config)
    results_dir = Path(cfg.get("results_dir", "results"))
    out_dir = ensure_dir(results_dir / "figures" / "paper")

    generated = []

    # Fig 1: Cross-layer ablation heatmap.
    sweep_csv = results_dir / "stats" / "sweep_summary.csv"
    if sweep_csv.exists():
        generated.append(_fig1_ablation_heatmap(sweep_csv, out_dir))
    else:
        logging.warning("Skipping Fig1: %s not found", sweep_csv)

    # Fig 2: Clamp dose-response.
    clamp_csv = results_dir / "stats" / "clamp_l24_cap_scale_summary.csv"
    if clamp_csv.exists():
        generated.append(_fig2_dose_response(clamp_csv, out_dir))
    else:
        logging.warning("Skipping Fig2: %s not found", clamp_csv)

    # Fig 3: Volcano plot (needs FDR-augmented delta CSV).
    delta_csv = results_dir / "polarization" / f"layer{args.layer}_train_delta.csv"
    if delta_csv.exists():
        generated.append(_fig3_volcano(delta_csv, out_dir))
    else:
        logging.warning("Skipping Fig3: %s not found", delta_csv)

    # Fig 4: Circuit graph comparison.
    graphs_dir = results_dir / "graphs"
    generated.append(_fig4_circuit_viz(graphs_dir, args.layer, out_dir))

    print(f"\nGenerated {len(generated)} figures in {out_dir}/:")
    for p in generated:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
