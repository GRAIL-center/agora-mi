from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "results" / "figures" / "paper" / "fig_methodology_policy_pipeline.png"


def add_box(ax, x: float, y: float, w: float, h: float, text: str, facecolor: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#4f4f4f",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="#1f1f1f",
        fontweight="semibold",
        wrap=True,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color="#6b7280",
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.95,
        "Policy Text Mechanistic Interpretability Pipeline",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#1f2937",
    )
    ax.text(
        0.5,
        0.91,
        "Proxy grounded discovery, cross proxy generalization, and family level mechanism tests",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4b5563",
    )

    top_y = 0.58
    box_w = 0.11
    box_h = 0.14
    xs = [0.03, 0.17, 0.31, 0.45, 0.60, 0.74]
    colors = ["#dcefd8", "#efe2d8", "#dbe8f6", "#eee3c1", "#eadcf4", "#f3d6d1"]
    labels = [
        "AGORA Segments\nAI related\nOperative\nAnnotated",
        "Proxy Families\nSustainability\nRights\nTransparency\nEquality",
        "Matched Negatives\nMetadata controls\nText similarity\nMasked lexical checks",
        "SAE Discovery\nTrain only\nLayer and site sweep\nCluster aware stats",
        "Cross Proxy Transfer\nHeld out reuse\nWithin versus cross family\nLexical baseline",
        "Family Core\nRepeat across proxies\nStable under bootstrap",
    ]

    for x, color, label in zip(xs, colors, labels):
        add_box(ax, x, top_y, box_w, box_h, label, color)

    for left, right in zip(xs[:-1], xs[1:]):
        add_arrow(ax, (left + box_w, top_y + box_h / 2), (right, top_y + box_h / 2))

    branch_source = (xs[-1] + box_w / 2, top_y)
    lower_y = 0.22
    lower_w = 0.12
    lower_h = 0.12
    lower_xs = [0.47, 0.63, 0.79]
    lower_colors = ["#cfe2f3", "#d9ead3", "#fce5cd"]
    lower_labels = [
        "Strategy Modulation\nRegress family scores on\nstrategy categories\nand metadata covariates",
        "Causal Intervention\nAblate or clamp family core\nfeatures and test held out\npolicy behavior",
        "Compositionality\nCompare multi family\nactivations against additive\nsingle family centroids",
    ]

    for x, color, label in zip(lower_xs, lower_colors, lower_labels):
        add_box(ax, x, lower_y, lower_w, lower_h, label, color)
        add_arrow(ax, branch_source, (x + lower_w / 2, lower_y + lower_h))

    ax.text(0.17, 0.51, "Proxy level analysis", fontsize=10, color="#6b7280", ha="center")
    ax.text(0.61, 0.51, "Generalization and consolidation", fontsize=10, color="#6b7280", ha="center")
    ax.text(0.66, 0.15, "Downstream mechanism tests", fontsize=10, color="#6b7280", ha="center")

    plt.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
