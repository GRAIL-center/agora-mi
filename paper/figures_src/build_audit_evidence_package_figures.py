"""Build figures for the audit evidence package paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import scienceplots  # noqa: F401
from tueplots import bundles

from figure_style import (
    BLUE,
    CARD,
    GREEN,
    GRID,
    LIGHT_BLUE,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    LIGHT_PURPLE,
    LIGHT_RED,
    MUTED,
    ORANGE,
    PAPER,
    PURPLE,
    RED,
    TEXT,
    rounded_box,
    save_figure,
    setup_matplotlib,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"
SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "audit_evidence_eval"
    / "analysis"
    / "all_conditions_qwen25_7b_final"
    / "report_quality_summary.csv"
)
HUMAN_REVIEW_PATH = (
    ROOT
    / "artifacts"
    / "audit_evidence_eval"
    / "human_review"
    / "diagnostic_review_sheet_after_gold_human_filled.csv"
)


def apply_research_plot_style(*, column: str = "full", ncols: int = 1, nrows: int = 1, base_size: float = 8.0) -> None:
    """Apply SciencePlots and tueplots settings for ICML sized figures."""

    plt.style.use(["science", "no-latex"])
    mpl.rcParams.update(bundles.icml2024(column=column, ncols=ncols, nrows=nrows, usetex=False))
    setup_matplotlib(base_size)
    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size - 1.0,
            "xtick.labelsize": base_size - 1.4,
            "ytick.labelsize": base_size - 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_pdf_and_png(fig: plt.Figure, stem: str) -> None:
    """Save a publication vector figure and a PNG preview."""

    save_figure(fig, FIGURE_DIR / f"{stem}.pdf")
    save_figure(fig, FIGURE_DIR / f"{stem}.png")


def add_arrow(ax: plt.Axes, xy_a: tuple[float, float], xy_b: tuple[float, float]) -> None:
    """Draw a compact arrow in axes coordinates."""

    arrow = FancyArrowPatch(
        xy_a,
        xy_b,
        arrowstyle="Simple,tail_width=0.6,head_width=5.0,head_length=6.0",
        transform=ax.transAxes,
        linewidth=0.0,
        facecolor=MUTED,
        edgecolor=MUTED,
        alpha=0.88,
        mutation_scale=1.0,
    )
    ax.add_patch(arrow)


def label_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    facecolor: str = CARD,
    edgecolor: str = GRID,
    title_color: str = TEXT,
) -> None:
    """Draw a labeled box with stable internal spacing."""

    rounded_box(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor, radius=0.018)
    ax.text(x + 0.018, y + h - 0.045, title, transform=ax.transAxes, fontsize=9.8, fontweight="bold", color=title_color)
    ax.text(x + 0.018, y + h - 0.091, body, transform=ax.transAxes, fontsize=7.5, color=TEXT, va="top", linespacing=1.17)


def build_package_design() -> None:
    """Render the evidence interface and evaluation design."""

    apply_research_plot_style(column="full", ncols=1, base_size=8.4)
    fig, ax = plt.subplots(figsize=(7.25, 3.45))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor(PAPER)

    label_box(
        ax,
        0.025,
        0.55,
        0.20,
        0.30,
        "Policy case",
        "AGORA passage\nFixed audit rubric\nHidden gold brief",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        title_color=BLUE,
    )
    label_box(
        ax,
        0.025,
        0.14,
        0.20,
        0.30,
        "Target model",
        "Gemma 2 2B\nResidual states\nTool evidence",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
        title_color=GREEN,
    )
    label_box(
        ax,
        0.295,
        0.58,
        0.19,
        0.25,
        "Black box",
        "Surface summary\nText spans\nOutput level notes",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        title_color=ORANGE,
    )
    label_box(
        ax,
        0.295,
        0.25,
        0.19,
        0.25,
        "White box",
        "SAE features\nLogit lens\nSteering\nActivation explanation",
        facecolor=LIGHT_PURPLE,
        edgecolor=PURPLE,
        title_color=PURPLE,
    )
    label_box(
        ax,
        0.555,
        0.42,
        0.18,
        0.28,
        "Auditor LLM",
        "Local Qwen 2.5 7B\nSame JSON schema\nCondition blinded prompt",
        facecolor=CARD,
        edgecolor=GRID,
    )
    label_box(
        ax,
        0.805,
        0.55,
        0.17,
        0.27,
        "Audit report",
        "Findings\nEvidence ids\nSupport spans\nConfidence",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        title_color=BLUE,
    )
    label_box(
        ax,
        0.805,
        0.15,
        0.17,
        0.28,
        "Evaluation",
        "Structural checks\nHuman gold review\nShuffled evidence control",
        facecolor=LIGHT_RED,
        edgecolor=RED,
        title_color=RED,
    )

    add_arrow(ax, (0.225, 0.70), (0.295, 0.70))
    add_arrow(ax, (0.225, 0.30), (0.295, 0.38))
    add_arrow(ax, (0.485, 0.70), (0.555, 0.58))
    add_arrow(ax, (0.485, 0.38), (0.555, 0.50))
    add_arrow(ax, (0.735, 0.56), (0.805, 0.68))
    add_arrow(ax, (0.885, 0.55), (0.885, 0.43))

    ax.text(
        0.025,
        0.94,
        "Evidence package audit design",
        transform=ax.transAxes,
        fontsize=12.0,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.025,
        0.90,
        "The experiment compares information access conditions, not retrieval systems.",
        transform=ax.transAxes,
        fontsize=8.0,
        color=MUTED,
    )

    save_pdf_and_png(fig, "audit_evidence_package_design")
    plt.close(fig)


def build_condition_summary() -> None:
    """Render structural report metrics for all evidence conditions."""

    summary = pd.read_csv(SUMMARY_PATH)
    order = [
        "C0_passage_only",
        "C1_blackbox_surface",
        "C2_sae_only",
        "C3_logit_lens_only",
        "C4_steering_only",
        "C5_activation_oracle_only",
        "C6_full_whitebox",
        "C7_hybrid_blackbox_whitebox",
        "C8_raw_autointerp_sae",
        "C9_shuffled_whitebox_control",
    ]
    short = {
        "C0_passage_only": "C0\npass.",
        "C1_blackbox_surface": "C1\nblack",
        "C2_sae_only": "C2\nSAE",
        "C3_logit_lens_only": "C3\nlens",
        "C4_steering_only": "C4\nsteer",
        "C5_activation_oracle_only": "C5\nexpl",
        "C6_full_whitebox": "C6\nwhite",
        "C7_hybrid_blackbox_whitebox": "C7\nhybrid",
        "C8_raw_autointerp_sae": "C8\nraw",
        "C9_shuffled_whitebox_control": "C9\nshuffle",
    }
    summary = summary.set_index("condition_id").loc[order].reset_index()

    apply_research_plot_style(column="full", ncols=2, nrows=2, base_size=7.2)
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.35, 4.85), gridspec_kw={"height_ratios": [1.0, 1.05]})
    axes = axes_grid.ravel()
    fig.patch.set_facecolor(PAPER)
    labels = [short[item] for item in summary["condition_id"]]
    x = range(len(labels))
    colors = [
        MUTED,
        ORANGE,
        PURPLE,
        PURPLE,
        PURPLE,
        PURPLE,
        PURPLE,
        GREEN,
        BLUE,
        RED,
    ]

    axes[0].bar(x, summary["cited_evidence_id_count_mean"], color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Cited evidence ids", fontsize=8.8, fontweight="bold")
    axes[0].set_ylabel("mean per report")
    axes[0].set_xticks(list(x), labels, fontsize=6.5)
    axes[0].grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    axes[0].set_ylim(0, 11.5)

    axes[1].bar(x, summary["package_evidence_item_count_mean"], color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Package evidence items", fontsize=8.8, fontweight="bold")
    axes[1].set_xticks(list(x), labels, fontsize=6.5)
    axes[1].grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    axes[1].set_ylim(0, 25.5)

    axes[2].bar(x, summary["repair_note_count_mean"], color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Repair burden", fontsize=8.8, fontweight="bold")
    axes[2].set_xticks(list(x), labels, fontsize=6.5)
    axes[2].grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    axes[2].set_ylim(0, 1.45)

    tool_columns = [
        ("blackbox_id_count_mean", "Black box", ORANGE),
        ("sae_id_count_mean", "SAE", PURPLE),
        ("logit_lens_id_count_mean", "Logit lens", BLUE),
        ("steering_vector_id_count_mean", "Steering", GREEN),
        ("activation_oracle_surrogate_id_count_mean", "Activation expl.", RED),
    ]
    bottom = pd.Series([0.0] * len(summary))
    for column, label, color in tool_columns:
        values = summary[column].astype(float)
        axes[3].bar(
            x,
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        bottom += values
    axes[3].set_title("Cited evidence type mix", fontsize=8.8, fontweight="bold")
    axes[3].set_xticks(list(x), labels, fontsize=6.5)
    axes[3].grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    axes[3].set_ylim(0, 11.5)
    axes[3].legend(frameon=False, fontsize=5.8, loc="upper left", ncols=2, columnspacing=0.8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)

    fig.suptitle("All condition structural summary", x=0.05, y=0.975, ha="left", fontsize=10.0, fontweight="bold")
    fig.text(
        0.05,
        0.922,
        "Single tool conditions change report behavior, but these structural metrics are not human validity scores.",
        fontsize=7.0,
        color=MUTED,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.89])
    save_pdf_and_png(fig, "audit_evidence_condition_summary")
    plt.close(fig)


def build_uptake_validity_summary() -> None:
    """Render the relation between evidence uptake and human validity."""

    summary = pd.read_csv(SUMMARY_PATH).set_index("condition_id")
    review = pd.read_csv(HUMAN_REVIEW_PATH)
    order = [
        "C1_blackbox_surface",
        "C6_full_whitebox",
        "C7_hybrid_blackbox_whitebox",
        "C9_shuffled_whitebox_control",
    ]
    labels = {
        "C1_blackbox_surface": "C1",
        "C6_full_whitebox": "C6",
        "C7_hybrid_blackbox_whitebox": "C7",
        "C9_shuffled_whitebox_control": "C9",
    }
    colors = {
        "C1_blackbox_surface": ORANGE,
        "C6_full_whitebox": PURPLE,
        "C7_hybrid_blackbox_whitebox": GREEN,
        "C9_shuffled_whitebox_control": RED,
    }
    means = review.groupby("condition_id").mean(numeric_only=True).loc[order]
    uptake = summary.loc[order, "cited_evidence_id_count_mean"].astype(float)

    apply_research_plot_style(column="full", ncols=2, base_size=7.7)
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.65), sharex=True)
    fig.patch.set_facecolor(PAPER)

    panels = [
        ("report_correctness_1_5", "Human correctness", "higher is better", (3.2, 4.8)),
        ("evidence_misuse_1_5", "Evidence misuse", "lower is better", (0.6, 4.75)),
    ]
    for ax, (metric, title, subtitle, ylim) in zip(axes, panels):
        for condition in order:
            ax.scatter(
                uptake.loc[condition],
                means.loc[condition, metric],
                s=105,
                color=colors[condition],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.text(
                uptake.loc[condition] + 0.14,
                means.loc[condition, metric] + 0.03,
                labels[condition],
                fontsize=7.6,
                fontweight="bold",
                color=colors[condition],
            )
        ax.set_title(title, fontsize=9.0, fontweight="bold")
        ax.text(0.02, 0.93, subtitle, transform=ax.transAxes, fontsize=6.8, color=MUTED)
        ax.set_xlabel("cited evidence ids per report")
        ax.set_ylim(*ylim)
        ax.grid(True, color=GRID, linewidth=0.55, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("mean human score")
    axes[0].set_xlim(3.2, 10.8)
    axes[1].axhline(3.0, color=GRID, linewidth=0.85, linestyle=":")

    fig.suptitle("Evidence uptake is not evidence validity", x=0.05, y=0.97, ha="left", fontsize=8.8, fontweight="bold")
    fig.text(
        0.05,
        0.905,
        "C6 and C9 cite many internal evidence ids, but human review penalizes grounding and misuse.",
        fontsize=6.6,
        color=MUTED,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.88])
    save_pdf_and_png(fig, "audit_evidence_uptake_validity")
    plt.close(fig)


def build_human_review_summary() -> None:
    """Render the pilot human review results for the main paper."""

    review = pd.read_csv(HUMAN_REVIEW_PATH)
    order = [
        "C1_blackbox_surface",
        "C6_full_whitebox",
        "C7_hybrid_blackbox_whitebox",
        "C9_shuffled_whitebox_control",
    ]
    labels = ["C1", "C6", "C7", "C9"]
    colors = [ORANGE, PURPLE, GREEN, RED]
    positive_metrics = [
        ("report_correctness_1_5", "Correctness"),
        ("span_grounding_1_5", "Grounding"),
        ("diagnostic_usefulness_1_5", "Usefulness"),
    ]

    means = review.groupby("condition_id").mean(numeric_only=True).loc[order]
    wide = review.pivot(index="case_id", columns="condition_id")

    apply_research_plot_style(column="full", ncols=3, base_size=7.3)
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.65), gridspec_kw={"width_ratios": [1.48, 0.90, 1.35]})
    fig.patch.set_facecolor(PAPER)

    x = range(len(order))
    width = 0.22
    offsets = [-width, 0.0, width]
    metric_colors = [BLUE, GREEN, ORANGE]
    for offset, (metric, label), color in zip(offsets, positive_metrics, metric_colors):
        axes[0].bar(
            [item + offset for item in x],
            means[metric],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=label,
        )
    axes[0].set_title("Report quality scores", fontweight="bold")
    axes[0].set_ylabel("mean score")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylim(0, 5.35)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.52, -0.16), fontsize=5.8, ncols=3, columnspacing=0.8)

    axes[1].bar(list(x), means["evidence_misuse_1_5"], color=colors, edgecolor="white", linewidth=0.45)
    axes[1].set_title("Evidence misuse", fontweight="bold")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylim(0, 5.35)
    axes[1].set_ylabel("lower is better")
    axes[1].axhline(3.0, color=GRID, linewidth=0.8, linestyle=":")

    comparisons = [
        ("C6 vs C1", "C6_full_whitebox", "C1_blackbox_surface"),
        ("C7 vs C1", "C7_hybrid_blackbox_whitebox", "C1_blackbox_surface"),
        ("C9 vs C1", "C9_shuffled_whitebox_control", "C1_blackbox_surface"),
    ]
    y_pos = range(len(comparisons))
    for idx, (label, condition, baseline) in enumerate(comparisons):
        diff = wide[("report_correctness_1_5", condition)] - wide[("report_correctness_1_5", baseline)]
        improved = int((diff > 0).sum())
        tied = int((diff == 0).sum())
        worse = int((diff < 0).sum())
        axes[2].barh(idx, improved, color=GREEN, edgecolor="white", linewidth=0.45, label="better" if idx == 0 else None)
        axes[2].barh(idx, tied, left=improved, color=GRID, edgecolor="white", linewidth=0.45, label="same" if idx == 0 else None)
        axes[2].barh(idx, worse, left=improved + tied, color=RED, edgecolor="white", linewidth=0.45, label="worse" if idx == 0 else None)
        axes[2].text(20.35, idx, f"{improved}/{tied}/{worse}", va="center", fontsize=5.9, color=MUTED)
    axes[2].set_title("Paired correctness", fontweight="bold")
    axes[2].set_yticks(list(y_pos), [item[0] for item in comparisons])
    axes[2].set_xlim(0, 22.2)
    axes[2].set_xlabel("cases out of 20")
    axes[2].grid(axis="x", color=GRID, linewidth=0.45, alpha=0.65)
    axes[2].grid(axis="y", visible=False)

    for ax in axes:
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.tick_params(axis="y", labelsize=6.2)

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    save_pdf_and_png(fig, "audit_evidence_human_review_summary")
    plt.close(fig)


def main() -> None:
    build_package_design()
    build_condition_summary()
    build_uptake_validity_summary()
    build_human_review_summary()


if __name__ == "__main__":
    main()
