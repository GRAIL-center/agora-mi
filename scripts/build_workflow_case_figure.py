from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "paper" / "figures" / "workflow_case_reports.png"

CASES = [
    (
        "case_eu_ai_act",
        "Case 1: EU AI Act, Article 10",
        "Data governance obligations for high-risk AI systems",
    ),
    (
        "case_algorithmic_recommendation_law",
        "Case 2: Chinese Algorithmic Recommendation Provisions, Articles 10 to 12",
        "User tags, manual intervention, and transparency obligations",
    ),
]


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def shorten(text: str, width: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= width:
        return clean
    return clean[: width - 3].rstrip() + "..."


def wrap_excerpt(text: str, char_limit: int, width: int) -> str:
    return wrap(shorten(text, char_limit), width)


def build_case_frame(case_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    base = ROOT / "artifacts" / "revised_mvp_2b_16k" / "features" / "batch_scorer" / case_name
    top = pd.read_parquet(base / "top_feature_evidence.parquet")
    proxies = pd.read_parquet(base / "proxy_overlay_summary.parquet")
    causal = pd.read_parquet(base / "causal_notes.parquet")
    source = ROOT / "artifacts" / "revised_mvp_2b_16k" / "paper_cases" / f"{case_name}.txt"
    text = source.read_text(encoding="utf-8").strip().replace("\n\n", " ")
    return top, proxies, causal, text


def draw_case(ax, case_name: str, title: str, subtitle: str) -> None:
    top, proxies, causal, input_text = build_case_frame(case_name)

    policy_specific = (
        top.loc[top["ranking_family"] == "policy_specific"]
        .drop_duplicates(["layer", "feature_id"])
        .sort_values(["feature_rank", "layer"], ascending=[True, True])
        .head(2)
        .copy()
    )
    global_anchor = (
        top.loc[top["ranking_family"] == "global_dominance"]
        .drop_duplicates(["layer", "feature_id"])
        .sort_values(["feature_rank", "layer"], ascending=[True, True])
        .head(1)
        .copy()
    )
    evidence = pd.concat([policy_specific, global_anchor], ignore_index=True)
    if evidence.empty:
        evidence = (
            top.drop_duplicates(["layer", "feature_id"])
            .sort_values(["feature_rank", "layer"], ascending=[True, True])
            .head(3)
            .copy()
        )
    evidence["summary"] = evidence.apply(
        lambda row: (
            f"{row.ranking_family} | L{int(row.layer)} F{int(row.feature_id)} "
            f"{row.best_proxy}: {shorten(row.top_token_span_text, 84)}"
        ),
        axis=1,
    )

    proxy_rows = proxies.sort_values("total_weighted_proxy_signal", ascending=False).head(2).copy()
    proxy_rows["summary"] = proxy_rows.apply(
        lambda row: f"{row.proxy}: AUC {row.mean_test_auc:.3f}, active {int(row.activated_feature_count)}",
        axis=1,
    )

    causal_rows = (
        causal.drop_duplicates(["feature_id", "primary_proxy"])
        .sort_values(["kl_divergence_delta", "paired_delta"], ascending=False)
        .head(2)
        .copy()
    )
    causal_rows["summary"] = causal_rows.apply(
        lambda row: (
            f"L{int(row.layer)} F{int(row.feature_id)} {row.primary_proxy}: "
            f"KL {row.kl_divergence_delta:+.4f}, dM {row.paired_delta:+.4f}"
        ),
        axis=1,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.92,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3,
        edgecolor="#2f3e46",
        facecolor="#fbfbfd",
    )
    ax.add_patch(panel)

    ax.text(0.04, 0.93, title, fontsize=15, fontweight="bold", va="center")
    ax.text(0.04, 0.885, subtitle, fontsize=11.2, color="#4f5d75", va="center")

    input_box = FancyBboxPatch(
        (0.04, 0.62),
        0.92,
        0.18,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0.8,
        edgecolor="#c8d0d9",
        facecolor="#ffffff",
    )
    ax.add_patch(input_box)
    ax.text(0.055, 0.775, "Input excerpt", fontsize=12.2, fontweight="bold", va="center")
    ax.text(
        0.055,
        0.72,
        wrap_excerpt(input_text, char_limit=310, width=106),
        fontsize=9.4,
        va="top",
    )

    left_box = FancyBboxPatch(
        (0.04, 0.13),
        0.50,
        0.42,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0.8,
        edgecolor="#c8d0d9",
        facecolor="#ffffff",
    )
    ax.add_patch(left_box)
    ax.text(0.055, 0.515, "Policy-specific feature evidence", fontsize=12.2, fontweight="bold", va="center")
    y = 0.425
    for line in evidence["summary"]:
        ax.text(0.055, y, wrap(line, 58), fontsize=10.0, va="top")
        y -= 0.12

    right_top = FancyBboxPatch(
        (0.57, 0.34),
        0.39,
        0.21,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0.8,
        edgecolor="#c8d0d9",
        facecolor="#ffffff",
    )
    ax.add_patch(right_top)
    ax.text(0.585, 0.515, "Proxy overlays", fontsize=12.2, fontweight="bold", va="center")
    y = 0.42
    for line in proxy_rows["summary"]:
        ax.text(0.585, y, wrap(line, 34), fontsize=10.0, va="top")
        y -= 0.09

    right_bottom = FancyBboxPatch(
        (0.57, 0.13),
        0.39,
        0.16,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0.8,
        edgecolor="#c8d0d9",
        facecolor="#ffffff",
    )
    ax.add_patch(right_bottom)
    ax.text(0.585, 0.26, "Causal notes", fontsize=12.2, fontweight="bold", va="center")
    y = 0.18
    for line in causal_rows["summary"]:
        ax.text(0.585, y, wrap(line, 34), fontsize=10.0, va="top")
        y -= 0.08


def main() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10.8))
    for ax, case in zip(axes, CASES):
        draw_case(ax, *case)
    fig.suptitle("Feature-First Audit Outputs on Real Statutory Text", fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
