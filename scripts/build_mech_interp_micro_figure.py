"""Build micro-level mechanistic interpretability figures for the paper."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "revised_mvp_2b_16k"
PAPER_FIGURE_ROOT = ROOT / "paper" / "figures"
EXPORT_ROOT = ARTIFACT_ROOT / "paper_exports"


TOKEN_TRACE_FEATURE = {
    "layer": 20,
    "feature_id": 15794,
    "segment_id": "1828_4",
    "label": "Layer 20 feature 15794: High Risk Decision Systems",
}

LOGIT_ATTRIBUTION_FEATURE = {
    "layer": 12,
    "feature_id": 12608,
    "ranking_family": "policy_specific",
    "label": "Layer 12 feature 12608: Control Measures Safety",
}

CAUSAL_TARGET = {
    "target_id": "privacy_individual_top3",
    "label": "Privacy sparse top-3 ablation",
}


def main() -> None:
    PAPER_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(13, 12),
        gridspec_kw={"height_ratios": [1.25, 1.0, 1.25]},
        constrained_layout=True,
    )

    trace_frame = build_token_trace_panel(axes[0])
    logit_frame = build_logit_attribution_panel(axes[1])
    causal_frame = build_causal_segments_panel(axes[2])

    output_path = PAPER_FIGURE_ROOT / "mech_interp_micro_views.png"
    fig.suptitle("Micro-Level Views of Policy Representations", fontsize=18, weight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(EXPORT_ROOT / "mech_interp_micro_views.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    trace_frame.to_csv(EXPORT_ROOT / "mech_interp_token_trace.csv", index=False)
    logit_frame.to_csv(EXPORT_ROOT / "mech_interp_logit_attribution.csv", index=False)
    causal_frame.to_csv(EXPORT_ROOT / "mech_interp_causal_segments.csv", index=False)


def build_token_trace_panel(ax: plt.Axes) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    segment_table = pd.read_parquet(
        ARTIFACT_ROOT / "extraction" / f"segment_top_features_layer_{TOKEN_TRACE_FEATURE['layer']}.parquet"
    )
    prepared_segments = pd.read_parquet(ARTIFACT_ROOT / "prepared_segments.parquet")

    match_row = segment_table.loc[
        (segment_table["segment_id"] == TOKEN_TRACE_FEATURE["segment_id"])
        & (segment_table["feature_id"] == TOKEN_TRACE_FEATURE["feature_id"])
    ].iloc[0]
    text = prepared_segments.loc[
        prepared_segments["segment_id"] == TOKEN_TRACE_FEATURE["segment_id"], "text"
    ].iloc[0]

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    cleaned_tokens = [_clean_token(token) for token in tokens]

    token_count = len(cleaned_tokens)
    activations = np.zeros(token_count, dtype=np.float32)
    peak_positions = list(map(int, match_row["token_positions"]))
    peak_values = list(map(float, match_row["token_values"]))
    for position, value in zip(peak_positions, peak_values):
        if 0 <= position < token_count:
            activations[position] = value

    peak_mask = activations > 0
    normalized = activations / max(float(activations.max()), 1.0)
    x = np.arange(token_count)

    window_start = max(0, int(min(peak_positions)) - 8)
    window_end = min(token_count, int(max(peak_positions)) + 9)

    ax.plot(x[window_start:window_end], activations[window_start:window_end], color="#3465a4", linewidth=2.0)
    visible_peak_positions = [position for position in x[peak_mask] if window_start <= position < window_end]
    ax.scatter(visible_peak_positions, activations[visible_peak_positions], color="#cc4125", s=70, zorder=3)
    for position in visible_peak_positions:
        ax.annotate(
            cleaned_tokens[position],
            (position, activations[position]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#333333",
        )

    for idx in range(window_start, window_end):
        token = cleaned_tokens[idx]
        color = plt.cm.YlOrRd(0.15 + 0.75 * normalized[idx]) if normalized[idx] > 0 else (0.94, 0.94, 0.94, 1.0)
        ax.text(
            idx,
            -0.10 * max(float(activations.max()), 1.0),
            token,
            rotation=60,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": color, "edgecolor": "none", "boxstyle": "round,pad=0.12"},
            clip_on=False,
        )

    ax.set_title(
        "A. Token-level activation trace on a high-risk AI definition excerpt",
        fontsize=13,
        loc="left",
        weight="bold",
    )
    ax.set_ylabel("Activation magnitude")
    ax.set_xlabel("Token position")
    ax.set_xlim(window_start - 1, window_end)
    ax.grid(alpha=0.2, axis="y")
    ax.text(
        0.01,
        0.98,
        "The feature spikes on the exclusion list, especially anti-fraud and anti-malware tokens,\n"
        "suggesting that the high-risk representation also tracks statutory carve-outs.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )

    return pd.DataFrame(
        {
            "token_index": x,
            "token": cleaned_tokens,
            "activation": activations,
            "normalized_activation": normalized,
            "segment_id": TOKEN_TRACE_FEATURE["segment_id"],
            "layer": TOKEN_TRACE_FEATURE["layer"],
            "feature_id": TOKEN_TRACE_FEATURE["feature_id"],
        }
    )


def build_logit_attribution_panel(ax: plt.Axes) -> pd.DataFrame:
    logit_table = pd.read_parquet(ARTIFACT_ROOT / "features" / "feature_catalog_logit_attribution.parquet")
    row = logit_table.loc[
        (logit_table["layer"] == LOGIT_ATTRIBUTION_FEATURE["layer"])
        & (logit_table["feature_id"] == LOGIT_ATTRIBUTION_FEATURE["feature_id"])
        & (logit_table["ranking_family"] == LOGIT_ATTRIBUTION_FEATURE["ranking_family"])
    ].iloc[0]

    positive_tokens = [_clean_token(token) for token in row["top_positive_tokens"][:8]]
    positive_scores = list(map(float, row["top_positive_scores"][:8]))
    negative_tokens = [_clean_token(token) for token in row["top_negative_tokens"][:8]]
    negative_scores = [-abs(float(score)) for score in row["top_negative_scores"][:8]]

    token_labels = negative_tokens[::-1] + positive_tokens
    token_scores = negative_scores[::-1] + positive_scores
    colors = ["#7f8c8d"] * len(negative_tokens) + ["#2e8b57"] * len(positive_tokens)

    y = np.arange(len(token_labels))
    ax.barh(y, token_scores, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(token_labels, fontsize=8)
    ax.set_xlabel("Decoder to logit projection")
    ax.set_title(
        "B. Decoder-to-logit attribution for a safety-control feature",
        fontsize=13,
        loc="left",
        weight="bold",
    )
    ax.grid(alpha=0.2, axis="x")
    ax.text(
        0.01,
        0.02,
        "The feature directly promotes control-like continuations and suppresses unrelated tokens,\n"
        "providing white-box evidence beyond the AutoInterp label alone.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )

    rows = []
    for token, score in zip(positive_tokens, positive_scores):
        rows.append({"direction": "promote", "token": token, "score": score})
    for token, score in zip(negative_tokens, negative_scores):
        rows.append({"direction": "suppress", "token": token, "score": score})
    return pd.DataFrame(rows)


def build_causal_segments_panel(ax: plt.Axes) -> pd.DataFrame:
    causal_segments = pd.read_parquet(ARTIFACT_ROOT / "interventions" / "feature_causal_per_segment.parquet")
    prepared_segments = pd.read_parquet(ARTIFACT_ROOT / "prepared_segments.parquet")
    target_rows = causal_segments.loc[
        causal_segments["target_id"] == CAUSAL_TARGET["target_id"]
    ].copy()
    target_rows = target_rows.merge(
        prepared_segments[["segment_id", "text"]],
        on="segment_id",
        how="left",
    )
    target_rows = target_rows.sort_values("kl_divergence_delta", ascending=False).head(6).copy()
    target_rows["snippet"] = target_rows["text"].map(_make_snippet_label)
    target_rows = target_rows.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(target_rows))
    ax.barh(y, target_rows["kl_divergence_delta"], color="#8e44ad", alpha=0.85, label="KL delta")
    ax.scatter(
        target_rows["top1_change_rate_delta"],
        y,
        color="#d35400",
        s=60,
        label="Top-1 change delta",
        zorder=3,
    )
    for idx, row in target_rows.iterrows():
        ax.text(
            float(row["kl_divergence_delta"]) + 0.003,
            idx,
            f"Δtop1={row['top1_change_rate_delta']:.03f}",
            va="center",
            fontsize=8,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(target_rows["snippet"], fontsize=8)
    ax.set_xlabel("Effect relative to matched random control")
    ax.set_title(
        "C. Per-passage causal effect for the privacy sparse top-3 set",
        fontsize=13,
        loc="left",
        weight="bold",
    )
    ax.grid(alpha=0.2, axis="x")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.text(
        0.01,
        0.98,
        "Strong effects concentrate on privacy-heavy passages such as autonomy and PET clauses,\n"
        "supporting the claim that policy semantics are distributed across small sparse sets.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )

    return target_rows[
        [
            "segment_id",
            "snippet",
            "kl_divergence_delta",
            "top1_change_rate_delta",
            "perplexity_shift_delta",
        ]
    ].copy()


def _clean_token(token: str) -> str:
    cleaned = (
        token.replace("▁", " ")
        .replace("<bos>", "<bos>")
        .replace("<eos>", "<eos>")
        .replace("\n", "\\n")
    )
    cleaned = cleaned.strip()
    if not cleaned:
        return "space"
    return cleaned


def _make_snippet_label(text: str) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= 72:
        return compact
    return compact[:69] + "..."


if __name__ == "__main__":
    main()
