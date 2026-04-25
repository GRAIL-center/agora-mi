"""Shared styling utilities for policy feature paper figures."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle


TEXT = "#172033"
MUTED = "#5A6472"
LIGHT_MUTED = "#7A8492"
GRID = "#D8DEE8"
PAPER = "#FFFFFF"
CARD = "#FBFCFD"
ORANGE = "#E59A2E"
LIGHT_ORANGE = "#F7DA95"
BLUE = "#2F6FB0"
LIGHT_BLUE = "#D6E5F7"
GREEN = "#2F9B73"
LIGHT_GREEN = "#D8EFE5"
RED = "#C65F46"
LIGHT_RED = "#F3CEC2"
PURPLE = "#7464A8"
LIGHT_PURPLE = "#E0DDF0"


@dataclass(frozen=True)
class SpanMark:
    """A span annotation used by the static token heatmap renderer."""

    text: str
    color: str = ORANGE
    light: str = LIGHT_ORANGE
    label: str = ""
    priority: int = 0
    start: int | None = None
    end: int | None = None
    strength: str = "medium"
    alpha: float | None = None


ACTIVATION_ALPHA = {
    "inactive": 0.00,
    "weak": 0.26,
    "medium": 0.50,
    "strong": 0.76,
    "peak": 0.95,
}


def activation_strength(value: float, reference: float) -> str:
    """Map a feature activation to a stable visual bin."""

    if value <= 0:
        return "inactive"
    if reference <= 0:
        return "strong"
    ratio = value / reference
    if ratio < 0.18:
        return "weak"
    if ratio < 0.55:
        return "medium"
    if ratio < 1.15:
        return "strong"
    return "peak"


def activation_alpha(strength: str) -> float:
    """Return the figure-wide opacity for an activation-strength bin."""

    return ACTIVATION_ALPHA.get(strength, ACTIVATION_ALPHA["medium"])


def activation_mark(
    text: str,
    *,
    color: str = ORANGE,
    light: str = LIGHT_ORANGE,
    strength: str = "medium",
    priority: int = 0,
) -> SpanMark:
    """Create a span mark whose opacity encodes activation strength."""

    return SpanMark(
        text=text,
        color=color,
        light=light,
        priority=priority,
        strength=strength,
        alpha=activation_alpha(strength),
    )


def setup_matplotlib(base_size: float = 8.7) -> None:
    """Apply a compact, paper-oriented Matplotlib style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": base_size,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "savefig.facecolor": PAPER,
        }
    )


def save_figure(fig: plt.Figure, figure_path: Path, export_path: Path | None = None) -> None:
    """Save a figure to the paper folder and optional artifact export folder."""

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=260, bbox_inches="tight")
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path, dpi=260, bbox_inches="tight")


def normalize_text(value: object) -> str:
    """Normalize whitespace while preserving readable punctuation."""

    return re.sub(r"\s+", " ", str(value)).strip()


def shorten_text(value: object, width: int = 330) -> str:
    """Shorten text at word boundaries for figure panels."""

    text = normalize_text(value)
    if len(text) <= width:
        return text
    return textwrap.shorten(text, width=width, placeholder=" ...")


def clean_token(token: object) -> str:
    """Clean model tokens for plot labels."""

    text = str(token).replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n").strip()
    text = text.replace("$", "\\$")
    return text if text else "<space>"


def add_panel_title(ax: Axes, label: str, title: str, subtitle: str | None = None) -> None:
    """Add a consistent panel title inside an axes."""

    ax.text(0.0, 1.045, f"{label}. {title}", transform=ax.transAxes, fontsize=11.2, fontweight="bold", va="bottom")
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=7.7, color=MUTED, va="top")


def rounded_box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str = CARD,
    edgecolor: str = GRID,
    linewidth: float = 0.9,
    radius: float = 0.018,
    transform=None,
    zorder: float = 0.0,
) -> FancyBboxPatch:
    """Draw a rounded rectangle in axes coordinates by default."""

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes if transform is None else transform,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    compact = normalize_text(text)
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", compact)]


def _find_span(text: str, span: str) -> tuple[int, int] | None:
    haystack = normalize_text(text)
    needle = normalize_text(span)
    if not needle:
        return None
    candidates = [needle]
    words = needle.split()
    for size in (28, 22, 16, 12, 9, 7, 5):
        if len(words) >= size:
            candidates.append(" ".join(words[:size]))
            candidates.append(" ".join(words[-size:]))
    lower_haystack = haystack.lower()
    for candidate in candidates:
        lower_candidate = candidate.lower()
        idx = lower_haystack.find(lower_candidate)
        if idx >= 0:
            return idx, idx + len(candidate)
    for size in (6, 5, 4):
        if len(words) < size:
            continue
        for start in range(0, len(words) - size + 1):
            candidate = " ".join(words[start : start + size])
            idx = lower_haystack.find(candidate.lower())
            if idx >= 0:
                return idx, idx + len(candidate)
    return None


def resolve_marks(text: str, marks: Iterable[SpanMark]) -> list[SpanMark]:
    """Return marks with resolved character spans in the display text."""

    resolved: list[SpanMark] = []
    for mark in marks:
        if mark.start is not None and mark.end is not None:
            resolved.append(mark)
            continue
        found = _find_span(text, mark.text)
        if found is None:
            continue
        resolved.append(
            SpanMark(
                text=mark.text,
                color=mark.color,
                light=mark.light,
                label=mark.label,
                priority=mark.priority,
                start=found[0],
                end=found[1],
                strength=mark.strength,
                alpha=mark.alpha,
            )
        )
    return sorted(resolved, key=lambda item: item.priority, reverse=True)


def render_token_heatmap(
    ax: Axes,
    text: str,
    marks: Iterable[SpanMark],
    *,
    x: float = 0.02,
    y: float = 0.90,
    width: float = 0.96,
    line_step: float = 0.092,
    font_size: float = 8.0,
    max_chars: int = 430,
    default_color: str = TEXT,
    draw_frame: bool = True,
) -> None:
    """Render a static token-level heatmap that is suitable for LaTeX figures."""

    display_text = shorten_text(text, max_chars)
    words = _words_with_offsets(display_text)
    resolved = resolve_marks(display_text, marks)
    if draw_frame:
        rounded_box(ax, x - 0.012, y - 0.015 - line_step * max(2, len(words) // 10), width + 0.024, 0.22, facecolor="#FFFFFF")

    cursor_x = x
    cursor_y = y
    char_w = font_size * 0.00145
    gap = 0.007
    for word, start, end in words:
        mark = next((item for item in resolved if item.start is not None and item.end is not None and end > item.start and start < item.end), None)
        clean = word
        token_w = max(0.018, len(clean) * char_w + 0.008)
        if cursor_x + token_w > x + width:
            cursor_x = x
            cursor_y -= line_step
        if mark is not None:
            alpha = activation_alpha(mark.strength) if mark.alpha is None else mark.alpha
            fill_color = mark.light
            fill_alpha = alpha
            if mark.strength == "strong":
                fill_color = mark.color
                fill_alpha = 0.48
            elif mark.strength == "peak":
                fill_color = mark.color
                fill_alpha = 0.72
            ax.add_patch(
                Rectangle(
                    (cursor_x - 0.002, cursor_y - 0.027),
                    token_w + 0.002,
                    0.042,
                    transform=ax.transAxes,
                    facecolor=to_rgba(fill_color, fill_alpha),
                    edgecolor=to_rgba(mark.color, min(0.95, alpha + 0.12)),
                    linewidth=0.25 + 0.45 * alpha,
                    zorder=0,
                )
            )
        ax.text(
            cursor_x,
            cursor_y,
            clean,
            transform=ax.transAxes,
            fontsize=font_size,
            color=default_color,
            va="center",
            fontfamily="DejaVu Sans Mono",
            zorder=1,
        )
        cursor_x += token_w + gap


def draw_small_legend(ax: Axes, items: list[tuple[str, str]], x: float, y: float, *, size: float = 0.022) -> None:
    """Draw a compact color legend in axes coordinates."""

    cursor = x
    for label, color in items:
        ax.add_patch(Rectangle((cursor, y - 0.010), size, size, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(cursor + size + 0.008, y, label, transform=ax.transAxes, va="center", fontsize=7.2, color=MUTED)
        cursor += size + 0.008 + min(0.19, 0.010 * len(label))
