from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

PROXY_ORDER = [
    "risk_factors_bias",
    "harms_discrimination",
    "risk_factors_privacy",
    "harms_violation_of_civil_or_human_rights_including_privacy",
    "risk_factors_transparency",
    "risk_factors_interpretability_and_explainability",
]

MODEL_MAP = {
    "policy_feature_benchmark_locked_2b": "Gemma 2 2B",
    "policy_feature_benchmark_locked_9b_primary": "Gemma 2 9B",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any) -> float:
    try:
        if value in {None, "", "nan", "NaN"}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _proxy_sort_key(proxy_slug: str) -> tuple[int, str]:
    if proxy_slug in PROXY_ORDER:
        return (PROXY_ORDER.index(proxy_slug), proxy_slug)
    return (len(PROXY_ORDER), proxy_slug)


def _classify(strict_positive: bool, relaxed_count: int, base_status: str) -> str:
    if strict_positive:
        return "strict_positive"
    if relaxed_count > 0:
        return "exploratory_positive"
    if base_status in {"stable_proxy_signal", "tested_not_passed", "ok"}:
        return "stable_without_positive_causality"
    return "no_positive_signal"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", required=True)
    parser.add_argument("--sensitivity_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--exploratory_effect_min", type=float, default=0.02)
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    sensitivity_root = Path(args.sensitivity_root).resolve()
    output_root = _ensure_dir(Path(args.output_root).resolve())
    effect_min = float(args.exploratory_effect_min)

    base_rows: list[dict[str, Any]] = []
    for folder_name, model_label in MODEL_MAP.items():
        summary_path = bundle_root / "results" / folder_name / "summary" / "proxy_causal_summary.csv"
        if not summary_path.exists():
            continue
        for row in _read_csv(summary_path):
            row["model_label"] = model_label
            base_rows.append(row)

    sensitivity_rows: list[dict[str, Any]] = []
    for path in sensitivity_root.rglob("proxy_causal_sensitivity_summary.csv"):
        model_label = "Unknown model"
        path_text = str(path).lower()
        if "9b" in path_text:
            model_label = "Gemma 2 9B"
        elif "2b" in path_text:
            model_label = "Gemma 2 2B"
        for row in _read_csv(path):
            row["model_label"] = model_label
            row["source_path"] = str(path)
            sensitivity_rows.append(row)

    sensitivity_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sensitivity_rows:
        sensitivity_by_key[(str(row["model_label"]), str(row["proxy_slug"]))].append(row)

    per_model_rows: list[dict[str, Any]] = []
    grouped_proxy_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in base_rows:
        model_label = str(row["model_label"])
        proxy_slug = str(row["proxy_slug"])
        key = (model_label, proxy_slug)
        base_status = str(row.get("status", ""))
        strict_positive = str(row.get("passes_positive_causality", "")).lower() == "true"
        variants = sensitivity_by_key.get(key, [])
        relaxed = [
            item for item in variants
            if _safe_float(item.get("causal_selectivity")) >= effect_min and _safe_float(item.get("ci_high")) > 0.0
        ]
        positive = [item for item in variants if _safe_float(item.get("causal_selectivity")) > 0.0]
        prompt_support = sorted({str(item.get("prompt_template_key")) for item in relaxed})
        k_support = sorted({str(item.get("k")) for item in relaxed}, key=lambda value: int(value))
        label_support = sorted({str(item.get("label_normalization")) for item in relaxed})
        max_variant = max((_safe_float(item.get("causal_selectivity")) for item in variants), default=float("nan"))
        mean_variant = (
            sum(_safe_float(item.get("causal_selectivity")) for item in variants) / len(variants)
            if variants else float("nan")
        )
        out_row = {
            "model_label": model_label,
            "proxy_slug": proxy_slug,
            "proxy_name": row.get("proxy_name"),
            "selected_layer": row.get("selected_layer"),
            "strict_positive": strict_positive,
            "base_status": base_status,
            "base_selectivity": row.get("causal_selectivity"),
            "base_ci_low": row.get("ci_low"),
            "base_ci_high": row.get("ci_high"),
            "sensitivity_run_count": len(variants),
            "positive_variant_count": len(positive),
            "relaxed_support_count": len(relaxed),
            "max_variant_selectivity": max_variant,
            "mean_variant_selectivity": mean_variant,
            "supported_k_values": "; ".join(k_support),
            "supported_prompt_templates": "; ".join(prompt_support),
            "supported_label_normalizations": "; ".join(label_support),
            "evidence_tier": _classify(strict_positive, len(relaxed), base_status),
        }
        per_model_rows.append(out_row)
        grouped_proxy_rows[proxy_slug].append(out_row)

    aggregate_rows: list[dict[str, Any]] = []
    for proxy_slug, rows in grouped_proxy_rows.items():
        rows.sort(key=lambda item: item["model_label"])
        strict_models = [item["model_label"] for item in rows if item["strict_positive"]]
        relaxed_models = [item["model_label"] for item in rows if int(item["relaxed_support_count"]) > 0]
        if len(strict_models) >= 2:
            aggregate_tier = "replicated_strict_positive"
        elif len(strict_models) == 1:
            aggregate_tier = "single_model_strict_positive"
        elif len(relaxed_models) >= 1:
            aggregate_tier = "exploratory_only_signal"
        else:
            aggregate_tier = "stable_or_null_only"
        aggregate_rows.append(
            {
                "proxy_slug": proxy_slug,
                "proxy_name": rows[0].get("proxy_name"),
                "strict_positive_models": "; ".join(strict_models),
                "exploratory_positive_models": "; ".join(relaxed_models),
                "aggregate_evidence_tier": aggregate_tier,
                "max_observed_selectivity": max(_safe_float(item.get("max_variant_selectivity")) for item in rows),
                "base_models": "; ".join(f"{item['model_label']}:{item['base_status']}" for item in rows),
            }
        )

    per_model_rows.sort(key=lambda item: (item["model_label"], _proxy_sort_key(item["proxy_slug"])))
    aggregate_rows.sort(key=lambda item: _proxy_sort_key(item["proxy_slug"]))

    _write_csv(output_root / "exploratory_proxy_evidence_by_model.csv", per_model_rows)
    _write_csv(output_root / "exploratory_proxy_evidence_gradient.csv", aggregate_rows)

    readme_lines = [
        "# Exploratory Proxy Evidence Gradient",
        "",
        "This appendix-oriented summary separates strict main-paper causal support from exploratory sensitivity support.",
        "",
        "## Interpretation rules",
        "",
        "1. Strict positive means the locked causal test passed the original positive-CI criterion.",
        f"2. Exploratory positive means at least one sensitivity variant reached causal selectivity >= {effect_min:.3f} with a positive upper confidence bound.",
        "3. Exploratory positives are not promoted to headline claims.",
        "4. Use these rows to show evidence gradients, not to replace the strict claim ladder.",
        "",
        "## Aggregate tiers",
        "",
    ]
    for row in aggregate_rows:
        readme_lines.append(
            f"1. {row['proxy_name']}: {row['aggregate_evidence_tier']} | strict models: {row['strict_positive_models'] or 'none'} | exploratory models: {row['exploratory_positive_models'] or 'none'}"
        )

    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
