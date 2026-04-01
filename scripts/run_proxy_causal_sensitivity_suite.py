from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from _common import ensure_dir, read_config, setup_logging


ROOT = Path(__file__).resolve().parents[1]


def _split_csv(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def _load_label_variants(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {"default": {}}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    variants = data.get("variants", data)
    out: dict[str, dict[str, str]] = {}
    for name, mapping in dict(variants).items():
        out[str(name)] = {str(key): str(value) for key, value in dict(mapping or {}).items()}
    return out or {"default": {}}


def _resolve_output_path(
    *,
    cfg: dict[str, Any],
    family: str,
    proxy_slug: str,
    layer: int,
    site: str,
    k: int,
    split: str,
    output_tag: str,
) -> Path:
    results_dir = Path(cfg.get("results_dir", "results/policy_mech_interp"))
    suffix = f"_{output_tag}" if output_tag else ""
    return results_dir / "proxy_causal" / family / proxy_slug / f"layer{layer}_{site}" / f"top{k}_{split}{suffix}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_mech_interp.yaml")
    parser.add_argument("--prompt_config", default=None)
    parser.add_argument("--label_variant_config", default=None)
    parser.add_argument("--manifest_root", default="data/processed/public_values")
    parser.add_argument("--family", required=True)
    parser.add_argument("--proxy_slug", required=True)
    parser.add_argument("--paired_proxy_slug", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--random_sets", type=int, default=100)
    parser.add_argument("--feature_ids", required=True)
    parser.add_argument("--k_values", default="1,3,5")
    parser.add_argument("--prompt_template_keys", default="proxy_forced_choice_template")
    parser.add_argument("--label_normalizations", default="sum,mean_token,mean_char")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    setup_logging("run_proxy_causal_sensitivity_suite")
    cfg = read_config(args.config)
    feature_ids = [int(value) for value in _split_csv(args.feature_ids)]
    k_values = [int(value) for value in _split_csv(args.k_values)]
    prompt_template_keys = _split_csv(args.prompt_template_keys)
    label_normalizations = _split_csv(args.label_normalizations)
    label_variants = _load_label_variants(args.label_variant_config)

    output_dir = ensure_dir(Path(args.output_dir))
    rows: list[dict[str, Any]] = []
    script_path = ROOT / "scripts" / "run_proxy_causal_eval.py"

    for template_key in prompt_template_keys:
        for label_variant_name, label_map in label_variants.items():
            for label_normalization in label_normalizations:
                for k in k_values:
                    selected = feature_ids[:k]
                    if not selected:
                        continue
                    output_tag = f"{template_key}__{label_variant_name}__{label_normalization}__k{k}"
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--config",
                        str(args.config),
                        "--manifest_root",
                        str(args.manifest_root),
                        "--family",
                        str(args.family),
                        "--proxy_slug",
                        str(args.proxy_slug),
                        "--paired_proxy_slug",
                        str(args.paired_proxy_slug),
                        "--layer",
                        str(int(args.layer)),
                        "--site",
                        str(args.site),
                        "--split",
                        str(args.split),
                        "--max_samples",
                        str(int(args.max_samples)),
                        "--random_sets",
                        str(int(args.random_sets)),
                        "--feature_ids",
                        ",".join(str(value) for value in selected),
                        "--prompt_template_key",
                        str(template_key),
                        "--prompt_variant_name",
                        str(template_key),
                        "--label_normalization",
                        str(label_normalization),
                        "--output_tag",
                        output_tag,
                    ]
                    if args.prompt_config:
                        cmd.extend(["--prompt_config", str(args.prompt_config)])
                    if args.proxy_slug in label_map:
                        cmd.extend(["--target_label_override", str(label_map[args.proxy_slug])])
                    if args.paired_proxy_slug in label_map:
                        cmd.extend(["--contrast_label_override", str(label_map[args.paired_proxy_slug])])
                    subprocess.run(cmd, check=True, cwd=str(ROOT))
                    output_path = _resolve_output_path(
                        cfg=cfg,
                        family=str(args.family),
                        proxy_slug=str(args.proxy_slug),
                        layer=int(args.layer),
                        site=str(args.site),
                        k=len(selected),
                        split=str(args.split),
                        output_tag=output_tag,
                    )
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "proxy_slug": args.proxy_slug,
                            "paired_proxy_slug": args.paired_proxy_slug,
                            "family": args.family,
                            "layer": int(args.layer),
                            "site": args.site,
                            "prompt_template_key": template_key,
                            "label_variant": label_variant_name,
                            "label_normalization": label_normalization,
                            "k": len(selected),
                            "feature_ids": ", ".join(str(value) for value in selected),
                            "target_label": payload.get("intervention", {}).get("target_label"),
                            "contrast_label": payload.get("intervention", {}).get("contrast_label"),
                            "target_label_token_count": payload.get("intervention", {}).get("target_label_token_count"),
                            "contrast_label_token_count": payload.get("intervention", {}).get("contrast_label_token_count"),
                            "causal_selectivity": payload.get("causal_selectivity"),
                            "ci_low": payload.get("causal_selectivity_ci_low"),
                            "ci_high": payload.get("causal_selectivity_ci_high"),
                            "passes_positive_causality": payload.get("passes_positive_causality"),
                            "mean_target_margin_drop": payload.get("mean_target_margin_drop"),
                            "mean_random_margin_drop": payload.get("mean_random_margin_drop"),
                            "details_path": str(output_path),
                        }
                    )

    out_csv = output_dir / "proxy_causal_sensitivity_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["proxy_slug"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": vars(args),
        "n_runs": len(rows),
        "summary_csv": str(out_csv),
    }
    (output_dir / "proxy_causal_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
