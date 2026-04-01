from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest_root", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--proxy_slug", required=True)
    parser.add_argument("--paired_proxy_slug", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature_ids", required=True)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--random_sets", type=int, default=100)
    parser.add_argument("--prompt_config", default="")
    parser.add_argument("--label_normalization", default="sum")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    results_dir = str(config.get("results_dir", "results/policy_mech_interp"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = [site.strip() for site in str(args.sites).split(",") if site.strip()]

    summary_rows: list[dict[str, Any]] = []
    for site in sites:
        site_tag = f"site_{site}"
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_proxy_causal_eval.py"),
            "--config",
            args.config,
            "--manifest_root",
            args.manifest_root,
            "--family",
            args.family,
            "--proxy_slug",
            args.proxy_slug,
            "--paired_proxy_slug",
            args.paired_proxy_slug,
            "--layer",
            str(args.layer),
            "--site",
            site,
            "--feature_ids",
            args.feature_ids,
            "--k",
            str(args.k),
            "--split",
            args.split,
            "--max_samples",
            str(args.max_samples),
            "--random_sets",
            str(args.random_sets),
            "--label_normalization",
            args.label_normalization,
            "--output_tag",
            site_tag,
        ]
        if args.prompt_config:
            command.extend(["--prompt_config", args.prompt_config])

        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
        site_dir = output_dir / site
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (site_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

        result_row: dict[str, Any] = {
            "site": site,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
        }
        if completed.returncode == 0:
            intermediate_root = repo_root / results_dir / "proxy_causal" / args.family / args.proxy_slug / f"layer{args.layer}_{site}"
            candidate = intermediate_root / f"{site_tag}_{args.split}.json"
            result_row["details_path"] = str(candidate)
            if candidate.exists():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                result_row.update(
                    {
                        "mean_target_margin_drop": payload.get("mean_target_margin_drop"),
                        "mean_random_margin_drop": payload.get("mean_random_margin_drop"),
                        "causal_selectivity": payload.get("causal_selectivity"),
                        "ci_low": payload.get("ci_low"),
                        "ci_high": payload.get("ci_high"),
                        "passes_positive_causality": payload.get("passes_positive_causality"),
                    }
                )
            else:
                result_row["status"] = "missing_output"
        summary_rows.append(result_row)

    _write_csv(output_dir / "site_sensitivity_summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
