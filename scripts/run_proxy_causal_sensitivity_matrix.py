from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return dict(payload or {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_config", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = (repo_root / args.matrix_config).resolve()
    matrix_cfg = _load_yaml(matrix_path)
    defaults = dict(matrix_cfg.get("defaults", {}))
    jobs = list(matrix_cfg.get("jobs", []))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    command_log: list[dict[str, Any]] = []
    script_path = repo_root / "scripts" / "run_proxy_causal_sensitivity_suite.py"

    for job in jobs:
        merged = dict(defaults)
        merged.update(dict(job))
        proxy_slug = str(merged["proxy_slug"])
        out_dir = output_root / proxy_slug
        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(merged["config"]),
            "--manifest_root",
            str(merged["manifest_root"]),
            "--family",
            str(merged["family"]),
            "--proxy_slug",
            proxy_slug,
            "--paired_proxy_slug",
            str(merged["paired_proxy_slug"]),
            "--layer",
            str(int(merged["layer"])),
            "--site",
            str(merged.get("site", "resid_post")),
            "--split",
            str(merged.get("split", "test")),
            "--max_samples",
            str(int(merged.get("max_samples", 100))),
            "--random_sets",
            str(int(merged.get("random_sets", 100))),
            "--feature_ids",
            ",".join(str(value) for value in merged["feature_ids"]),
            "--k_values",
            ",".join(str(value) for value in merged.get("k_values", [1, 3, 5])),
            "--prompt_template_keys",
            ",".join(str(value) for value in merged.get("prompt_template_keys", ["proxy_forced_choice_template"])),
            "--label_normalizations",
            ",".join(str(value) for value in merged.get("label_normalizations", ["sum"])),
            "--output_dir",
            str(out_dir),
        ]
        if merged.get("prompt_config"):
            cmd.extend(["--prompt_config", str(merged["prompt_config"])])
        if merged.get("label_variant_config"):
            cmd.extend(["--label_variant_config", str(merged["label_variant_config"])])

        subprocess.run(cmd, check=True, cwd=repo_root)
        command_log.append(
            {
                "proxy_slug": proxy_slug,
                "output_dir": str(out_dir),
                "command": cmd,
            }
        )

    (output_root / "matrix_run_manifest.json").write_text(json.dumps({"matrix_config": str(matrix_path), "jobs": command_log}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
