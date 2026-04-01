from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return dict(payload or {})


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _strategy_slug(strategy: str, additional: list[str]) -> str:
    suffix = "__".join(additional) if additional else "none"
    return f"{strategy}__extra_{suffix}"


def _parse_suite(values: list[str]) -> list[tuple[str, list[str]]]:
    suite: list[tuple[str, list[str]]] = []
    for raw in values:
        left, _, right = raw.partition(":")
        strategy = left.strip()
        additional = [item.strip() for item in right.split(",") if item.strip()] if right else []
        if not strategy:
            raise ValueError(f"Invalid masking suite entry: {raw}")
        suite.append((strategy, additional))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_config", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--suite",
        nargs="+",
        required=True,
        help="Entries like keyword_mask or expanded_keyword_mask:char_mask",
    )
    parser.add_argument("--skip_preflight", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (repo_root / args.benchmark_config).resolve()
    out_root = Path(args.output_root).resolve()
    suite = _parse_suite(list(args.suite))
    base_config = _load_yaml(base_config_path)

    generated_configs: list[dict[str, Any]] = []
    for strategy, additional in suite:
        config = json.loads(json.dumps(base_config))
        config["benchmark_id"] = f"{base_config.get('benchmark_id', 'benchmark')}_{_strategy_slug(strategy, additional)}"
        config["output_root"] = str(out_root / config["benchmark_id"])
        masking = dict(config.get("masking", {}))
        masking["keyword_source"] = masking.get("keyword_source", "proxy")
        masking["strategy"] = strategy
        masking["additional_strategies"] = additional
        config["masking"] = masking
        config_path = out_root / "configs" / f"{config['benchmark_id']}.yaml"
        _write_yaml(config_path, config)
        generated_configs.append(
            {
                "strategy": strategy,
                "additional_strategies": additional,
                "config_path": str(config_path),
                "output_root": config["output_root"],
            }
        )

    command_log: list[list[str]] = []
    for entry in generated_configs:
        config_path = entry["config_path"]
        output_root = entry["output_root"]
        if Path(output_root).exists():
            shutil.rmtree(output_root)

        if not args.skip_preflight:
            command = [
                sys.executable,
                str(repo_root / "scripts" / "run_policy_feature_benchmark.py"),
                "--preflight_only",
                "--config",
                config_path,
                "--output_root",
                output_root,
            ]
            subprocess.run(command, check=True, cwd=repo_root)
            command_log.append(command)

        benchmark_command = [
            sys.executable,
            str(repo_root / "scripts" / "run_policy_feature_benchmark.py"),
            "--config",
            config_path,
            "--output_root",
            output_root,
        ]
        subprocess.run(benchmark_command, check=True, cwd=repo_root)
        command_log.append(benchmark_command)

        aggregate_command = [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_policy_feature_benchmark.py"),
            "--config",
            config_path,
            "--output_root",
            output_root,
        ]
        subprocess.run(aggregate_command, check=True, cwd=repo_root)
        command_log.append(aggregate_command)

    summary = {
        "base_config": str(base_config_path),
        "output_root": str(out_root),
        "suite": generated_configs,
        "commands": command_log,
    }
    (out_root / "masking_suite_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
