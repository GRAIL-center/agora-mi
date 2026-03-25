from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

from data.io import read_jsonl
from data.matching import proxy_keywords
from runtime import ensure_dir, load_yaml, save_json


ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _nanmean(values: list[float]) -> float:
    filtered = [float(v) for v in values if not math.isnan(float(v))]
    return sum(filtered) / len(filtered) if filtered else float("nan")


def _clip(value: float, minimum: float, maximum: float) -> float:
    if math.isnan(value):
        return value
    return float(max(minimum, min(maximum, value)))


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config_path = _resolve_path(path)
    cfg = load_yaml(config_path)
    cfg["__config_path"] = str(config_path)
    cfg["output_root"] = str(_resolve_path(cfg["output_root"]))
    cfg["manifest_root"] = str(_resolve_path(cfg["manifest_root"]))
    cfg["policy_config"] = str(_resolve_path(cfg["policy_config"]))
    if cfg.get("manifest_summary_path"):
        cfg["manifest_summary_path"] = str(_resolve_path(cfg["manifest_summary_path"]))
    cfg.setdefault("masking", {"keyword_source": "proxy"})
    cfg["__policy_config"] = load_yaml(cfg["policy_config"])
    return cfg


def _task_paths(manifest_root: Path, family: str, proxy_slug: str) -> dict[str, dict[str, str]]:
    return {
        "positive": {
            split: str(manifest_root / family / "proxies" / proxy_slug / f"{split}.jsonl")
            for split in ("train", "dev", "test")
        },
        "negative": {
            split: str(manifest_root / family / "negatives" / proxy_slug / f"{split}.jsonl")
            for split in ("train", "dev", "test")
        },
        "validated_positive": {
            split: str(manifest_root / family / "validated" / proxy_slug / f"{split}.jsonl")
            for split in ("train", "dev", "test")
        },
        "validated_negative": {
            split: str(manifest_root / family / "validated_negatives" / proxy_slug / f"{split}.jsonl")
            for split in ("train", "dev", "test")
        },
    }


def _split_counts(path: str) -> int:
    p = Path(path)
    return len(read_jsonl(p)) if p.exists() else 0


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_task_registry(config: dict[str, Any]) -> dict[str, Any]:
    manifest_root = Path(config["manifest_root"])
    coverage_tasks: list[dict[str, Any]] = []
    coverage_task_map: dict[str, dict[str, Any]] = {}
    consistency_tasks: list[dict[str, Any]] = []
    families: dict[str, list[str]] = {}
    pair_map: dict[str, dict[str, Any]] = {}

    for pair in config["v1_pairs"]:
        family = str(pair["family"])
        left = dict(pair["left"])
        right = dict(pair["right"])
        pair_id = f"{left['proxy_slug']}__{right['proxy_slug']}"
        pair_tasks = [left, right]
        families.setdefault(family, [])
        pair_map[pair_id] = {
            "pair_id": pair_id,
            "family": family,
            "left_task_id": str(left["proxy_slug"]),
            "right_task_id": str(right["proxy_slug"]),
            "left_display_name": str(left.get("display_name", left["proxy_name"])),
            "right_display_name": str(right.get("display_name", right["proxy_name"])),
        }
        for proxy in pair_tasks:
            task_id = str(proxy["proxy_slug"])
            task = {
                "task_id": task_id,
                "family": family,
                "proxy_name": str(proxy["proxy_name"]),
                "proxy_slug": task_id,
                "display_name": str(proxy.get("display_name", proxy["proxy_name"])),
                "pair_id": pair_id,
                "paths": _task_paths(manifest_root, family, task_id),
                "mask_keywords": sorted(set(proxy_keywords(str(proxy["proxy_name"])) + proxy_keywords(str(proxy.get("display_name", ""))))),
                "matched_negative_diagnostics": _load_json_if_exists(
                    manifest_root / family / "negatives" / task_id / "diagnostics.json"
                ),
                "validated_negative_diagnostics": _load_json_if_exists(
                    manifest_root / family / "validated_negatives" / task_id / "diagnostics.json"
                ),
            }
            task["counts"] = {
                "train_positive": _split_counts(task["paths"]["positive"]["train"]),
                "dev_positive": _split_counts(task["paths"]["positive"]["dev"]),
                "test_positive": _split_counts(task["paths"]["positive"]["test"]),
                "train_negative": _split_counts(task["paths"]["negative"]["train"]),
                "dev_negative": _split_counts(task["paths"]["negative"]["dev"]),
                "test_negative": _split_counts(task["paths"]["negative"]["test"]),
                "validated_train_positive": _split_counts(task["paths"]["validated_positive"]["train"]),
                "validated_dev_positive": _split_counts(task["paths"]["validated_positive"]["dev"]),
                "validated_test_positive": _split_counts(task["paths"]["validated_positive"]["test"]),
                "validated_train_negative": _split_counts(task["paths"]["validated_negative"]["train"]),
                "validated_dev_negative": _split_counts(task["paths"]["validated_negative"]["dev"]),
                "validated_test_negative": _split_counts(task["paths"]["validated_negative"]["test"]),
            }
            coverage_tasks.append(task)
            coverage_task_map[task_id] = task
            families[family].append(task_id)

        left_task_id = str(left["proxy_slug"])
        right_task_id = str(right["proxy_slug"])
        coverage_task_map[left_task_id]["paired_target_task_id"] = right_task_id
        coverage_task_map[right_task_id]["paired_target_task_id"] = left_task_id
        consistency_tasks.append(
            {
                "task_id": f"{left_task_id}__to__{right_task_id}",
                "family": family,
                "source_task_id": left_task_id,
                "target_task_id": right_task_id,
                "family_relation": "within_family",
            }
        )
        consistency_tasks.append(
            {
                "task_id": f"{right_task_id}__to__{left_task_id}",
                "family": family,
                "source_task_id": right_task_id,
                "target_task_id": left_task_id,
                "family_relation": "within_family",
            }
        )

    registry = {
        "benchmark_name": config["benchmark_name"],
        "benchmark_id": config["benchmark_id"],
        "config_path": config["__config_path"],
        "manifest_root": config["manifest_root"],
        "coverage_tasks": coverage_tasks,
        "coverage_task_map": coverage_task_map,
        "consistency_tasks": consistency_tasks,
        "families": families,
        "pairs": pair_map,
    }
    return registry


def assert_no_leakage(task_registry: dict[str, Any]) -> None:
    for task in task_registry["coverage_tasks"]:
        split_docs: dict[str, set[str]] = {}
        for split in ("train", "dev", "test"):
            pos_rows = read_jsonl(task["paths"]["positive"][split])
            neg_rows = read_jsonl(task["paths"]["negative"][split])
            split_docs[split] = {str(row["document_id"]) for row in pos_rows + neg_rows}
        if split_docs["train"] & split_docs["dev"]:
            raise AssertionError(f"Train/dev document leakage detected for task={task['task_id']}")
        if split_docs["train"] & split_docs["test"]:
            raise AssertionError(f"Train/test document leakage detected for task={task['task_id']}")
        if split_docs["dev"] & split_docs["test"]:
            raise AssertionError(f"Dev/test document leakage detected for task={task['task_id']}")


def _load_manifest_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = config.get("manifest_summary_path")
    if not summary_path:
        return None
    return _load_json_if_exists(Path(summary_path))


def _task_file_exists(task: dict[str, Any], bucket: str) -> bool:
    return all(Path(path).exists() for path in task["paths"][bucket].values())


def build_preflight_report(config: dict[str, Any], task_registry: dict[str, Any]) -> dict[str, Any]:
    preflight_cfg = dict(config.get("preflight", {}))
    required_task_ids = sorted(str(task_id) for task_id in preflight_cfg.get("reference_test_positive_counts", {}).keys())
    actual_task_ids = sorted(task["task_id"] for task in task_registry["coverage_tasks"])

    leakage_pass = True
    leakage_error = None
    try:
        assert_no_leakage(task_registry)
    except AssertionError as exc:
        leakage_pass = False
        leakage_error = str(exc)

    exact_task_set_pass = True
    if bool(preflight_cfg.get("require_exact_task_set", False)) and required_task_ids:
        exact_task_set_pass = actual_task_ids == required_task_ids

    manifest_summary = _load_manifest_summary(config)
    actual_split_counts = {}
    actual_validated_split_counts = {}
    if manifest_summary:
        actual_split_counts = dict(manifest_summary.get("summary", {}).get("split_counts", {}))
        actual_validated_split_counts = dict(manifest_summary.get("summary", {}).get("validated_split_counts", {}))
    split_counts_pass = True
    if preflight_cfg.get("reference_split_counts"):
        split_counts_pass = actual_split_counts == dict(preflight_cfg["reference_split_counts"])
    validated_split_counts_pass = True
    if preflight_cfg.get("reference_validated_split_counts"):
        validated_split_counts_pass = actual_validated_split_counts == dict(preflight_cfg["reference_validated_split_counts"])

    task_reports: dict[str, Any] = {}
    for task in task_registry["coverage_tasks"]:
        counts = dict(task["counts"])
        reference_test_positive = preflight_cfg.get("reference_test_positive_counts", {}).get(task["task_id"])
        negative_files_exist = _task_file_exists(task, "negative")
        validated_positive_exists = _task_file_exists(task, "validated_positive")
        validated_negative_exists = _task_file_exists(task, "validated_negative")
        validated_reruns_exist = validated_positive_exists and validated_negative_exists
        if bool(preflight_cfg.get("require_validated_reruns", False)):
            validated_reruns_exist = validated_reruns_exist and counts["validated_test_positive"] > 0 and counts["validated_test_negative"] > 0
        test_positive_matches_reference = True if reference_test_positive is None else counts["test_positive"] == int(reference_test_positive)
        task_pass = all(
            [
                (negative_files_exist or not bool(preflight_cfg.get("require_negative_files", False))),
                validated_reruns_exist,
                test_positive_matches_reference,
                counts["test_negative"] > 0,
            ]
        )
        task_reports[task["task_id"]] = {
            "task_id": task["task_id"],
            "family": task["family"],
            "pair_id": task["pair_id"],
            "counts": counts,
            "paths": task["paths"],
            "negative_files_exist": negative_files_exist,
            "validated_positive_exists": validated_positive_exists,
            "validated_negative_exists": validated_negative_exists,
            "validated_reruns_exist": validated_reruns_exist,
            "reference_test_positive": reference_test_positive,
            "test_positive_matches_reference": test_positive_matches_reference,
            "task_pass": task_pass,
        }

    pair_reports: dict[str, Any] = {}
    for pair_id, pair in task_registry["pairs"].items():
        left_report = task_reports[pair["left_task_id"]]
        right_report = task_reports[pair["right_task_id"]]
        pair_pass = bool(left_report["task_pass"] and right_report["task_pass"])
        pair_reports[pair_id] = {
            **pair,
            "pair_pass": pair_pass,
            "task_ids": [pair["left_task_id"], pair["right_task_id"]],
            "failed_tasks": [task_id for task_id in [pair["left_task_id"], pair["right_task_id"]] if not task_reports[task_id]["task_pass"]],
        }

    headline_pairs = [pair_id for pair_id, payload in pair_reports.items() if payload["pair_pass"]]
    appendix_pairs = [pair_id for pair_id, payload in pair_reports.items() if not payload["pair_pass"]]
    hard_fail = not leakage_pass or not exact_task_set_pass
    headline_ready = all(
        [
            leakage_pass,
            exact_task_set_pass,
            split_counts_pass,
            validated_split_counts_pass,
            all(payload["task_pass"] for payload in task_reports.values()),
        ]
    )

    return {
        "benchmark_name": config["benchmark_name"],
        "benchmark_id": config["benchmark_id"],
        "config_path": config["__config_path"],
        "manifest_root": config["manifest_root"],
        "manifest_summary_path": config.get("manifest_summary_path"),
        "required_task_ids": required_task_ids,
        "actual_task_ids": actual_task_ids,
        "exact_task_set_pass": exact_task_set_pass,
        "leakage_pass": leakage_pass,
        "leakage_error": leakage_error,
        "reference_split_counts": preflight_cfg.get("reference_split_counts", {}),
        "actual_split_counts": actual_split_counts,
        "split_counts_pass": split_counts_pass,
        "reference_validated_split_counts": preflight_cfg.get("reference_validated_split_counts", {}),
        "actual_validated_split_counts": actual_validated_split_counts,
        "validated_split_counts_pass": validated_split_counts_pass,
        "tasks": task_reports,
        "pairs": pair_reports,
        "headline_pairs": headline_pairs,
        "appendix_pairs": appendix_pairs,
        "hard_fail": hard_fail,
        "headline_ready": headline_ready,
    }


def _enabled_methods(config: dict[str, Any]) -> list[str]:
    return [name for name, method_cfg in config["methods"].items() if bool(method_cfg.get("enabled", False))]


def summarize_method_result(
    config: dict[str, Any],
    task_registry: dict[str, Any],
    method_result: dict[str, Any],
) -> dict[str, Any]:
    task_ids = [task["task_id"] for task in task_registry["coverage_tasks"]]
    coverage_scores = [
        _safe_float(method_result.get("coverage", {}).get(task_id, {}).get("coverage_auc"))
        for task_id in task_ids
    ]
    coverage_score = _nanmean(coverage_scores)

    clip_cfg = config["scoring"]["robustness_clip"]
    robustness_values: list[float] = []
    for task_id in task_ids:
        task_payload = method_result.get("coverage", {}).get(task_id, {})
        original_auc = _safe_float(task_payload.get("coverage_auc"))
        masked_auc = _safe_float(task_payload.get("masked_coverage_auc"))
        if math.isnan(original_auc) or math.isnan(masked_auc) or math.isclose(original_auc, 0.0):
            robustness_values.append(float("nan"))
            continue
        robustness_values.append(_clip(masked_auc / original_auc, float(clip_cfg["min"]), float(clip_cfg["max"])))
    robustness_score = _nanmean(robustness_values)

    consistency_gaps: list[float] = []
    consistency_details: dict[str, float] = {}
    for directed in task_registry["consistency_tasks"]:
        task_id = directed["task_id"]
        source_task_id = directed["source_task_id"]
        within_auc = _safe_float(method_result.get("consistency", {}).get(task_id, {}).get("transfer_auc"))
        mean_cross_auc = _safe_float(
            method_result.get("cross_family_controls", {}).get(source_task_id, {}).get("mean_cross_family_auc")
        )
        gap = float("nan")
        if not math.isnan(within_auc) and not math.isnan(mean_cross_auc):
            gap = within_auc - mean_cross_auc
            consistency_gaps.append(gap)
        consistency_details[task_id] = gap
    consistency_score = _nanmean(consistency_gaps)

    causality_values = [
        _safe_float(entry.get("causality_score"))
        for entry in method_result.get("causality", {}).values()
        if str(entry.get("status", "")).lower() == "ok"
    ]
    causality_score = _nanmean(causality_values)

    core_weights = config["scoring"]["core_weights"]
    core_components = {
        "coverage": coverage_score,
        "consistency": consistency_score,
        "robustness": robustness_score,
    }
    weighted_values = []
    weighted_weights = []
    for axis, axis_value in core_components.items():
        if math.isnan(axis_value):
            continue
        weight = float(core_weights[axis])
        weighted_values.append(axis_value * weight)
        weighted_weights.append(weight)
    core_score = float(sum(weighted_values) / sum(weighted_weights)) if weighted_weights else float("nan")

    return {
        "method_name": method_result["method_name"],
        "CoverageScore": coverage_score,
        "ConsistencyScore": consistency_score,
        "RobustnessScore": robustness_score,
        "CausalityScore": causality_score,
        "CoreScore": core_score,
        "coverage_task_scores": {task_id: value for task_id, value in zip(task_ids, coverage_scores)},
        "consistency_gaps": consistency_details,
        "robustness_task_scores": {task_id: value for task_id, value in zip(task_ids, robustness_values)},
        "n_causality_families": len(causality_values),
    }


def _write_core_leaderboard(summary_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    out_path = summary_dir / "core_leaderboard.csv"
    rows = sorted(
        summaries,
        key=lambda row: (-float("-inf") if math.isnan(_safe_float(row["CoreScore"])) else -float(row["CoreScore"]), row["method_name"]),
    )
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method_name", "CoverageScore", "ConsistencyScore", "RobustnessScore", "CoreScore"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method_name": row["method_name"],
                    "CoverageScore": row["CoverageScore"],
                    "ConsistencyScore": row["ConsistencyScore"],
                    "RobustnessScore": row["RobustnessScore"],
                    "CoreScore": row["CoreScore"],
                }
            )
    return out_path


def _write_mechanistic_qualification(summary_dir: Path, summaries: list[dict[str, Any]], method_results: list[dict[str, Any]]) -> Path:
    out_path = summary_dir / "mechanistic_qualification.csv"
    summary_by_method = {row["method_name"]: row for row in summaries}
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method_name", "CausalityScore", "n_families_scored", "status"],
        )
        writer.writeheader()
        for method_result in method_results:
            method_name = method_result["method_name"]
            summary = summary_by_method[method_name]
            score = summary["CausalityScore"]
            status = "NA" if math.isnan(_safe_float(score)) else "ok"
            writer.writerow(
                {
                    "method_name": method_name,
                    "CausalityScore": "" if math.isnan(_safe_float(score)) else score,
                    "n_families_scored": summary["n_causality_families"],
                    "status": status,
                }
            )
    return out_path


def _write_preflight_report(summary_dir: Path, preflight_report: dict[str, Any]) -> Path:
    out_path = summary_dir / "preflight_report.json"
    save_json(out_path, preflight_report)
    return out_path


def _write_main_table(summary_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    out_path = summary_dir / "main_table.csv"
    rows = sorted(
        summaries,
        key=lambda row: (-float("-inf") if math.isnan(_safe_float(row["CoreScore"])) else -float(row["CoreScore"]), row["method_name"]),
    )
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method_name", "CoverageScore", "ConsistencyScore", "RobustnessScore", "CoreScore", "CausalityScore"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method_name": row["method_name"],
                    "CoverageScore": row["CoverageScore"],
                    "ConsistencyScore": row["ConsistencyScore"],
                    "RobustnessScore": row["RobustnessScore"],
                    "CoreScore": row["CoreScore"],
                    "CausalityScore": row["CausalityScore"],
                }
            )
    return out_path


def _write_coverage_task_summary(summary_dir: Path, method_results: list[dict[str, Any]], task_registry: dict[str, Any]) -> Path:
    out_path = summary_dir / "coverage_task_summary.csv"
    task_map = task_registry["coverage_task_map"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "task_id",
                "family",
                "pair_id",
                "coverage_auc",
                "validated_coverage_auc",
                "n_positive_test",
                "n_negative_test",
                "selected_layer",
                "site",
                "pooling",
            ],
        )
        writer.writeheader()
        for method_result in method_results:
            for task_id, payload in method_result.get("coverage", {}).items():
                task = task_map[task_id]
                writer.writerow(
                    {
                        "method_name": method_result["method_name"],
                        "task_id": task_id,
                        "family": task["family"],
                        "pair_id": task["pair_id"],
                        "coverage_auc": payload.get("coverage_auc"),
                        "validated_coverage_auc": payload.get("validated_coverage_auc"),
                        "n_positive_test": payload.get("n_positive_test"),
                        "n_negative_test": payload.get("n_negative_test"),
                        "selected_layer": payload.get("selected_layer"),
                        "site": payload.get("site"),
                        "pooling": payload.get("pooling"),
                    }
                )
    return out_path


def _write_consistency_summary(summary_dir: Path, method_results: list[dict[str, Any]], task_registry: dict[str, Any]) -> Path:
    out_path = summary_dir / "consistency_summary.csv"
    pair_map = task_registry["pairs"]
    task_map = task_registry["coverage_task_map"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "pair_id",
                "family",
                "source_task_id",
                "target_task_id",
                "within_family_auc",
                "mean_cross_family_auc",
                "consistency_gap",
                "selected_layer",
            ],
        )
        writer.writeheader()
        for method_result in method_results:
            for directed in task_registry["consistency_tasks"]:
                task_id = directed["task_id"]
                source_task_id = directed["source_task_id"]
                target_task_id = directed["target_task_id"]
                pair_id = task_map[source_task_id]["pair_id"]
                within_auc = _safe_float(method_result.get("consistency", {}).get(task_id, {}).get("transfer_auc"))
                mean_cross_auc = _safe_float(method_result.get("cross_family_controls", {}).get(source_task_id, {}).get("mean_cross_family_auc"))
                gap = within_auc - mean_cross_auc if not math.isnan(within_auc) and not math.isnan(mean_cross_auc) else float("nan")
                writer.writerow(
                    {
                        "method_name": method_result["method_name"],
                        "pair_id": pair_id,
                        "family": pair_map[pair_id]["family"],
                        "source_task_id": source_task_id,
                        "target_task_id": target_task_id,
                        "within_family_auc": within_auc,
                        "mean_cross_family_auc": mean_cross_auc,
                        "consistency_gap": gap,
                        "selected_layer": method_result.get("consistency", {}).get(task_id, {}).get("selected_layer"),
                    }
                )
    return out_path


def _write_robustness_summary(summary_dir: Path, method_results: list[dict[str, Any]], task_registry: dict[str, Any]) -> Path:
    out_path = summary_dir / "robustness_summary.csv"
    task_map = task_registry["coverage_task_map"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "task_id",
                "family",
                "pair_id",
                "coverage_auc",
                "masked_coverage_auc",
                "retention",
            ],
        )
        writer.writeheader()
        for method_result in method_results:
            for task_id, payload in method_result.get("coverage", {}).items():
                coverage_auc = _safe_float(payload.get("coverage_auc"))
                masked_auc = _safe_float(payload.get("masked_coverage_auc"))
                retention = float("nan")
                if not math.isnan(coverage_auc) and not math.isnan(masked_auc) and not math.isclose(coverage_auc, 0.0):
                    retention = masked_auc / coverage_auc
                task = task_map[task_id]
                writer.writerow(
                    {
                        "method_name": method_result["method_name"],
                        "task_id": task_id,
                        "family": task["family"],
                        "pair_id": task["pair_id"],
                        "coverage_auc": coverage_auc,
                        "masked_coverage_auc": masked_auc,
                        "retention": retention,
                    }
                )
    return out_path


def _write_causality_summary(summary_dir: Path, method_results: list[dict[str, Any]]) -> Path:
    out_path = summary_dir / "sae_causality_summary.csv"
    sae_result = next((result for result in method_results if result["method_name"] == "sparse_sae_feature_bank"), None)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family_name", "status", "selected_layer", "n_core_features", "causality_score", "details_path"],
        )
        writer.writeheader()
        if sae_result is not None:
            for family_name, payload in sae_result.get("causality", {}).items():
                writer.writerow(
                    {
                        "family_name": family_name,
                        "status": payload.get("status"),
                        "selected_layer": payload.get("layer"),
                        "n_core_features": payload.get("n_core_features"),
                        "causality_score": payload.get("causality_score"),
                        "details_path": payload.get("details_path"),
                    }
                )
    return out_path


def _write_appendix_task_inventory(
    summary_dir: Path,
    method_results: list[dict[str, Any]],
    task_registry: dict[str, Any],
) -> Path:
    out_path = summary_dir / "appendix_task_inventory.csv"
    task_map = task_registry["coverage_task_map"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "task_id",
                "family",
                "pair_id",
                "test_size",
                "test_positive_count",
                "validated_test_positive_count",
                "selected_layer",
                "feature_count",
                "selection_source",
            ],
        )
        writer.writeheader()
        for method_result in method_results:
            for task_id, payload in method_result.get("coverage", {}).items():
                task = task_map[task_id]
                writer.writerow(
                    {
                        "method_name": method_result["method_name"],
                        "task_id": task_id,
                        "family": task["family"],
                        "pair_id": task["pair_id"],
                        "test_size": int(payload.get("n_positive_test", 0) or 0) + int(payload.get("n_negative_test", 0) or 0),
                        "test_positive_count": payload.get("n_positive_test"),
                        "validated_test_positive_count": payload.get("n_positive_validated_test"),
                        "selected_layer": payload.get("selected_layer"),
                        "feature_count": payload.get("feature_count"),
                        "selection_source": payload.get("selection_source"),
                    }
                )
    return out_path


def _pair_mean(values: list[float]) -> float:
    return _nanmean(values)


def _evaluate_paper_readout(
    config: dict[str, Any],
    task_registry: dict[str, Any],
    summaries: list[dict[str, Any]],
    method_results: list[dict[str, Any]],
    preflight_report: dict[str, Any],
) -> dict[str, Any]:
    paper_cfg = dict(config.get("paper", {}))
    criteria_cfg = dict(paper_cfg.get("success_criteria", {}))
    summary_by_method = {row["method_name"]: row for row in summaries}
    result_by_method = {row["method_name"]: row for row in method_results}
    headline_method = str(paper_cfg.get("headline_method", "sparse_sae_feature_bank"))
    baseline_method = str(paper_cfg.get("baseline_method", "lexical_tfidf_logreg"))
    headline_summary = summary_by_method.get(headline_method, {})
    baseline_summary = summary_by_method.get(baseline_method, {})
    headline_result = result_by_method.get(headline_method, {})
    baseline_result = result_by_method.get(baseline_method, {})

    top_core_score = max((_safe_float(row["CoreScore"]) for row in summaries), default=float("nan"))
    headline_core_score = _safe_float(headline_summary.get("CoreScore"))
    is_top_core = (
        not math.isnan(headline_core_score)
        and not math.isnan(top_core_score)
        and headline_core_score >= top_core_score - 1e-12
    )

    axis_names = ["CoverageScore", "ConsistencyScore", "RobustnessScore"]
    axes_beating_baseline = [
        axis
        for axis in axis_names
        if _safe_float(headline_summary.get(axis)) > _safe_float(baseline_summary.get(axis))
    ]

    pair_consistency: dict[str, Any] = {}
    positive_consistency_pairs = 0
    for pair_id, pair in task_registry["pairs"].items():
        directions = [
            f"{pair['left_task_id']}__to__{pair['right_task_id']}",
            f"{pair['right_task_id']}__to__{pair['left_task_id']}",
        ]
        gaps = [
            _safe_float(headline_summary.get("consistency_gaps", {}).get(direction))
            for direction in directions
        ]
        mean_gap = _pair_mean(gaps)
        passes = not math.isnan(mean_gap) and mean_gap > 0.0
        if passes:
            positive_consistency_pairs += 1
        pair_consistency[pair_id] = {
            **pair,
            "directions": directions,
            "mean_consistency_gap": mean_gap,
            "passes_positive_gap": passes,
        }

    pair_robustness: dict[str, Any] = {}
    positive_robustness_pairs = 0
    for pair_id, pair in task_registry["pairs"].items():
        task_ids = [pair["left_task_id"], pair["right_task_id"]]
        headline_retentions: list[float] = []
        baseline_retentions: list[float] = []
        for task_id in task_ids:
            headline_retention = _safe_float(headline_summary.get("robustness_task_scores", {}).get(task_id))
            baseline_retention = _safe_float(baseline_summary.get("robustness_task_scores", {}).get(task_id))
            headline_retentions.append(headline_retention)
            baseline_retentions.append(baseline_retention)
        mean_headline_retention = _pair_mean(headline_retentions)
        mean_baseline_retention = _pair_mean(baseline_retentions)
        passes = (
            not math.isnan(mean_headline_retention)
            and not math.isnan(mean_baseline_retention)
            and mean_headline_retention > mean_baseline_retention
        )
        if passes:
            positive_robustness_pairs += 1
        pair_robustness[pair_id] = {
            **pair,
            "headline_mean_retention": mean_headline_retention,
            "baseline_mean_retention": mean_baseline_retention,
            "passes_beating_baseline": passes,
        }

    positive_causality_families = 0
    causality_by_family: dict[str, Any] = {}
    for family_name, payload in headline_result.get("causality", {}).items():
        score = _safe_float(payload.get("causality_score"))
        passes = str(payload.get("status", "")).lower() == "ok" and not math.isnan(score) and score > 0.0
        if passes:
            positive_causality_families += 1
        causality_by_family[family_name] = {
            **payload,
            "passes_positive_causality": passes,
        }

    criteria_results = {
        "headline_top_core": {
            "required": bool(criteria_cfg.get("require_headline_top_core", True)),
            "value": is_top_core,
            "passes": (not bool(criteria_cfg.get("require_headline_top_core", True))) or is_top_core,
        },
        "axes_beating_baseline": {
            "required_min": int(criteria_cfg.get("min_axes_beating_baseline", 2)),
            "value": len(axes_beating_baseline),
            "axes": axes_beating_baseline,
            "passes": len(axes_beating_baseline) >= int(criteria_cfg.get("min_axes_beating_baseline", 2)),
        },
        "positive_consistency_pairs": {
            "required_min": int(criteria_cfg.get("min_pairs_positive_consistency_gap", 2)),
            "value": positive_consistency_pairs,
            "passes": positive_consistency_pairs >= int(criteria_cfg.get("min_pairs_positive_consistency_gap", 2)),
        },
        "robustness_pairs_beating_baseline": {
            "required_min": int(criteria_cfg.get("min_pairs_robustness_beating_baseline", 2)),
            "value": positive_robustness_pairs,
            "passes": positive_robustness_pairs >= int(criteria_cfg.get("min_pairs_robustness_beating_baseline", 2)),
        },
        "positive_causality_families": {
            "required_min": int(criteria_cfg.get("min_positive_causality_families", 1)),
            "value": positive_causality_families,
            "passes": positive_causality_families >= int(criteria_cfg.get("min_positive_causality_families", 1)),
        },
    }
    headline_success = bool(preflight_report.get("headline_ready", False)) and all(
        item["passes"] for item in criteria_results.values()
    )

    return {
        "headline_method": headline_method,
        "baseline_method": baseline_method,
        "headline_core_score": headline_core_score,
        "top_core_score": top_core_score,
        "pair_consistency": pair_consistency,
        "pair_robustness": pair_robustness,
        "causality_by_family": causality_by_family,
        "criteria": criteria_results,
        "headline_success": headline_success,
        "headline_pairs": preflight_report.get("headline_pairs", []),
        "appendix_pairs": preflight_report.get("appendix_pairs", []),
    }


def _build_report(
    config: dict[str, Any],
    task_registry: dict[str, Any],
    method_results: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    preflight_report: dict[str, Any],
    paper_readout: dict[str, Any],
) -> dict[str, Any]:
    statistical_testing = dict(config.get("statistical_testing", {}))
    paper_cfg = dict(config.get("paper", {}))
    task_report = {}
    for task in task_registry["coverage_tasks"]:
        task_report[task["task_id"]] = {
            "family": task["family"],
            "proxy_name": task["proxy_name"],
            "mask_keywords": task["mask_keywords"],
            "counts": task["counts"],
            "matched_negative_diagnostics": task["matched_negative_diagnostics"],
            "validated_negative_diagnostics": task["validated_negative_diagnostics"],
        }
    return {
        "benchmark_name": config["benchmark_name"],
        "benchmark_id": config["benchmark_id"],
        "config_path": config["__config_path"],
        "output_root": config["output_root"],
        "n_coverage_tasks": len(task_registry["coverage_tasks"]),
        "n_consistency_tasks": len(task_registry["consistency_tasks"]),
        "statistical_testing": statistical_testing,
        "primary_comparisons": list(paper_cfg.get("primary_comparisons", [])),
        "exploratory_comparisons": list(paper_cfg.get("exploratory_comparisons", [])),
        "coverage_tasks": task_report,
        "preflight": preflight_report,
        "paper_readout": paper_readout,
        "method_summaries": {summary["method_name"]: summary for summary in summaries},
        "method_results": {result["method_name"]: result for result in method_results},
    }


def _run_method(
    method_name: str,
    runner: Callable[[dict[str, Any], dict[str, Any], Path], Any],
    config: dict[str, Any],
    task_registry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    payload = runner(config, task_registry, output_root)
    if hasattr(payload, "run"):
        payload = payload.run()
    if not isinstance(payload, dict):
        raise TypeError(f"Runner for {method_name} must return a dict result payload.")
    return payload


def run_preflight(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_benchmark_config(config_path)
    if output_root is not None:
        config["output_root"] = str(_resolve_path(output_root))
    output_dir = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_dir / "summary")
    task_registry = build_task_registry(config)
    preflight_report = build_preflight_report(config, task_registry)
    preflight_path = _write_preflight_report(summary_dir, preflight_report)
    save_json(output_dir / "task_registry.json", task_registry)
    return {
        "task_registry_path": str(output_dir / "task_registry.json"),
        "preflight_report_path": str(preflight_path),
        "hard_fail": bool(preflight_report["hard_fail"]),
        "headline_ready": bool(preflight_report["headline_ready"]),
    }


def aggregate_existing_results(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_benchmark_config(config_path)
    if output_root is not None:
        config["output_root"] = str(_resolve_path(output_root))
    output_dir = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_dir / "summary")
    task_registry = build_task_registry(config)
    preflight_report = build_preflight_report(config, task_registry)
    preflight_path = _write_preflight_report(summary_dir, preflight_report)
    save_json(output_dir / "task_registry.json", task_registry)

    method_results: list[dict[str, Any]] = []
    for method_name in _enabled_methods(config):
        result_path = output_dir / "method_results" / f"{method_name}.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing method result: {result_path}")
        method_results.append(json.loads(result_path.read_text(encoding="utf-8")))

    summaries = [summarize_method_result(config, task_registry, method_result) for method_result in method_results]
    core_leaderboard_path = _write_core_leaderboard(summary_dir, summaries)
    main_table_path = _write_main_table(summary_dir, summaries)
    mechanistic_path = _write_mechanistic_qualification(summary_dir, summaries, method_results)
    coverage_summary_path = _write_coverage_task_summary(summary_dir, method_results, task_registry)
    consistency_summary_path = _write_consistency_summary(summary_dir, method_results, task_registry)
    robustness_summary_path = _write_robustness_summary(summary_dir, method_results, task_registry)
    causality_summary_path = _write_causality_summary(summary_dir, method_results)
    appendix_inventory_path = _write_appendix_task_inventory(summary_dir, method_results, task_registry)
    paper_readout = _evaluate_paper_readout(config, task_registry, summaries, method_results, preflight_report)
    save_json(summary_dir / "paper_readout.json", paper_readout)
    report = _build_report(config, task_registry, method_results, summaries, preflight_report, paper_readout)
    save_json(summary_dir / "benchmark_report.json", report)
    return {
        "task_registry_path": str(output_dir / "task_registry.json"),
        "preflight_report_path": str(preflight_path),
        "core_leaderboard_path": str(core_leaderboard_path),
        "main_table_path": str(main_table_path),
        "mechanistic_qualification_path": str(mechanistic_path),
        "coverage_task_summary_path": str(coverage_summary_path),
        "consistency_summary_path": str(consistency_summary_path),
        "robustness_summary_path": str(robustness_summary_path),
        "sae_causality_summary_path": str(causality_summary_path),
        "appendix_task_inventory_path": str(appendix_inventory_path),
        "paper_readout_path": str(summary_dir / "paper_readout.json"),
        "benchmark_report_path": str(summary_dir / "benchmark_report.json"),
    }


def run_benchmark(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    runner_factories: dict[str, Callable[[dict[str, Any], dict[str, Any], Path], Any]] | None = None,
) -> dict[str, Any]:
    config = load_benchmark_config(config_path)
    if output_root is not None:
        config["output_root"] = str(_resolve_path(output_root))
    output_dir = ensure_dir(config["output_root"])
    method_results_dir = ensure_dir(output_dir / "method_results")
    summary_dir = ensure_dir(output_dir / "summary")

    task_registry = build_task_registry(config)
    preflight_report = build_preflight_report(config, task_registry)
    preflight_path = _write_preflight_report(summary_dir, preflight_report)
    if preflight_report["hard_fail"]:
        raise AssertionError(f"Benchmark preflight failed: {preflight_report}")
    save_json(output_dir / "task_registry.json", task_registry)

    if runner_factories is None:
        from benchmark.methods import RUNNER_FACTORIES as DEFAULT_RUNNERS

        runner_factories = DEFAULT_RUNNERS

    method_results: list[dict[str, Any]] = []
    for method_name in _enabled_methods(config):
        if method_name not in runner_factories:
            raise KeyError(f"Missing benchmark runner for method={method_name}")
        method_result = _run_method(method_name, runner_factories[method_name], config, task_registry, output_dir)
        save_json(method_results_dir / f"{method_name}.json", method_result)
        method_results.append(method_result)

    summaries = [summarize_method_result(config, task_registry, method_result) for method_result in method_results]
    core_leaderboard_path = _write_core_leaderboard(summary_dir, summaries)
    main_table_path = _write_main_table(summary_dir, summaries)
    mechanistic_path = _write_mechanistic_qualification(summary_dir, summaries, method_results)
    coverage_summary_path = _write_coverage_task_summary(summary_dir, method_results, task_registry)
    consistency_summary_path = _write_consistency_summary(summary_dir, method_results, task_registry)
    robustness_summary_path = _write_robustness_summary(summary_dir, method_results, task_registry)
    causality_summary_path = _write_causality_summary(summary_dir, method_results)
    appendix_inventory_path = _write_appendix_task_inventory(summary_dir, method_results, task_registry)
    paper_readout = _evaluate_paper_readout(config, task_registry, summaries, method_results, preflight_report)
    save_json(summary_dir / "paper_readout.json", paper_readout)
    report = _build_report(config, task_registry, method_results, summaries, preflight_report, paper_readout)
    save_json(summary_dir / "benchmark_report.json", report)
    return {
        "task_registry_path": str(output_dir / "task_registry.json"),
        "preflight_report_path": str(preflight_path),
        "core_leaderboard_path": str(core_leaderboard_path),
        "main_table_path": str(main_table_path),
        "mechanistic_qualification_path": str(mechanistic_path),
        "coverage_task_summary_path": str(coverage_summary_path),
        "consistency_summary_path": str(consistency_summary_path),
        "robustness_summary_path": str(robustness_summary_path),
        "sae_causality_summary_path": str(causality_summary_path),
        "appendix_task_inventory_path": str(appendix_inventory_path),
        "paper_readout_path": str(summary_dir / "paper_readout.json"),
        "benchmark_report_path": str(summary_dir / "benchmark_report.json"),
    }
