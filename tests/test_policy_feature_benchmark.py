from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.policy_feature_benchmark import (
    aggregate_existing_results,
    assert_no_leakage,
    build_task_registry,
    load_benchmark_config,
    run_preflight,
    run_benchmark,
    summarize_method_result,
)
from data.io import write_jsonl


def _make_row(*, family: str, proxy_name: str, proxy_slug: str, split: str, doc_id: str, segment_id: str) -> dict:
    return {
        "id": segment_id,
        "segment_id": segment_id,
        "doc_id": doc_id,
        "document_id": doc_id,
        "text": f"{proxy_name} text {segment_id}",
        "summary": None,
        "split": split,
        "family_name": family,
        "proxy_name": proxy_name,
        "proxy_slug": proxy_slug,
        "strategy_categories": [],
        "quality_flags": {"segment_validated": True},
        "metadata": {"authority": "agency", "jurisdiction": "us_federal", "document_form": "law", "year": 2024},
    }


def _build_fixture(manifest_root: Path, config: dict) -> None:
    split_doc_suffix = {"train": "tr", "dev": "dv", "test": "te"}
    for pair in config["v1_pairs"]:
        family = pair["family"]
        for proxy in (pair["left"], pair["right"]):
            proxy_slug = proxy["proxy_slug"]
            proxy_name = proxy["proxy_name"]
            proxy_dir = manifest_root / family / "proxies" / proxy_slug
            neg_dir = manifest_root / family / "negatives" / proxy_slug
            validated_dir = manifest_root / family / "validated" / proxy_slug
            validated_neg_dir = manifest_root / family / "validated_negatives" / proxy_slug
            proxy_dir.mkdir(parents=True, exist_ok=True)
            neg_dir.mkdir(parents=True, exist_ok=True)
            validated_dir.mkdir(parents=True, exist_ok=True)
            validated_neg_dir.mkdir(parents=True, exist_ok=True)
            diagnostics = {}
            for split in ("train", "dev", "test"):
                suffix = split_doc_suffix[split]
                pos_rows = [
                    _make_row(
                        family=family,
                        proxy_name=proxy_name,
                        proxy_slug=proxy_slug,
                        split=split,
                        doc_id=f"{proxy_slug}_{suffix}_p{i}",
                        segment_id=f"{proxy_slug}_{suffix}_p{i}",
                    )
                    for i in range(2)
                ]
                neg_rows = [
                    _make_row(
                        family=family,
                        proxy_name=f"negative_for_{proxy_name}",
                        proxy_slug=proxy_slug,
                        split=split,
                        doc_id=f"{proxy_slug}_{suffix}_n{i}",
                        segment_id=f"{proxy_slug}_{suffix}_n{i}",
                    )
                    for i in range(2)
                ]
                write_jsonl(proxy_dir / f"{split}.jsonl", pos_rows)
                write_jsonl(neg_dir / f"{split}.jsonl", neg_rows)
                write_jsonl(validated_dir / f"{split}.jsonl", pos_rows[:1])
                write_jsonl(validated_neg_dir / f"{split}.jsonl", neg_rows[:1])
                diagnostics[split] = {
                    "n_positives": len(pos_rows),
                    "n_candidates": 4,
                    "n_matched": len(neg_rows),
                    "encoding_method_used": "sentence",
                }
            (neg_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
            (validated_neg_dir / "diagnostics.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def _make_stub_runner(method_name: str, *, selected_layer: int | None, causality_enabled: bool):
    def _runner(config: dict, task_registry: dict, output_root: Path) -> dict:
        task_ids = [task["task_id"] for task in task_registry["coverage_tasks"]]
        family_by_task = {task["task_id"]: task["family"] for task in task_registry["coverage_tasks"]}
        result = {
            "benchmark_name": config["benchmark_name"],
            "benchmark_id": config["benchmark_id"],
            "method_name": method_name,
            "coverage": {},
            "consistency": {},
            "cross_family_controls": {},
            "causality": {},
        }
        for index, task in enumerate(task_registry["coverage_tasks"]):
            coverage_auc = 0.70 + (index * 0.01)
            result["coverage"][task["task_id"]] = {
                "task_id": task["task_id"],
                "family": task["family"],
                "proxy_name": task["proxy_name"],
                "coverage_auc": coverage_auc,
                "masked_coverage_auc": coverage_auc * 0.9,
                "validated_coverage_auc": coverage_auc * 0.95,
                "n_positive_train": task["counts"]["train_positive"],
                "n_negative_train": task["counts"]["train_negative"],
                "n_positive_test": task["counts"]["test_positive"],
                "n_negative_test": task["counts"]["test_negative"],
                "n_positive_validated_test": task["counts"]["validated_test_positive"],
                "n_negative_validated_test": task["counts"]["validated_test_negative"],
                "selected_layer": selected_layer,
                "site": "resid_post" if selected_layer is not None else None,
                "pooling": "mean" if selected_layer is not None else None,
                "robustness_pooling": "max" if selected_layer is not None else None,
                "selection_metric": 0.8 if selected_layer is not None else None,
                "selection_source": "inner_train_valid" if selected_layer is not None else "none",
                "mask_keywords": task["mask_keywords"],
                "feature_count": 32 if selected_layer is not None else None,
                "evaluation_segment_ids": [f"{task['task_id']}_eval_0", f"{task['task_id']}_eval_1"],
                "masked_evaluation_segment_ids": [f"{task['task_id']}_eval_0", f"{task['task_id']}_eval_1"],
            }
        for directed in task_registry["consistency_tasks"]:
            result["consistency"][directed["task_id"]] = {
                "source_task_id": directed["source_task_id"],
                "target_task_id": directed["target_task_id"],
                "family_relation": "within_family",
                "transfer_auc": 0.72,
                "selected_layer": selected_layer,
            }
        for source_task_id in task_ids:
            cross_targets = [
                target_task_id
                for target_task_id in task_ids
                if target_task_id != source_task_id and family_by_task[target_task_id] != family_by_task[source_task_id]
            ]
            result["cross_family_controls"][source_task_id] = {
                "source_task_id": source_task_id,
                "target_task_ids": cross_targets,
                "target_aucs": {target_task_id: 0.55 for target_task_id in cross_targets},
                "mean_cross_family_auc": 0.55,
            }
        for family_name in config["causality"]["families"]:
            if causality_enabled:
                result["causality"][family_name] = {
                    "status": "ok",
                    "layer": selected_layer,
                    "site": "resid_post",
                    "n_core_features": 3,
                    "causality_score": 0.2,
                    "details_path": f"{family_name}.json",
                }
            else:
                result["causality"][family_name] = {
                    "status": "na",
                    "layer": None,
                    "site": None,
                    "n_core_features": None,
                    "causality_score": None,
                    "details_path": None,
                }
        return result

    return _runner


class PolicyFeatureBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.manifest_root = self.base / "manifests"
        self.output_root = self.base / "results"
        config = yaml.safe_load((ROOT / "configs" / "policy_feature_benchmark.yaml").read_text(encoding="utf-8"))
        config["manifest_root"] = str(self.manifest_root)
        config["output_root"] = str(self.output_root)
        config["policy_config"] = str(ROOT / "configs" / "policy_mech_interp.yaml")
        config["manifest_summary_path"] = str(self.base / "manifest_summary.json")
        config["preflight"]["reference_split_counts"] = {"train": 12, "dev": 12, "test": 12}
        config["preflight"]["reference_validated_split_counts"] = {"train": 6, "dev": 6, "test": 6}
        config["preflight"]["reference_test_positive_counts"] = {
            "risk_factors_bias": 2,
            "harms_discrimination": 2,
            "risk_factors_privacy": 2,
            "harms_violation_of_civil_or_human_rights_including_privacy": 2,
            "risk_factors_transparency": 2,
            "risk_factors_interpretability_and_explainability": 2,
        }
        self.config_path = self.base / "policy_feature_benchmark.yaml"
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        _build_fixture(self.manifest_root, config)
        manifest_summary = {
            "summary": {
                "split_counts": {"train": 12, "dev": 12, "test": 12},
                "validated_split_counts": {"train": 6, "dev": 6, "test": 6},
            }
        }
        (self.base / "manifest_summary.json").write_text(json.dumps(manifest_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_task_registry_has_expected_v1_tasks(self):
        config = load_benchmark_config(self.config_path)
        registry = build_task_registry(config)
        self.assertEqual(len(registry["coverage_tasks"]), 6)
        self.assertEqual(len(registry["consistency_tasks"]), 6)
        self.assertEqual(
            sorted(task["task_id"] for task in registry["coverage_tasks"]),
            [
                "harms_discrimination",
                "harms_violation_of_civil_or_human_rights_including_privacy",
                "risk_factors_bias",
                "risk_factors_interpretability_and_explainability",
                "risk_factors_privacy",
                "risk_factors_transparency",
            ],
        )

    def test_no_document_leakage(self):
        config = load_benchmark_config(self.config_path)
        registry = build_task_registry(config)
        assert_no_leakage(registry)

    def test_run_preflight_writes_report(self):
        outputs = run_preflight(self.config_path)
        self.assertTrue(Path(outputs["preflight_report_path"]).exists())
        self.assertFalse(outputs["hard_fail"])
        self.assertTrue(outputs["headline_ready"])

    def test_run_benchmark_smoke_with_stub_runners(self):
        runner_factories = {
            "lexical_tfidf_logreg": _make_stub_runner("lexical_tfidf_logreg", selected_layer=None, causality_enabled=False),
            "semantic_sentence_embed_logreg": _make_stub_runner("semantic_sentence_embed_logreg", selected_layer=None, causality_enabled=False),
            "dense_residual_logreg": _make_stub_runner("dense_residual_logreg", selected_layer=16, causality_enabled=False),
            "sparse_sae_feature_bank": _make_stub_runner("sparse_sae_feature_bank", selected_layer=20, causality_enabled=True),
        }
        outputs = run_benchmark(self.config_path, runner_factories=runner_factories)
        for path_key in (
            "task_registry_path",
            "preflight_report_path",
            "core_leaderboard_path",
            "main_table_path",
            "mechanistic_qualification_path",
            "coverage_task_summary_path",
            "consistency_summary_path",
            "robustness_summary_path",
            "sae_causality_summary_path",
            "paper_readout_path",
            "benchmark_report_path",
        ):
            self.assertTrue(Path(outputs[path_key]).exists())
        for method_name in runner_factories:
            self.assertTrue((self.output_root / "method_results" / f"{method_name}.json").exists())

    def test_aggregation_excludes_na_causality_from_core(self):
        runner_factories = {
            "lexical_tfidf_logreg": _make_stub_runner("lexical_tfidf_logreg", selected_layer=None, causality_enabled=False),
            "semantic_sentence_embed_logreg": _make_stub_runner("semantic_sentence_embed_logreg", selected_layer=None, causality_enabled=False),
            "dense_residual_logreg": _make_stub_runner("dense_residual_logreg", selected_layer=12, causality_enabled=False),
            "sparse_sae_feature_bank": _make_stub_runner("sparse_sae_feature_bank", selected_layer=24, causality_enabled=True),
        }
        run_benchmark(self.config_path, runner_factories=runner_factories)
        aggregate_existing_results(self.config_path)
        with (self.output_root / "summary" / "core_leaderboard.csv").open("r", encoding="utf-8") as handle:
            leaderboard_rows = list(csv.DictReader(handle))
        self.assertEqual(len(leaderboard_rows), 4)
        with (self.output_root / "summary" / "mechanistic_qualification.csv").open("r", encoding="utf-8") as handle:
            mech_rows = list(csv.DictReader(handle))
        lexical_row = next(row for row in mech_rows if row["method_name"] == "lexical_tfidf_logreg")
        sae_row = next(row for row in mech_rows if row["method_name"] == "sparse_sae_feature_bank")
        self.assertEqual(lexical_row["status"], "NA")
        self.assertEqual(sae_row["status"], "ok")
        self.assertTrue((self.output_root / "summary" / "main_table.csv").exists())
        self.assertTrue((self.output_root / "summary" / "paper_readout.json").exists())

    def test_cross_family_averaging_is_deterministic(self):
        runner_factories = {
            "lexical_tfidf_logreg": _make_stub_runner("lexical_tfidf_logreg", selected_layer=None, causality_enabled=False),
            "semantic_sentence_embed_logreg": _make_stub_runner("semantic_sentence_embed_logreg", selected_layer=None, causality_enabled=False),
            "dense_residual_logreg": _make_stub_runner("dense_residual_logreg", selected_layer=16, causality_enabled=False),
            "sparse_sae_feature_bank": _make_stub_runner("sparse_sae_feature_bank", selected_layer=20, causality_enabled=True),
        }
        run_benchmark(self.config_path, runner_factories=runner_factories)
        config = load_benchmark_config(self.config_path)
        registry = build_task_registry(config)
        lexical_result = json.loads((self.output_root / "method_results" / "lexical_tfidf_logreg.json").read_text(encoding="utf-8"))
        summary = summarize_method_result(config, registry, lexical_result)
        self.assertAlmostEqual(summary["ConsistencyScore"], 0.17, places=6)

    def test_masked_runs_reuse_same_eval_rows(self):
        runner_factories = {
            "lexical_tfidf_logreg": _make_stub_runner("lexical_tfidf_logreg", selected_layer=None, causality_enabled=False),
            "semantic_sentence_embed_logreg": _make_stub_runner("semantic_sentence_embed_logreg", selected_layer=None, causality_enabled=False),
            "dense_residual_logreg": _make_stub_runner("dense_residual_logreg", selected_layer=16, causality_enabled=False),
            "sparse_sae_feature_bank": _make_stub_runner("sparse_sae_feature_bank", selected_layer=20, causality_enabled=True),
        }
        run_benchmark(self.config_path, runner_factories=runner_factories)
        dense_result = json.loads((self.output_root / "method_results" / "dense_residual_logreg.json").read_text(encoding="utf-8"))
        for task_payload in dense_result["coverage"].values():
            self.assertEqual(task_payload["evaluation_segment_ids"], task_payload["masked_evaluation_segment_ids"])

    def test_report_reproduces_counts_and_selected_layers(self):
        runner_factories = {
            "lexical_tfidf_logreg": _make_stub_runner("lexical_tfidf_logreg", selected_layer=None, causality_enabled=False),
            "semantic_sentence_embed_logreg": _make_stub_runner("semantic_sentence_embed_logreg", selected_layer=None, causality_enabled=False),
            "dense_residual_logreg": _make_stub_runner("dense_residual_logreg", selected_layer=16, causality_enabled=False),
            "sparse_sae_feature_bank": _make_stub_runner("sparse_sae_feature_bank", selected_layer=20, causality_enabled=True),
        }
        run_benchmark(self.config_path, runner_factories=runner_factories)
        report = json.loads((self.output_root / "summary" / "benchmark_report.json").read_text(encoding="utf-8"))
        task = report["coverage_tasks"]["risk_factors_bias"]
        self.assertEqual(task["counts"]["train_positive"], 2)
        dense_method = report["method_results"]["dense_residual_logreg"]
        self.assertEqual(dense_method["coverage"]["risk_factors_bias"]["selected_layer"], 16)
        self.assertTrue(report["preflight"]["headline_ready"])
        self.assertIn("paper_readout", report)


if __name__ == "__main__":
    unittest.main()
