from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from analysis.cluster_stats import cluster_bootstrap_selection_frequency, cluster_permutation_pvalues
from analysis.fdr import benjamini_hochberg
from analysis.metrics import quick_roc_auc
from analysis.polarization import polarization_table
from data.io import read_jsonl
from data.matching import (
    fit_tfidf_logistic,
    masked_texts,
    score_tfidf_logistic,
)
from data.split import split_doc_ids
from model.hooks import capture_layer_site_sequence, pool_sequence_activations
from model.load_model import load_model_bundle, resolve_device
from model.prompt import build_prompts, load_prompt_config
from runtime import ensure_dir, save_json
from sae.encode import encode_features
from sae.load_sae import load_sae_for_layer


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"


def _nanmean(values: list[float]) -> float:
    filtered = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(filtered)) if filtered else float("nan")


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0 or labels.shape[0] != scores.shape[0]:
        return float("nan")
    return quick_roc_auc(labels, scores)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logreg_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)


def _array_scores(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    margin = model.decision_function(features)
    return _sigmoid(np.asarray(margin, dtype=np.float64))


def _rows_to_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["text"]) for row in rows]


def _rows_to_doc_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["document_id"]) for row in rows]


def _rows_to_segment_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["segment_id"]) for row in rows]


def _mask_rows(rows: list[dict[str, Any]], keywords: list[str]) -> list[dict[str, Any]]:
    texts = masked_texts(_rows_to_texts(rows), keywords)
    return [{**row, "text": text} for row, text in zip(rows, texts)]


def _load_task_split_rows(
    task: dict[str, Any],
    split: str,
    *,
    validated: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive_paths = task["paths"]["validated_positive" if validated else "positive"]
    negative_paths = task["paths"]["validated_negative" if validated else "negative"]
    pos_path = Path(positive_paths[split])
    neg_path = Path(negative_paths[split])
    pos_rows = read_jsonl(pos_path) if pos_path.exists() else []
    neg_rows = read_jsonl(neg_path) if neg_path.exists() else []
    return pos_rows, neg_rows


def _fit_tfidf_train(train_pos: list[dict[str, Any]], train_neg: list[dict[str, Any]]):
    train_texts = _rows_to_texts(train_pos) + _rows_to_texts(train_neg)
    labels = [1] * len(train_pos) + [0] * len(train_neg)
    vectorizer, model = fit_tfidf_logistic(train_texts, labels)
    return vectorizer, model


class SentenceEmbeddingEncoder:
    def __init__(self, *, model_id: str, batch_size: int, device: str) -> None:
        self.model_id = model_id
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            with torch.no_grad():
                model_out = self.model(**enc)
                hidden = model_out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        matrix = np.concatenate(outputs, axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return matrix / norms


class InternalFeatureExtractor:
    def __init__(self, policy_config: dict[str, Any], *, site: str, use_sae: bool) -> None:
        self.policy_config = dict(policy_config)
        self.site = site
        self.use_sae = use_sae
        self.bundle = load_model_bundle(policy_config)
        self.model = self.bundle.model
        self.tokenizer = self.bundle.tokenizer
        self.device = self.bundle.device
        prompt_cfg = load_prompt_config(policy_config.get("prompt_config", ROOT / "configs" / "policy_text_prompt.yaml"))
        self.template = str(prompt_cfg["template_v1"])
        self.use_chat_template = bool(policy_config.get("use_chat_template", False))
        self.batch_size = int(policy_config.get("batch_size", 4))
        self.max_length = int(policy_config.get("max_length", 1024))
        self._sae_cache: dict[int, Any] = {}
        self._feature_cache: dict[tuple[str, ...], np.ndarray] = {}

    def _sae(self, layer: int):
        if layer not in self._sae_cache:
            self._sae_cache[layer] = load_sae_for_layer(
                self.policy_config,
                layer=layer,
                site=self.site,
                device=self.device,
            )
        return self._sae_cache[layer]

    def extract(self, rows: list[dict[str, Any]], *, layer: int, pooling: str) -> np.ndarray:
        row_ids = tuple(_rows_to_segment_ids(rows))
        cache_key = (
            "sae" if self.use_sae else "dense",
            str(layer),
            pooling,
            *row_ids,
        )
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        prompts = build_prompts(
            rows,
            self.template,
            tokenizer=self.tokenizer,
            use_chat_template=self.use_chat_template,
        )
        outputs: list[np.ndarray] = []
        sae = self._sae(layer) if self.use_sae else None
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            with torch.no_grad():
                seq = capture_layer_site_sequence(self.model, layer=layer, site=self.site, inputs=enc)
                pooled = pool_sequence_activations(seq, attention_mask=enc["attention_mask"], pooling=pooling)
                if sae is not None:
                    pooled = encode_features(sae, pooled.to(torch.float32))
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        matrix = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)
        self._feature_cache[cache_key] = matrix
        return matrix


def _split_rows_by_docs(
    pos_rows: list[dict[str, Any]],
    neg_rows: list[dict[str, Any]],
    *,
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    doc_ids = sorted({*map(str, _rows_to_doc_ids(pos_rows)), *map(str, _rows_to_doc_ids(neg_rows))})
    if len(doc_ids) < 2 or valid_ratio <= 0.0:
        return pos_rows, neg_rows, [], [], "train_fallback"
    split = split_doc_ids(
        doc_ids,
        train_ratio=1.0 - valid_ratio,
        dev_ratio=valid_ratio,
        test_ratio=0.0,
        seed=seed,
    )
    if not split.train or not split.dev:
        return pos_rows, neg_rows, [], [], "train_fallback"
    train_doc_ids = set(split.train)
    valid_doc_ids = set(split.dev)
    pos_inner_train = [row for row in pos_rows if str(row["document_id"]) in train_doc_ids]
    neg_inner_train = [row for row in neg_rows if str(row["document_id"]) in train_doc_ids]
    pos_inner_valid = [row for row in pos_rows if str(row["document_id"]) in valid_doc_ids]
    neg_inner_valid = [row for row in neg_rows if str(row["document_id"]) in valid_doc_ids]
    if not pos_inner_train or not neg_inner_train or not pos_inner_valid or not neg_inner_valid:
        return pos_rows, neg_rows, [], [], "train_fallback"
    return pos_inner_train, neg_inner_train, pos_inner_valid, neg_inner_valid, "inner_train_valid"


def _evaluate_transfer_scores(
    *,
    task_ids: list[str],
    paired_targets: dict[str, str],
    score_target,
    family_by_task: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    consistency: dict[str, Any] = {}
    cross_controls: dict[str, Any] = {}
    for source_task_id in task_ids:
        source_family = family_by_task[source_task_id]
        cross_candidates: list[tuple[str, float]] = []
        for target_task_id in task_ids:
            if target_task_id == source_task_id:
                continue
            transfer_auc = score_target(source_task_id, target_task_id)
            if math.isnan(transfer_auc):
                continue
            relation = "within_family" if family_by_task[target_task_id] == source_family else "cross_family"
            if relation == "cross_family":
                cross_candidates.append((target_task_id, transfer_auc))
            elif paired_targets.get(source_task_id) == target_task_id:
                consistency[f"{source_task_id}__to__{target_task_id}"] = {
                    "source_task_id": source_task_id,
                    "target_task_id": target_task_id,
                    "family_relation": relation,
                    "transfer_auc": float(transfer_auc),
                }
        cross_controls[source_task_id] = {
            "source_task_id": source_task_id,
            "target_task_ids": [task_id for task_id, _ in cross_candidates],
            "target_aucs": {task_id: float(auc) for task_id, auc in cross_candidates},
            "mean_cross_family_auc": _nanmean([auc for _, auc in cross_candidates]),
        }
    return consistency, cross_controls


def _base_result(benchmark_config: dict[str, Any], method_name: str) -> dict[str, Any]:
    return {
        "benchmark_name": benchmark_config["benchmark_name"],
        "benchmark_id": benchmark_config["benchmark_id"],
        "method_name": method_name,
        "coverage": {},
        "consistency": {},
        "cross_family_controls": {},
        "causality": {},
    }


def run_lexical_tfidf_logreg(
    benchmark_config: dict[str, Any],
    task_registry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    result = _base_result(benchmark_config, "lexical_tfidf_logreg")
    task_ids = [task["task_id"] for task in task_registry["coverage_tasks"]]
    family_by_task = {task["task_id"]: task["family"] for task in task_registry["coverage_tasks"]}
    paired_targets = {task["task_id"]: task["paired_target_task_id"] for task in task_registry["coverage_tasks"]}
    fitted_models: dict[str, tuple[Any, Any]] = {}

    for task in task_registry["coverage_tasks"]:
        task_id = task["task_id"]
        keywords = list(task.get("mask_keywords", []))
        train_pos, train_neg = _load_task_split_rows(task, "train", validated=False)
        test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)
        validated_pos, validated_neg = _load_task_split_rows(task, "test", validated=True)
        vectorizer, model = _fit_tfidf_train(train_pos, train_neg)
        fitted_models[task_id] = (vectorizer, model)

        test_texts = _rows_to_texts(test_pos) + _rows_to_texts(test_neg)
        labels = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
        scores = score_tfidf_logistic(vectorizer, model, test_texts)
        coverage_auc = _safe_auc(labels, scores)

        masked_rows = _mask_rows(test_pos + test_neg, keywords)
        masked_scores = score_tfidf_logistic(vectorizer, model, _rows_to_texts(masked_rows))
        masked_auc = _safe_auc(labels, masked_scores)

        validated_auc = float("nan")
        if validated_pos and validated_neg:
            validated_labels = np.concatenate(
                [np.ones(len(validated_pos), dtype=np.int64), np.zeros(len(validated_neg), dtype=np.int64)]
            )
            validated_scores = score_tfidf_logistic(
                vectorizer,
                model,
                _rows_to_texts(validated_pos) + _rows_to_texts(validated_neg),
            )
            validated_auc = _safe_auc(validated_labels, validated_scores)

        result["coverage"][task_id] = {
            "task_id": task_id,
            "family": task["family"],
            "proxy_name": task["proxy_name"],
            "coverage_auc": float(coverage_auc),
            "masked_coverage_auc": float(masked_auc),
            "validated_coverage_auc": None if math.isnan(validated_auc) else float(validated_auc),
            "n_positive_train": len(train_pos),
            "n_negative_train": len(train_neg),
            "n_positive_test": len(test_pos),
            "n_negative_test": len(test_neg),
            "n_positive_validated_test": len(validated_pos),
            "n_negative_validated_test": len(validated_neg),
            "selected_layer": None,
            "site": None,
            "pooling": None,
            "robustness_pooling": None,
            "selection_metric": None,
            "selection_source": "none",
            "mask_keywords": keywords,
            "feature_count": None,
            "evaluation_segment_ids": _rows_to_segment_ids(test_pos + test_neg),
            "masked_evaluation_segment_ids": _rows_to_segment_ids(test_pos + test_neg),
        }

    def _score_target(source_task_id: str, target_task_id: str) -> float:
        vectorizer, model = fitted_models[source_task_id]
        target_task = task_registry["coverage_task_map"][target_task_id]
        pos_rows, neg_rows = _load_task_split_rows(target_task, "test", validated=False)
        labels = np.concatenate([np.ones(len(pos_rows), dtype=np.int64), np.zeros(len(neg_rows), dtype=np.int64)])
        texts = _rows_to_texts(pos_rows) + _rows_to_texts(neg_rows)
        scores = score_tfidf_logistic(vectorizer, model, texts)
        return _safe_auc(labels, scores)

    consistency, cross_controls = _evaluate_transfer_scores(
        task_ids=task_ids,
        paired_targets=paired_targets,
        score_target=_score_target,
        family_by_task=family_by_task,
    )
    result["consistency"] = consistency
    result["cross_family_controls"] = cross_controls
    for family_name in benchmark_config["causality"]["families"]:
        result["causality"][family_name] = {
            "status": "na",
            "layer": None,
            "site": None,
            "n_core_features": None,
            "causality_score": None,
            "details_path": None,
        }
    return result


def run_semantic_sentence_embed_logreg(
    benchmark_config: dict[str, Any],
    task_registry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    result = _base_result(benchmark_config, "semantic_sentence_embed_logreg")
    method_cfg = benchmark_config["methods"]["semantic_sentence_embed_logreg"]
    policy_cfg = benchmark_config["__policy_config"]
    encoder = SentenceEmbeddingEncoder(
        model_id=str(method_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        batch_size=int(method_cfg.get("batch_size", 32)),
        device=str(policy_cfg.get("device", "auto")),
    )
    task_ids = [task["task_id"] for task in task_registry["coverage_tasks"]]
    family_by_task = {task["task_id"]: task["family"] for task in task_registry["coverage_tasks"]}
    paired_targets = {task["task_id"]: task["paired_target_task_id"] for task in task_registry["coverage_tasks"]}
    fitted_models: dict[str, LogisticRegression] = {}

    for task in task_registry["coverage_tasks"]:
        task_id = task["task_id"]
        keywords = list(task.get("mask_keywords", []))
        train_pos, train_neg = _load_task_split_rows(task, "train", validated=False)
        test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)
        validated_pos, validated_neg = _load_task_split_rows(task, "test", validated=True)

        x_train = encoder.encode(_rows_to_texts(train_pos) + _rows_to_texts(train_neg))
        y_train = np.concatenate([np.ones(len(train_pos), dtype=np.int64), np.zeros(len(train_neg), dtype=np.int64)])
        model = _logreg_model()
        model.fit(x_train, y_train)
        fitted_models[task_id] = model

        x_test = encoder.encode(_rows_to_texts(test_pos) + _rows_to_texts(test_neg))
        y_test = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
        scores = _array_scores(model, x_test)
        coverage_auc = _safe_auc(y_test, scores)

        masked_rows = _mask_rows(test_pos + test_neg, keywords)
        masked_x = encoder.encode(_rows_to_texts(masked_rows))
        masked_scores = _array_scores(model, masked_x)
        masked_auc = _safe_auc(y_test, masked_scores)

        validated_auc = float("nan")
        if validated_pos and validated_neg:
            x_validated = encoder.encode(_rows_to_texts(validated_pos) + _rows_to_texts(validated_neg))
            y_validated = np.concatenate(
                [np.ones(len(validated_pos), dtype=np.int64), np.zeros(len(validated_neg), dtype=np.int64)]
            )
            validated_scores = _array_scores(model, x_validated)
            validated_auc = _safe_auc(y_validated, validated_scores)

        result["coverage"][task_id] = {
            "task_id": task_id,
            "family": task["family"],
            "proxy_name": task["proxy_name"],
            "coverage_auc": float(coverage_auc),
            "masked_coverage_auc": float(masked_auc),
            "validated_coverage_auc": None if math.isnan(validated_auc) else float(validated_auc),
            "n_positive_train": len(train_pos),
            "n_negative_train": len(train_neg),
            "n_positive_test": len(test_pos),
            "n_negative_test": len(test_neg),
            "n_positive_validated_test": len(validated_pos),
            "n_negative_validated_test": len(validated_neg),
            "selected_layer": None,
            "site": None,
            "pooling": None,
            "robustness_pooling": None,
            "selection_metric": None,
            "selection_source": "none",
            "mask_keywords": keywords,
            "feature_count": int(x_train.shape[1]) if x_train.ndim == 2 else None,
            "evaluation_segment_ids": _rows_to_segment_ids(test_pos + test_neg),
            "masked_evaluation_segment_ids": _rows_to_segment_ids(test_pos + test_neg),
        }

    def _score_target(source_task_id: str, target_task_id: str) -> float:
        model = fitted_models[source_task_id]
        target_task = task_registry["coverage_task_map"][target_task_id]
        pos_rows, neg_rows = _load_task_split_rows(target_task, "test", validated=False)
        x_test = encoder.encode(_rows_to_texts(pos_rows) + _rows_to_texts(neg_rows))
        y_test = np.concatenate([np.ones(len(pos_rows), dtype=np.int64), np.zeros(len(neg_rows), dtype=np.int64)])
        scores = _array_scores(model, x_test)
        return _safe_auc(y_test, scores)

    consistency, cross_controls = _evaluate_transfer_scores(
        task_ids=task_ids,
        paired_targets=paired_targets,
        score_target=_score_target,
        family_by_task=family_by_task,
    )
    result["consistency"] = consistency
    result["cross_family_controls"] = cross_controls
    for family_name in benchmark_config["causality"]["families"]:
        result["causality"][family_name] = {
            "status": "na",
            "layer": None,
            "site": None,
            "n_core_features": None,
            "causality_score": None,
            "details_path": None,
        }
    return result


def _dense_task_fit(
    extractor: InternalFeatureExtractor,
    *,
    task: dict[str, Any],
    candidate_layers: list[int],
    site: str,
    pooling: str,
    robustness_pooling: str,
    inner_valid_ratio: float,
    inner_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], LogisticRegression, LogisticRegression]:
    train_pos, train_neg = _load_task_split_rows(task, "train", validated=False)
    test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)
    validated_pos, validated_neg = _load_task_split_rows(task, "test", validated=True)

    best_layer = None
    best_metric = float("-inf")
    best_source = "train_fallback"
    for layer in candidate_layers:
        pos_inner_train, neg_inner_train, pos_inner_valid, neg_inner_valid, source = _split_rows_by_docs(
            train_pos,
            train_neg,
            valid_ratio=inner_valid_ratio,
            seed=inner_seed + int(layer),
        )
        train_features = np.concatenate(
            [
                extractor.extract(pos_inner_train, layer=layer, pooling=pooling),
                extractor.extract(neg_inner_train, layer=layer, pooling=pooling),
            ],
            axis=0,
        )
        train_labels = np.concatenate(
            [np.ones(len(pos_inner_train), dtype=np.int64), np.zeros(len(neg_inner_train), dtype=np.int64)]
        )
        model = _logreg_model()
        model.fit(train_features, train_labels)

        if source == "inner_train_valid":
            valid_features = np.concatenate(
                [
                    extractor.extract(pos_inner_valid, layer=layer, pooling=pooling),
                    extractor.extract(neg_inner_valid, layer=layer, pooling=pooling),
                ],
                axis=0,
            )
            valid_labels = np.concatenate(
                [np.ones(len(pos_inner_valid), dtype=np.int64), np.zeros(len(neg_inner_valid), dtype=np.int64)]
            )
            metric = _safe_auc(valid_labels, _array_scores(model, valid_features))
        else:
            metric = _safe_auc(train_labels, _array_scores(model, train_features))
        if math.isnan(metric):
            metric = float("-inf")
        if metric > best_metric:
            best_metric = metric
            best_layer = int(layer)
            best_source = source

    if best_layer is None:
        raise RuntimeError(f"Unable to select layer for task={task['task_id']}")

    x_train = np.concatenate(
        [
            extractor.extract(train_pos, layer=best_layer, pooling=pooling),
            extractor.extract(train_neg, layer=best_layer, pooling=pooling),
        ],
        axis=0,
    )
    y_train = np.concatenate([np.ones(len(train_pos), dtype=np.int64), np.zeros(len(train_neg), dtype=np.int64)])
    final_model = _logreg_model()
    final_model.fit(x_train, y_train)

    x_train_robust = np.concatenate(
        [
            extractor.extract(train_pos, layer=best_layer, pooling=robustness_pooling),
            extractor.extract(train_neg, layer=best_layer, pooling=robustness_pooling),
        ],
        axis=0,
    )
    robust_model = _logreg_model()
    robust_model.fit(x_train_robust, y_train)

    x_test = np.concatenate(
        [
            extractor.extract(test_pos, layer=best_layer, pooling=pooling),
            extractor.extract(test_neg, layer=best_layer, pooling=pooling),
        ],
        axis=0,
    )
    y_test = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
    coverage_auc = _safe_auc(y_test, _array_scores(final_model, x_test))

    validated_auc = float("nan")
    if validated_pos and validated_neg:
        x_validated = np.concatenate(
            [
                extractor.extract(validated_pos, layer=best_layer, pooling=pooling),
                extractor.extract(validated_neg, layer=best_layer, pooling=pooling),
            ],
            axis=0,
        )
        y_validated = np.concatenate(
            [np.ones(len(validated_pos), dtype=np.int64), np.zeros(len(validated_neg), dtype=np.int64)]
        )
        validated_auc = _safe_auc(y_validated, _array_scores(final_model, x_validated))

    return (
        {
            "selected_layer": best_layer,
            "selection_metric": None if best_metric == float("-inf") else float(best_metric),
            "selection_source": best_source,
            "feature_dim": int(x_train.shape[1]),
            "coverage_auc": float(coverage_auc),
            "validated_coverage_auc": None if math.isnan(validated_auc) else float(validated_auc),
        },
        {
            "train_pos": train_pos,
            "train_neg": train_neg,
            "test_pos": test_pos,
            "test_neg": test_neg,
            "validated_pos": validated_pos,
            "validated_neg": validated_neg,
        },
        final_model,
        robust_model,
    )


def run_dense_residual_logreg(
    benchmark_config: dict[str, Any],
    task_registry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    result = _base_result(benchmark_config, "dense_residual_logreg")
    method_cfg = benchmark_config["methods"]["dense_residual_logreg"]
    policy_cfg = benchmark_config["__policy_config"]
    site = str(method_cfg.get("site", "resid_post"))
    pooling = str(method_cfg.get("pooling", "mean"))
    robustness_pooling = str(method_cfg.get("robustness_pooling", "max"))
    extractor = InternalFeatureExtractor(policy_cfg, site=site, use_sae=False)
    task_ids = [task["task_id"] for task in task_registry["coverage_tasks"]]
    family_by_task = {task["task_id"]: task["family"] for task in task_registry["coverage_tasks"]}
    paired_targets = {task["task_id"]: task["paired_target_task_id"] for task in task_registry["coverage_tasks"]}
    models: dict[str, dict[str, Any]] = {}

    for index, task in enumerate(task_registry["coverage_tasks"]):
        summary, splits, final_model, robust_model = _dense_task_fit(
            extractor,
            task=task,
            candidate_layers=[int(v) for v in method_cfg.get("candidate_layers", [])],
            site=site,
            pooling=pooling,
            robustness_pooling=robustness_pooling,
            inner_valid_ratio=float(method_cfg.get("inner_valid_ratio", 0.2)),
            inner_seed=int(method_cfg.get("inner_seed", 13)) + index,
        )
        selected_layer = int(summary["selected_layer"])
        test_rows = splits["test_pos"] + splits["test_neg"]
        masked_test_rows = _mask_rows(test_rows, list(task.get("mask_keywords", [])))
        masked_pos = masked_test_rows[: len(splits["test_pos"])]
        masked_neg = masked_test_rows[len(splits["test_pos"]) :]
        x_masked = np.concatenate(
            [
                extractor.extract(masked_pos, layer=selected_layer, pooling=robustness_pooling),
                extractor.extract(masked_neg, layer=selected_layer, pooling=robustness_pooling),
            ],
            axis=0,
        )
        y_test = np.concatenate(
            [np.ones(len(splits["test_pos"]), dtype=np.int64), np.zeros(len(splits["test_neg"]), dtype=np.int64)]
        )
        masked_auc = _safe_auc(y_test, _array_scores(robust_model, x_masked))

        result["coverage"][task["task_id"]] = {
            "task_id": task["task_id"],
            "family": task["family"],
            "proxy_name": task["proxy_name"],
            "coverage_auc": float(summary["coverage_auc"]),
            "masked_coverage_auc": float(masked_auc),
            "validated_coverage_auc": summary["validated_coverage_auc"],
            "n_positive_train": len(splits["train_pos"]),
            "n_negative_train": len(splits["train_neg"]),
            "n_positive_test": len(splits["test_pos"]),
            "n_negative_test": len(splits["test_neg"]),
            "n_positive_validated_test": len(splits["validated_pos"]),
            "n_negative_validated_test": len(splits["validated_neg"]),
            "selected_layer": selected_layer,
            "site": site,
            "pooling": pooling,
            "robustness_pooling": robustness_pooling,
            "selection_metric": summary["selection_metric"],
            "selection_source": summary["selection_source"],
            "mask_keywords": list(task.get("mask_keywords", [])),
            "feature_count": int(summary["feature_dim"]),
            "evaluation_segment_ids": _rows_to_segment_ids(test_rows),
            "masked_evaluation_segment_ids": _rows_to_segment_ids(test_rows),
        }
        models[task["task_id"]] = {
            "model": final_model,
            "selected_layer": selected_layer,
        }

    def _score_target(source_task_id: str, target_task_id: str) -> float:
        fitted = models[source_task_id]
        target_task = task_registry["coverage_task_map"][target_task_id]
        pos_rows, neg_rows = _load_task_split_rows(target_task, "test", validated=False)
        x_target = np.concatenate(
            [
                extractor.extract(pos_rows, layer=fitted["selected_layer"], pooling=pooling),
                extractor.extract(neg_rows, layer=fitted["selected_layer"], pooling=pooling),
            ],
            axis=0,
        )
        y_target = np.concatenate([np.ones(len(pos_rows), dtype=np.int64), np.zeros(len(neg_rows), dtype=np.int64)])
        return _safe_auc(y_target, _array_scores(fitted["model"], x_target))

    consistency, cross_controls = _evaluate_transfer_scores(
        task_ids=task_ids,
        paired_targets=paired_targets,
        score_target=_score_target,
        family_by_task=family_by_task,
    )
    for key, payload in consistency.items():
        payload["selected_layer"] = models[payload["source_task_id"]]["selected_layer"]
    result["consistency"] = consistency
    result["cross_family_controls"] = cross_controls
    for family_name in benchmark_config["causality"]["families"]:
        result["causality"][family_name] = {
            "status": "na",
            "layer": None,
            "site": None,
            "n_core_features": None,
            "causality_score": None,
            "details_path": None,
        }
    return result


def _weighted_score(features: np.ndarray, feature_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if feature_ids.size == 0:
        return np.zeros(features.shape[0], dtype=np.float64)
    return features[:, feature_ids] @ weights


def _fit_sae_feature_bank(
    *,
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    pos_doc_ids: list[str],
    neg_doc_ids: list[str],
    topk: int,
    perm_n: int,
    bootstrap_b: int,
    fdr_q: float,
    seed: int,
) -> dict[str, Any]:
    train_df = polarization_table(features_pos, features_neg, perm_N=0, seed=seed)
    p_values = cluster_permutation_pvalues(
        features_pos,
        features_neg,
        pos_cluster_ids=np.asarray(pos_doc_ids, dtype=object),
        neg_cluster_ids=np.asarray(neg_doc_ids, dtype=object),
        n_perm=perm_n,
        seed=seed,
    )
    train_df["p_value"] = p_values
    fdr = benjamini_hochberg(train_df["p_value"].values, q=fdr_q)
    train_df["q_value"] = fdr["q_values"]
    train_df["fdr_reject"] = fdr["reject"]
    positive_candidates = train_df[train_df["delta"] > 0].copy()
    if positive_candidates["fdr_reject"].any():
        positive_candidates = positive_candidates[positive_candidates["fdr_reject"]]
    if positive_candidates.empty:
        positive_candidates = train_df[train_df["delta"] > 0].copy()
    selected = positive_candidates.sort_values("delta", ascending=False).head(topk)
    feature_ids = selected["feature_id"].astype(int).to_numpy()
    raw_weights = selected["delta"].to_numpy(dtype=np.float64)
    denom = np.abs(raw_weights).sum()
    weights = raw_weights / denom if denom > 0 else np.zeros_like(raw_weights)
    stability = cluster_bootstrap_selection_frequency(
        features_pos,
        features_neg,
        pos_cluster_ids=np.asarray(pos_doc_ids, dtype=object),
        neg_cluster_ids=np.asarray(neg_doc_ids, dtype=object),
        topk=topk,
        n_boot=bootstrap_b,
        seed=seed + 1,
    )
    stability_map = {int(fid): float(stability[int(fid)]) for fid in feature_ids.tolist()}
    return {
        "feature_ids": feature_ids,
        "feature_weights": weights,
        "raw_weights": raw_weights,
        "selected_q_values": selected["q_value"].tolist(),
        "bootstrap_stability": stability_map,
        "n_fdr_survivors": int(train_df["fdr_reject"].sum()),
        "train_delta": train_df,
    }


def _write_sae_intermediate(
    *,
    intermediate_root: Path,
    family: str,
    proxy_slug: str,
    proxy_name: str,
    layer: int,
    site: str,
    pooling: str,
    feature_bank: dict[str, Any],
    eval_auc: float,
    train_counts: tuple[int, int],
    eval_counts: tuple[int, int],
) -> Path:
    run_dir = ensure_dir(intermediate_root / "policy_discovery" / family / proxy_slug / f"layer{layer}_{site}")
    bank_payload = {
        "family_name": family,
        "proxy_slug": proxy_slug,
        "proxy_name": proxy_name,
        "layer": int(layer),
        "site": site,
        "pooling": pooling,
        "discovery_split": "train",
        "evaluation_split": "test",
        "feature_ids": feature_bank["feature_ids"].astype(int).tolist(),
        "feature_weights": feature_bank["feature_weights"].tolist(),
        "train_top_deltas": feature_bank["raw_weights"].tolist(),
        "train_q_values": feature_bank["selected_q_values"],
        "bootstrap_stability": {str(key): value for key, value in feature_bank["bootstrap_stability"].items()},
    }
    (run_dir / "feature_bank.json").write_text(json.dumps(bank_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "bootstrap_stability.json").write_text(
        json.dumps({"feature_stability": bank_payload["bootstrap_stability"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_json(
        run_dir / "eval_summary.json",
        {
            "feature_bank_path": str(run_dir / "feature_bank.json"),
            "eval_auc": float(eval_auc),
            "n_positive_train": int(train_counts[0]),
            "n_negative_train": int(train_counts[1]),
            "n_positive_eval": int(eval_counts[0]),
            "n_negative_eval": int(eval_counts[1]),
            "n_selected_features": int(feature_bank["feature_ids"].size),
            "n_fdr_survivors": int(feature_bank["n_fdr_survivors"]),
        },
    )
    return run_dir


def _prepare_runtime_policy_config(
    benchmark_config: dict[str, Any],
    *,
    results_dir: Path,
    manifest_root: Path,
) -> Path:
    runtime_dir = ensure_dir(Path(benchmark_config["output_root"]) / "_runtime")
    runtime_config_path = runtime_dir / "policy_benchmark_runtime.yaml"
    runtime_cfg = dict(benchmark_config["__policy_config"])
    runtime_cfg["results_dir"] = str(results_dir)
    runtime_cfg["processed_dir"] = str(manifest_root)
    runtime_cfg["layers"] = [int(v) for v in benchmark_config["methods"]["sparse_sae_feature_bank"]["candidate_layers"]]
    runtime_cfg["sites"] = [str(benchmark_config["methods"]["sparse_sae_feature_bank"]["site"])]
    runtime_cfg["pooling"] = str(benchmark_config["methods"]["sparse_sae_feature_bank"]["pooling"])
    runtime_cfg["robustness_pooling"] = str(benchmark_config["methods"]["sparse_sae_feature_bank"]["robustness_pooling"])
    runtime_cfg["topk"] = int(benchmark_config["methods"]["sparse_sae_feature_bank"].get("topk", 64))
    runtime_cfg["perm_N"] = int(runtime_cfg.get("perm_N", 2000))
    runtime_cfg["bootstrap_B"] = int(runtime_cfg.get("bootstrap_B", 500))
    with runtime_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_cfg, handle, sort_keys=False)
    return runtime_config_path


def _run_subprocess(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def _run_sae_causality_suite(
    benchmark_config: dict[str, Any],
    intermediate_root: Path,
) -> dict[str, Any]:
    manifest_root = Path(benchmark_config["manifest_root"])
    runtime_config_path = _prepare_runtime_policy_config(
        benchmark_config,
        results_dir=intermediate_root,
        manifest_root=manifest_root,
    )
    site = str(benchmark_config["methods"]["sparse_sae_feature_bank"]["site"])
    candidate_layers = [int(v) for v in benchmark_config["methods"]["sparse_sae_feature_bank"]["candidate_layers"]]
    causality_cfg = benchmark_config["causality"]
    results: dict[str, Any] = {}

    for family_name in causality_cfg["families"]:
        best_layer = None
        best_core_count = -1
        best_core_stability = float("-inf")
        best_core_path = None
        for layer in candidate_layers:
            _run_subprocess(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_family_core_bank.py"),
                    "--config",
                    str(runtime_config_path),
                    "--family",
                    family_name,
                    "--layer",
                    str(layer),
                    "--site",
                    site,
                    "--min_proxy_support",
                    str(int(causality_cfg.get("min_proxy_support", 2))),
                    "--min_bootstrap_stability",
                    str(float(causality_cfg.get("min_bootstrap_stability", 0.25))),
                ]
            )
            core_path = intermediate_root / "family_core" / family_name / f"layer{layer}_{site}" / "family_core_bank.json"
            if not core_path.exists():
                continue
            core_obj = json.loads(core_path.read_text(encoding="utf-8"))
            core_features = core_obj.get("core_features", [])
            if not core_features:
                continue
            core_count = len(core_features)
            mean_stability = _nanmean([row.get("mean_bootstrap_stability", float("nan")) for row in core_features])
            if core_count > best_core_count or (
                core_count == best_core_count
                and (mean_stability > best_core_stability or (math.isclose(mean_stability, best_core_stability) and (best_layer is None or layer < best_layer)))
            ):
                best_layer = layer
                best_core_count = core_count
                best_core_stability = mean_stability
                best_core_path = core_path

        if best_layer is None or best_core_path is None:
            results[family_name] = {
                "status": "skipped",
                "layer": None,
                "site": site,
                "n_core_features": 0,
                "causality_score": None,
                "details_path": None,
            }
            continue

        _run_subprocess(
            [
                sys.executable,
                str(SCRIPTS_DIR / "run_family_causal_eval.py"),
                "--config",
                str(runtime_config_path),
                "--manifest_root",
                str(manifest_root),
                "--family",
                family_name,
                "--layer",
                str(best_layer),
                "--site",
                site,
                "--split",
                str(causality_cfg.get("split", "test")),
                "--mode",
                str(causality_cfg.get("mode", "ablate")),
                "--alpha",
                str(float(causality_cfg.get("alpha", 0.2))),
                "--max_samples",
                str(int(causality_cfg.get("max_samples", 100))),
            ]
        )
        details_path = intermediate_root / "family_causal" / family_name / f"layer{best_layer}_{site}" / f"{causality_cfg.get('mode', 'ablate')}_{causality_cfg.get('split', 'test')}.json"
        if not details_path.exists():
            results[family_name] = {
                "status": "skipped",
                "layer": int(best_layer),
                "site": site,
                "n_core_features": int(best_core_count),
                "causality_score": None,
                "details_path": None,
            }
            continue
        details = json.loads(details_path.read_text(encoding="utf-8"))
        causality_score = None
        if not details.get("skipped"):
            mean_margin_delta = float(details.get("mean_margin_delta", float("nan")))
            mean_random_margin_delta = float(details.get("mean_random_margin_delta", float("nan")))
            if not math.isnan(mean_margin_delta) and not math.isnan(mean_random_margin_delta):
                causality_score = float(mean_random_margin_delta - mean_margin_delta)
        results[family_name] = {
            "status": "ok" if causality_score is not None else "skipped",
            "layer": int(best_layer),
            "site": site,
            "n_core_features": int(best_core_count),
            "causality_score": causality_score,
            "details_path": str(details_path),
            "family_core_path": str(best_core_path),
        }
    return results


def run_sparse_sae_feature_bank(
    benchmark_config: dict[str, Any],
    task_registry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    result = _base_result(benchmark_config, "sparse_sae_feature_bank")
    method_cfg = benchmark_config["methods"]["sparse_sae_feature_bank"]
    policy_cfg = benchmark_config["__policy_config"]
    site = str(method_cfg.get("site", "resid_post"))
    pooling = str(method_cfg.get("pooling", "mean"))
    robustness_pooling = str(method_cfg.get("robustness_pooling", "max"))
    extractor = InternalFeatureExtractor(policy_cfg, site=site, use_sae=True)
    seed = int(policy_cfg.get("seed", 0))
    perm_n = int(policy_cfg.get("perm_N", 2000))
    bootstrap_b = int(policy_cfg.get("bootstrap_B", 500))
    fdr_q = float(policy_cfg.get("fdr_q", 0.05))
    topk = int(method_cfg.get("topk", policy_cfg.get("topk", 64)))
    task_ids = [task["task_id"] for task in task_registry["coverage_tasks"]]
    family_by_task = {task["task_id"]: task["family"] for task in task_registry["coverage_tasks"]}
    paired_targets = {task["task_id"]: task["paired_target_task_id"] for task in task_registry["coverage_tasks"]}
    fitted_banks: dict[str, dict[str, Any]] = {}
    intermediate_root = ensure_dir(output_root / "_intermediate")

    for index, task in enumerate(task_registry["coverage_tasks"]):
        train_pos, train_neg = _load_task_split_rows(task, "train", validated=False)
        test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)
        validated_pos, validated_neg = _load_task_split_rows(task, "test", validated=True)

        best_layer = None
        best_metric = float("-inf")
        best_source = "train_fallback"
        for layer in [int(v) for v in method_cfg.get("candidate_layers", [])]:
            pos_inner_train, neg_inner_train, pos_inner_valid, neg_inner_valid, source = _split_rows_by_docs(
                train_pos,
                train_neg,
                valid_ratio=float(method_cfg.get("inner_valid_ratio", 0.2)),
                seed=int(method_cfg.get("inner_seed", 13)) + index + layer,
            )
            train_pos_features = extractor.extract(pos_inner_train, layer=layer, pooling=pooling)
            train_neg_features = extractor.extract(neg_inner_train, layer=layer, pooling=pooling)
            bank = _fit_sae_feature_bank(
                features_pos=train_pos_features,
                features_neg=train_neg_features,
                pos_doc_ids=_rows_to_doc_ids(pos_inner_train),
                neg_doc_ids=_rows_to_doc_ids(neg_inner_train),
                topk=topk,
                perm_n=perm_n,
                bootstrap_b=bootstrap_b,
                fdr_q=fdr_q,
                seed=seed + index + layer,
            )
            if source == "inner_train_valid":
                valid_features = np.concatenate(
                    [
                        extractor.extract(pos_inner_valid, layer=layer, pooling=pooling),
                        extractor.extract(neg_inner_valid, layer=layer, pooling=pooling),
                    ],
                    axis=0,
                )
                valid_labels = np.concatenate(
                    [np.ones(len(pos_inner_valid), dtype=np.int64), np.zeros(len(neg_inner_valid), dtype=np.int64)]
                )
                scores = _weighted_score(valid_features, bank["feature_ids"], bank["feature_weights"])
                metric = _safe_auc(valid_labels, scores)
            else:
                train_features = np.concatenate([train_pos_features, train_neg_features], axis=0)
                train_labels = np.concatenate(
                    [np.ones(len(pos_inner_train), dtype=np.int64), np.zeros(len(neg_inner_train), dtype=np.int64)]
                )
                scores = _weighted_score(train_features, bank["feature_ids"], bank["feature_weights"])
                metric = _safe_auc(train_labels, scores)
            if math.isnan(metric):
                metric = float("-inf")
            if metric > best_metric:
                best_metric = metric
                best_layer = int(layer)
                best_source = source

        if best_layer is None:
            raise RuntimeError(f"Unable to select SAE layer for task={task['task_id']}")

        train_pos_features = extractor.extract(train_pos, layer=best_layer, pooling=pooling)
        train_neg_features = extractor.extract(train_neg, layer=best_layer, pooling=pooling)
        bank = _fit_sae_feature_bank(
            features_pos=train_pos_features,
            features_neg=train_neg_features,
            pos_doc_ids=_rows_to_doc_ids(train_pos),
            neg_doc_ids=_rows_to_doc_ids(train_neg),
            topk=topk,
            perm_n=perm_n,
            bootstrap_b=bootstrap_b,
            fdr_q=fdr_q,
            seed=seed + index + best_layer,
        )
        eval_features = np.concatenate(
            [
                extractor.extract(test_pos, layer=best_layer, pooling=pooling),
                extractor.extract(test_neg, layer=best_layer, pooling=pooling),
            ],
            axis=0,
        )
        y_test = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
        coverage_auc = _safe_auc(y_test, _weighted_score(eval_features, bank["feature_ids"], bank["feature_weights"]))
        _write_sae_intermediate(
            intermediate_root=intermediate_root,
            family=task["family"],
            proxy_slug=task["proxy_slug"],
            proxy_name=task["proxy_name"],
            layer=best_layer,
            site=site,
            pooling=pooling,
            feature_bank=bank,
            eval_auc=coverage_auc,
            train_counts=(len(train_pos), len(train_neg)),
            eval_counts=(len(test_pos), len(test_neg)),
        )

        masked_test_rows = _mask_rows(test_pos + test_neg, list(task.get("mask_keywords", [])))
        masked_pos = masked_test_rows[: len(test_pos)]
        masked_neg = masked_test_rows[len(test_pos) :]
        train_pos_robust = extractor.extract(train_pos, layer=best_layer, pooling=robustness_pooling)
        train_neg_robust = extractor.extract(train_neg, layer=best_layer, pooling=robustness_pooling)
        robust_bank = _fit_sae_feature_bank(
            features_pos=train_pos_robust,
            features_neg=train_neg_robust,
            pos_doc_ids=_rows_to_doc_ids(train_pos),
            neg_doc_ids=_rows_to_doc_ids(train_neg),
            topk=topk,
            perm_n=perm_n,
            bootstrap_b=bootstrap_b,
            fdr_q=fdr_q,
            seed=seed + index + best_layer + 1000,
        )
        masked_features = np.concatenate(
            [
                extractor.extract(masked_pos, layer=best_layer, pooling=robustness_pooling),
                extractor.extract(masked_neg, layer=best_layer, pooling=robustness_pooling),
            ],
            axis=0,
        )
        masked_auc = _safe_auc(y_test, _weighted_score(masked_features, robust_bank["feature_ids"], robust_bank["feature_weights"]))

        validated_auc = float("nan")
        if validated_pos and validated_neg:
            validated_features = np.concatenate(
                [
                    extractor.extract(validated_pos, layer=best_layer, pooling=pooling),
                    extractor.extract(validated_neg, layer=best_layer, pooling=pooling),
                ],
                axis=0,
            )
            y_validated = np.concatenate(
                [np.ones(len(validated_pos), dtype=np.int64), np.zeros(len(validated_neg), dtype=np.int64)]
            )
            validated_auc = _safe_auc(
                y_validated,
                _weighted_score(validated_features, bank["feature_ids"], bank["feature_weights"]),
            )

        result["coverage"][task["task_id"]] = {
            "task_id": task["task_id"],
            "family": task["family"],
            "proxy_name": task["proxy_name"],
            "coverage_auc": float(coverage_auc),
            "masked_coverage_auc": float(masked_auc),
            "validated_coverage_auc": None if math.isnan(validated_auc) else float(validated_auc),
            "n_positive_train": len(train_pos),
            "n_negative_train": len(train_neg),
            "n_positive_test": len(test_pos),
            "n_negative_test": len(test_neg),
            "n_positive_validated_test": len(validated_pos),
            "n_negative_validated_test": len(validated_neg),
            "selected_layer": int(best_layer),
            "site": site,
            "pooling": pooling,
            "robustness_pooling": robustness_pooling,
            "selection_metric": None if best_metric == float("-inf") else float(best_metric),
            "selection_source": best_source,
            "mask_keywords": list(task.get("mask_keywords", [])),
            "feature_count": int(bank["feature_ids"].size),
            "evaluation_segment_ids": _rows_to_segment_ids(test_pos + test_neg),
            "masked_evaluation_segment_ids": _rows_to_segment_ids(test_pos + test_neg),
        }
        fitted_banks[task["task_id"]] = {
            "selected_layer": int(best_layer),
            "feature_ids": bank["feature_ids"],
            "feature_weights": bank["feature_weights"],
        }

    def _score_target(source_task_id: str, target_task_id: str) -> float:
        bank = fitted_banks[source_task_id]
        target_task = task_registry["coverage_task_map"][target_task_id]
        pos_rows, neg_rows = _load_task_split_rows(target_task, "test", validated=False)
        features = np.concatenate(
            [
                extractor.extract(pos_rows, layer=bank["selected_layer"], pooling=pooling),
                extractor.extract(neg_rows, layer=bank["selected_layer"], pooling=pooling),
            ],
            axis=0,
        )
        labels = np.concatenate([np.ones(len(pos_rows), dtype=np.int64), np.zeros(len(neg_rows), dtype=np.int64)])
        scores = _weighted_score(features, bank["feature_ids"], bank["feature_weights"])
        return _safe_auc(labels, scores)

    consistency, cross_controls = _evaluate_transfer_scores(
        task_ids=task_ids,
        paired_targets=paired_targets,
        score_target=_score_target,
        family_by_task=family_by_task,
    )
    for key, payload in consistency.items():
        payload["selected_layer"] = fitted_banks[payload["source_task_id"]]["selected_layer"]
    result["consistency"] = consistency
    result["cross_family_controls"] = cross_controls
    result["causality"] = _run_sae_causality_suite(benchmark_config, intermediate_root)
    return result


RUNNER_FACTORIES = {
    "lexical_tfidf_logreg": run_lexical_tfidf_logreg,
    "semantic_sentence_embed_logreg": run_semantic_sentence_embed_logreg,
    "dense_residual_logreg": run_dense_residual_logreg,
    "sparse_sae_feature_bank": run_sparse_sae_feature_bank,
}
