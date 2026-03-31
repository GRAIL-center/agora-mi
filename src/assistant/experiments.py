from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analysis.metrics import (
    average_precision,
    cosine_similarity,
    first_relevant_rank_binary,
    ndcg_at_k_binary,
    precision_at_k_binary,
    quick_roc_auc,
    recall_at_k_binary,
    reciprocal_rank_from_order,
)
from assistant.documents import normalize_document_input, segment_document
from assistant.render import build_document_summary_note, build_segment_note
from benchmark.finetuned_encoder import fit_finetuned_proxy_encoder, predict_proxy_scores
from benchmark.policy_feature_benchmark import build_task_registry, load_benchmark_config
from benchmark.methods import (
    InternalFeatureExtractor,
    SentenceEmbeddingEncoder,
    _array_scores,
    _dense_task_fit,
    _fit_sae_feature_bank,
    _fit_tfidf_train,
    _load_task_split_rows,
    _logreg_model,
    _mask_rows,
    _rows_to_doc_ids,
    _rows_to_texts,
    _split_rows_by_docs,
    _weighted_score,
)
from data.io import read_jsonl
from runtime import ensure_dir, load_yaml, save_json


ROOT = Path(__file__).resolve().parents[2]
FAMILY_DISPLAY_NAMES = {
    "equality_neutrality": "Equality and neutrality",
    "individual_rights": "Individual rights",
    "transparency_accountability": "Transparency and accountability",
}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _nanmean(values: list[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value))) if not math.isnan(value) else float("nan")


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if math.isclose(lo, hi):
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _load_assistant_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["__config_path"] = str(_resolve_path(path))
    cfg["benchmark_config"] = str(_resolve_path(cfg["benchmark_config"]))
    cfg["output_root"] = str(_resolve_path(cfg["output_root"]))
    cfg["benchmark_output_root"] = str(_resolve_path(cfg["benchmark_output_root"]))
    cfg["__benchmark"] = load_benchmark_config(cfg["benchmark_config"])
    return cfg


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _causal_badge_from_row(row: dict[str, Any] | None) -> str:
    if not row:
        return "not_tested"
    if _parse_boolish(row.get("passes_positive_causality", False)):
        return "passed"
    status = str(row.get("status", "")).strip().lower()
    if status in {"ok", "tested", "completed"}:
        return "tested_not_passed"
    return "not_tested"


def _load_sparse_proxy_evidence(assistant_cfg: dict[str, Any]) -> dict[str, Any]:
    benchmark_output_root = Path(assistant_cfg["benchmark_output_root"])
    summary_root = benchmark_output_root / "summary"
    proxy_causal_rows = _read_csv_rows(summary_root / "proxy_causal_summary.csv")
    proxy_feature_rows = _read_csv_rows(summary_root / "proxy_feature_summary.csv")
    pair_rows = _read_csv_rows(summary_root / "pair_mechanistic_summary.csv")
    dossier_rows = read_jsonl(summary_root / "feature_dossiers.jsonl") if (summary_root / "feature_dossiers.jsonl").exists() else []
    if not proxy_causal_rows and not pair_rows:
        return {
            "status": "missing_benchmark_results",
            "benchmark_output_root": str(benchmark_output_root),
            "summary_root": str(summary_root),
            "proxy_causal_rows": [],
            "proxy_feature_rows": [],
            "pair_rows": [],
            "feature_dossier_rows": [],
            "proxy_causal_map": {},
            "proxy_feature_map": {},
            "proxy_causal_badges": {},
            "dossier_by_proxy_feature": {},
        }
    proxy_causal_map = {str(row.get("proxy_slug", "")): row for row in proxy_causal_rows if row.get("proxy_slug")}
    proxy_feature_map = {str(row.get("proxy_slug", "")): row for row in proxy_feature_rows if row.get("proxy_slug")}
    dossier_by_proxy_feature = {
        (str(row.get("proxy_slug", "")), int(row.get("feature_id", -1))): row
        for row in dossier_rows
        if row.get("proxy_slug") and str(row.get("feature_id", "")).strip()
    }
    return {
        "status": "ok",
        "benchmark_output_root": str(benchmark_output_root),
        "summary_root": str(summary_root),
        "proxy_causal_rows": proxy_causal_rows,
        "proxy_feature_rows": proxy_feature_rows,
        "pair_rows": pair_rows,
        "feature_dossier_rows": dossier_rows,
        "proxy_causal_map": proxy_causal_map,
        "proxy_feature_map": proxy_feature_map,
        "proxy_causal_badges": {
            proxy_slug: _causal_badge_from_row(row) for proxy_slug, row in proxy_causal_map.items()
        },
        "dossier_by_proxy_feature": dossier_by_proxy_feature,
    }


def _eligible_rows(manifest_root: Path, split: str) -> list[dict[str, Any]]:
    return read_jsonl(manifest_root / "eligible" / f"{split}.jsonl")


def _family_defs(benchmark_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    defs: list[dict[str, Any]] = []
    for pair in benchmark_cfg["v1_pairs"]:
        family = str(pair["family"])
        defs.append(
            {
                "family": family,
                "family_display_name": FAMILY_DISPLAY_NAMES.get(family, family.replace("_", " ").title()),
                "left_task_id": str(pair["left"]["proxy_slug"]),
                "right_task_id": str(pair["right"]["proxy_slug"]),
                "left_proxy_name": str(pair["left"]["proxy_name"]),
                "right_proxy_name": str(pair["right"]["proxy_name"]),
                "left_display_name": str(pair["left"]["display_name"]),
                "right_display_name": str(pair["right"]["display_name"]),
            }
        )
    return defs


def _labels_for_family(rows: list[dict[str, Any]], family_def: dict[str, Any]) -> np.ndarray:
    left = family_def["left_proxy_name"]
    right = family_def["right_proxy_name"]
    return np.asarray([1 if left in row.get("all_tags", []) or right in row.get("all_tags", []) else 0 for row in rows], dtype=np.int64)


def _group_row_indices_by_doc(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["document_id"]), []).append(index)
    return groups


def _mean_document_metric(rows: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray, metric_fn) -> float:
    values: list[float] = []
    for indices in _group_row_indices_by_doc(rows).values():
        y_doc = labels[indices]
        if int((y_doc == 1).sum()) == 0:
            continue
        value = metric_fn(y_doc, scores[indices])
        if not math.isnan(value):
            values.append(value)
    return _nanmean(values)


def _document_family_aggregation(cards: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_doc_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for card in cards:
        by_doc_family.setdefault((str(card["document_id"]), str(card["family"])), []).append(card)
    family_rows: list[dict[str, Any]] = []
    for (document_id, family), family_cards in by_doc_family.items():
        family_cards = sorted(family_cards, key=lambda row: (-float(row["priority_score"]), row["segment_id"]))
        top_cards = family_cards[:top_k]
        family_rows.append(
            {
                "document_id": document_id,
                "family": family,
                "family_display_name": top_cards[0]["family_display_name"],
                "document_family_score": float(np.mean([float(row["priority_score"]) for row in top_cards])),
                "top_segment_id": top_cards[0]["segment_id"],
                "top_priority_score": float(top_cards[0]["priority_score"]),
                "top_segments": [row["segment_id"] for row in top_cards],
            }
        )
    briefs: list[dict[str, Any]] = []
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in family_rows:
        by_doc.setdefault(str(row["document_id"]), []).append(row)
    for document_id, rows_for_doc in by_doc.items():
        rows_for_doc = sorted(rows_for_doc, key=lambda row: (-float(row["document_family_score"]), row["family"]))
        brief = {
            "document_id": document_id,
            "dominant_families": rows_for_doc,
            "top_segments_by_family": {row["family"]: row["top_segments"] for row in rows_for_doc},
            "family_scores": {row["family"]: float(row["document_family_score"]) for row in rows_for_doc},
            "review_priority_order": rows_for_doc,
        }
        brief["summary_note"] = build_document_summary_note(brief)
        briefs.append(brief)
    return family_rows, briefs


def _fit_lexical_artifact(task_registry: dict[str, Any], eligible_rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_scores: dict[str, np.ndarray] = {}
    task_scores_norm: dict[str, np.ndarray] = {}
    task_reliability: dict[str, float] = {}
    midpoint = max(1, len(eligible_rows) // 2)
    global_vectorizer, _ = _fit_tfidf_train(eligible_rows[:midpoint], eligible_rows[midpoint:])
    global_matrix = global_vectorizer.transform(_rows_to_texts(eligible_rows)).toarray().astype(np.float32)
    for task in task_registry["coverage_tasks"]:
        train_pos, train_neg = _load_task_split_rows(task, "train", validated=False)
        test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)
        vectorizer, model = _fit_tfidf_train(train_pos, train_neg)
        eligible_scores = model.predict_proba(vectorizer.transform(_rows_to_texts(eligible_rows)))[:, 1]
        task_scores[task["task_id"]] = eligible_scores.astype(np.float64)
        task_scores_norm[task["task_id"]] = _minmax_normalize(eligible_scores)
        y_test = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
        test_scores = model.predict_proba(vectorizer.transform(_rows_to_texts(test_pos) + _rows_to_texts(test_neg)))[:, 1]
        masked_test = _mask_rows(test_pos + test_neg, list(task.get("mask_keywords", [])))
        masked_scores = model.predict_proba(vectorizer.transform(_rows_to_texts(masked_test)))[:, 1]
        original_auc = quick_roc_auc(y_test, test_scores)
        masked_auc = quick_roc_auc(y_test, masked_scores)
        task_reliability[task["task_id"]] = _clip01(masked_auc / original_auc) if original_auc and not math.isnan(original_auc) else float("nan")
    return {
        "method_name": "lexical_tfidf_logreg",
        "task_scores": task_scores,
        "task_scores_norm": task_scores_norm,
        "task_reliability": task_reliability,
        "retrieval_matrix": global_matrix,
    }


def _fit_sentence_artifact(benchmark_cfg: dict[str, Any], task_registry: dict[str, Any], eligible_rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_cfg = benchmark_cfg["methods"]["semantic_sentence_embed_logreg"]
    encoder = SentenceEmbeddingEncoder(
        model_id=str(method_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        batch_size=int(method_cfg.get("batch_size", 32)),
        device=str(benchmark_cfg["__policy_config"].get("device", "auto")),
    )
    eligible_embeddings = encoder.encode(_rows_to_texts(eligible_rows))
    task_scores: dict[str, np.ndarray] = {}
    task_scores_norm: dict[str, np.ndarray] = {}
    task_reliability: dict[str, float] = {}
    for task in task_registry["coverage_tasks"]:
        train_pos, train_neg = _load_task_split_rows(task, "train", validated=False)
        test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)
        x_train = encoder.encode(_rows_to_texts(train_pos) + _rows_to_texts(train_neg))
        y_train = np.concatenate([np.ones(len(train_pos), dtype=np.int64), np.zeros(len(train_neg), dtype=np.int64)])
        model = _logreg_model()
        model.fit(x_train, y_train)
        eligible_scores = _array_scores(model, eligible_embeddings)
        task_scores[task["task_id"]] = eligible_scores.astype(np.float64)
        task_scores_norm[task["task_id"]] = _minmax_normalize(eligible_scores)
        x_test = encoder.encode(_rows_to_texts(test_pos) + _rows_to_texts(test_neg))
        y_test = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
        original_auc = quick_roc_auc(y_test, _array_scores(model, x_test))
        masked_test = _mask_rows(test_pos + test_neg, list(task.get("mask_keywords", [])))
        masked_auc = quick_roc_auc(y_test, _array_scores(model, encoder.encode(_rows_to_texts(masked_test))))
        task_reliability[task["task_id"]] = _clip01(masked_auc / original_auc) if original_auc and not math.isnan(original_auc) else float("nan")
    return {
        "method_name": "semantic_sentence_embed_logreg",
        "task_scores": task_scores,
        "task_scores_norm": task_scores_norm,
        "task_reliability": task_reliability,
        "retrieval_matrix": eligible_embeddings.astype(np.float32),
    }


def _fit_finetuned_encoder_artifact(
    benchmark_cfg: dict[str, Any],
    task_registry: dict[str, Any],
    eligible_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    method_cfg = benchmark_cfg["methods"]["finetuned_encoder_multilabel"]
    policy_cfg = benchmark_cfg["__policy_config"]
    manifest_root = Path(benchmark_cfg["manifest_root"])
    train_rows = read_jsonl(manifest_root / "eligible" / f"{benchmark_cfg['splits']['train']}.jsonl")
    dev_rows = read_jsonl(manifest_root / "eligible" / f"{benchmark_cfg['splits']['dev']}.jsonl")
    bundle = fit_finetuned_proxy_encoder(
        task_registry=task_registry,
        train_rows=train_rows,
        dev_rows=dev_rows,
        method_cfg=method_cfg,
        device_name=str(policy_cfg.get("device", "auto")),
        seed=int(policy_cfg.get("seed", 0)),
    )
    batch_size = int(method_cfg.get("batch_size", 8))
    eligible_probs, eligible_embeddings = predict_proxy_scores(bundle, eligible_rows, batch_size=batch_size)
    task_index = {task_id: index for index, task_id in enumerate(bundle.task_ids)}

    task_scores: dict[str, np.ndarray] = {}
    task_scores_norm: dict[str, np.ndarray] = {}
    task_reliability: dict[str, float] = {}
    for task in task_registry["coverage_tasks"]:
        task_id = str(task["task_id"])
        proxy_index = int(task_index[task_id])
        task_scores[task_id] = eligible_probs[:, proxy_index].astype(np.float64)
        task_scores_norm[task_id] = _minmax_normalize(task_scores[task_id])

        test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)
        test_rows = test_pos + test_neg
        test_probs, _ = predict_proxy_scores(bundle, test_rows, batch_size=batch_size)
        labels = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
        original_auc = quick_roc_auc(labels, test_probs[:, proxy_index])
        masked_rows = _mask_rows(test_rows, list(task.get("mask_keywords", [])))
        masked_probs, _ = predict_proxy_scores(bundle, masked_rows, batch_size=batch_size)
        masked_auc = quick_roc_auc(labels, masked_probs[:, proxy_index])
        task_reliability[task_id] = _clip01(masked_auc / original_auc) if original_auc and not math.isnan(original_auc) else float("nan")

    return {
        "method_name": "finetuned_encoder_multilabel",
        "task_scores": task_scores,
        "task_scores_norm": task_scores_norm,
        "task_reliability": task_reliability,
        "retrieval_matrix": eligible_embeddings.astype(np.float32),
        "encoder_model_id": bundle.model_id,
        "encoder_train_metrics": bundle.train_metrics,
    }


def _corr01(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape[0] != right.shape[0]:
        return float("nan")
    lvec = np.asarray(left, dtype=np.float64)
    rvec = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(lvec) & np.isfinite(rvec)
    if int(mask.sum()) < 2:
        return float("nan")
    lvec = lvec[mask]
    rvec = rvec[mask]
    if math.isclose(float(np.std(lvec)), 0.0) or math.isclose(float(np.std(rvec)), 0.0):
        return float("nan")
    corr = float(np.corrcoef(lvec, rvec)[0, 1])
    return _clip01((corr + 1.0) / 2.0)


def _fit_dense_artifact(
    benchmark_cfg: dict[str, Any],
    task_registry: dict[str, Any],
    eligible_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    method_cfg = benchmark_cfg["methods"]["dense_residual_logreg"]
    policy_cfg = benchmark_cfg["__policy_config"]
    site = str(method_cfg.get("site", "resid_post"))
    pooling = str(method_cfg.get("pooling", "mean"))
    robustness_pooling = str(method_cfg.get("robustness_pooling", "max"))
    extractor = InternalFeatureExtractor(policy_cfg, site=site, use_sae=False)

    task_scores: dict[str, np.ndarray] = {}
    task_scores_norm: dict[str, np.ndarray] = {}
    task_reliability: dict[str, float] = {}
    task_layers: dict[str, int] = {}
    selection_metrics: dict[str, float] = {}
    feature_dims: dict[str, int] = {}

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
        task_id = str(task["task_id"])
        selected_layer = int(summary["selected_layer"])
        eligible_features = extractor.extract(eligible_rows, layer=selected_layer, pooling=pooling)
        eligible_scores = _array_scores(final_model, eligible_features).astype(np.float64)
        task_scores[task_id] = eligible_scores
        task_scores_norm[task_id] = _minmax_normalize(eligible_scores)
        y_test = np.concatenate(
            [np.ones(len(splits["test_pos"]), dtype=np.int64), np.zeros(len(splits["test_neg"]), dtype=np.int64)]
        )
        masked_test_rows = _mask_rows(splits["test_pos"] + splits["test_neg"], list(task.get("mask_keywords", [])))
        masked_pos = masked_test_rows[: len(splits["test_pos"])]
        masked_neg = masked_test_rows[len(splits["test_pos"]) :]
        masked_features = np.concatenate(
            [
                extractor.extract(masked_pos, layer=selected_layer, pooling=robustness_pooling),
                extractor.extract(masked_neg, layer=selected_layer, pooling=robustness_pooling),
            ],
            axis=0,
        )
        masked_auc = quick_roc_auc(y_test, _array_scores(robust_model, masked_features))
        original_auc = _safe_float(summary["coverage_auc"])
        task_reliability[task_id] = (
            _clip01(masked_auc / original_auc) if not math.isnan(original_auc) and not math.isclose(original_auc, 0.0) else float("nan")
        )
        task_layers[task_id] = selected_layer
        selection_metrics[task_id] = _safe_float(summary["selection_metric"])
        feature_dims[task_id] = int(summary["feature_dim"])

    family_dense_vectors: dict[str, np.ndarray] = {}
    family_profiles: dict[str, np.ndarray] = {}
    family_layer_info: dict[str, dict[str, Any]] = {}
    for family_def in _family_defs(benchmark_cfg):
        left_task_id = family_def["left_task_id"]
        right_task_id = family_def["right_task_id"]
        candidates = [
            (left_task_id, _safe_float(selection_metrics.get(left_task_id)), task_layers.get(left_task_id)),
            (right_task_id, _safe_float(selection_metrics.get(right_task_id)), task_layers.get(right_task_id)),
        ]
        candidates = [row for row in candidates if row[2] is not None]
        if not candidates:
            continue
        best_task_id, best_metric, best_layer = sorted(
            candidates,
            key=lambda row: (
                math.isnan(row[1]),
                -(row[1] if not math.isnan(row[1]) else float("-inf")),
                int(row[2]),
            ),
        )[0]
        family = family_def["family"]
        family_dense_vectors[family] = extractor.extract(eligible_rows, layer=int(best_layer), pooling=pooling)
        family_profiles[family] = np.stack(
            [task_scores_norm[left_task_id], task_scores_norm[right_task_id]],
            axis=1,
        ).astype(np.float32)
        family_layer_info[family] = {
            "selected_layer": int(best_layer),
            "selected_from_task": best_task_id,
            "selection_metric": None if math.isnan(best_metric) else float(best_metric),
            "site": site,
            "pooling": pooling,
        }

    return {
        "method_name": "dense_residual_logreg",
        "task_scores": task_scores,
        "task_scores_norm": task_scores_norm,
        "task_reliability": task_reliability,
        "task_layers": task_layers,
        "selection_metrics": selection_metrics,
        "feature_dims": feature_dims,
        "family_dense_vectors": family_dense_vectors,
        "family_profiles": family_profiles,
        "family_layer_info": family_layer_info,
        "site": site,
        "pooling": pooling,
        "robustness_pooling": robustness_pooling,
    }


def _fit_sparse_artifact(
    benchmark_cfg: dict[str, Any],
    task_registry: dict[str, Any],
    eligible_rows: list[dict[str, Any]],
    *,
    dense_reference_norm: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    method_cfg = benchmark_cfg["methods"]["sparse_sae_feature_bank"]
    policy_cfg = benchmark_cfg["__policy_config"]
    site = str(method_cfg.get("site", "resid_post"))
    pooling = str(method_cfg.get("pooling", "mean"))
    robustness_pooling = str(method_cfg.get("robustness_pooling", "max"))
    seed = int(policy_cfg.get("seed", 0))
    perm_n = int(policy_cfg.get("perm_N", 2000))
    bootstrap_b = int(policy_cfg.get("bootstrap_B", 500))
    fdr_q = float(policy_cfg.get("fdr_q", 0.05))
    topk = int(method_cfg.get("topk", policy_cfg.get("topk", 64)))

    extractor = InternalFeatureExtractor(policy_cfg, site=site, use_sae=True)
    dense_extractor = InternalFeatureExtractor(policy_cfg, site=site, use_sae=False)

    task_scores: dict[str, np.ndarray] = {}
    task_scores_norm: dict[str, np.ndarray] = {}
    task_reliability: dict[str, float] = {}
    task_layers: dict[str, int] = {}
    selection_metrics: dict[str, float] = {}
    feature_counts: dict[str, int] = {}
    task_feature_ids: dict[str, list[int]] = {}
    task_feature_weights: dict[str, list[float]] = {}
    task_bootstrap_means: dict[str, float] = {}
    task_dense_sparse_agreement: dict[str, float] = {}
    task_feature_stability: dict[str, dict[int, float]] = {}
    task_selected_feature_activations: dict[str, np.ndarray] = {}
    task_banks: dict[str, dict[str, Any]] = {}

    for index, task in enumerate(task_registry["coverage_tasks"]):
        task_id = str(task["task_id"])
        train_pos, train_neg = _load_task_split_rows(task, "train", validated=False)
        test_pos, test_neg = _load_task_split_rows(task, "test", validated=False)

        best_layer = None
        best_metric = float("-inf")
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
            if source == "inner_train_valid" and pos_inner_valid and neg_inner_valid:
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
                metric = quick_roc_auc(valid_labels, _weighted_score(valid_features, bank["feature_ids"], bank["feature_weights"]))
            else:
                train_features = np.concatenate([train_pos_features, train_neg_features], axis=0)
                train_labels = np.concatenate(
                    [np.ones(len(pos_inner_train), dtype=np.int64), np.zeros(len(neg_inner_train), dtype=np.int64)]
                )
                metric = quick_roc_auc(train_labels, _weighted_score(train_features, bank["feature_ids"], bank["feature_weights"]))
            if math.isnan(metric):
                metric = float("-inf")
            if metric > best_metric:
                best_metric = metric
                best_layer = layer

        if best_layer is None:
            raise RuntimeError(f"Unable to select SAE layer for task={task_id}")

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

        eligible_features = extractor.extract(eligible_rows, layer=best_layer, pooling=pooling)
        eligible_scores = _weighted_score(eligible_features, bank["feature_ids"], bank["feature_weights"]).astype(np.float64)
        task_scores[task_id] = eligible_scores
        task_scores_norm[task_id] = _minmax_normalize(eligible_scores)

        test_features = np.concatenate(
            [
                extractor.extract(test_pos, layer=best_layer, pooling=pooling),
                extractor.extract(test_neg, layer=best_layer, pooling=pooling),
            ],
            axis=0,
        )
        y_test = np.concatenate([np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)])
        original_auc = quick_roc_auc(y_test, _weighted_score(test_features, bank["feature_ids"], bank["feature_weights"]))

        masked_rows = _mask_rows(test_pos + test_neg, list(task.get("mask_keywords", [])))
        masked_pos = masked_rows[: len(test_pos)]
        masked_neg = masked_rows[len(test_pos) :]
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
        masked_auc = quick_roc_auc(y_test, _weighted_score(masked_features, robust_bank["feature_ids"], robust_bank["feature_weights"]))
        task_reliability[task_id] = (
            _clip01(masked_auc / original_auc) if not math.isnan(original_auc) and not math.isclose(original_auc, 0.0) else float("nan")
        )
        task_layers[task_id] = int(best_layer)
        selection_metrics[task_id] = None if best_metric == float("-inf") else float(best_metric)
        feature_counts[task_id] = int(bank["feature_ids"].size)
        task_feature_ids[task_id] = [int(fid) for fid in bank["feature_ids"].tolist()]
        task_feature_weights[task_id] = [float(value) for value in bank["feature_weights"].tolist()]
        task_bootstrap_means[task_id] = _nanmean(list(bank["bootstrap_stability"].values()))
        task_feature_stability[task_id] = {
            int(feature_id): float(value) for feature_id, value in bank["bootstrap_stability"].items()
        }
        if bank["feature_ids"].size > 0:
            task_selected_feature_activations[task_id] = eligible_features[:, bank["feature_ids"]].astype(np.float32)
        else:
            task_selected_feature_activations[task_id] = np.zeros((len(eligible_rows), 0), dtype=np.float32)
        dense_norm = None if dense_reference_norm is None else dense_reference_norm.get(task_id)
        task_dense_sparse_agreement[task_id] = _corr01(task_scores_norm[task_id], dense_norm) if dense_norm is not None else float("nan")
        task_banks[task_id] = bank

    family_sparse_vectors: dict[str, np.ndarray] = {}
    family_dense_vectors: dict[str, np.ndarray] = {}
    family_profiles: dict[str, np.ndarray] = {}
    family_layer_info: dict[str, dict[str, Any]] = {}
    for family_def in _family_defs(benchmark_cfg):
        family = family_def["family"]
        left_task_id = family_def["left_task_id"]
        right_task_id = family_def["right_task_id"]
        candidates = [
            (left_task_id, _safe_float(selection_metrics.get(left_task_id)), task_layers.get(left_task_id)),
            (right_task_id, _safe_float(selection_metrics.get(right_task_id)), task_layers.get(right_task_id)),
        ]
        candidates = [row for row in candidates if row[2] is not None]
        if not candidates:
            continue
        best_task_id, best_metric, best_layer = sorted(
            candidates,
            key=lambda row: (
                math.isnan(row[1]),
                -(row[1] if not math.isnan(row[1]) else float("-inf")),
                int(row[2]),
            ),
        )[0]
        family_feature_ids: list[int] = []
        for task_id in (left_task_id, right_task_id):
            if task_layers.get(task_id) == int(best_layer):
                family_feature_ids.extend(task_feature_ids.get(task_id, []))
        if not family_feature_ids:
            family_feature_ids = list(task_feature_ids.get(best_task_id, []))
        family_feature_ids = sorted(set(int(fid) for fid in family_feature_ids))
        sparse_matrix = extractor.extract(eligible_rows, layer=int(best_layer), pooling=pooling)
        family_sparse_vectors[family] = (
            sparse_matrix[:, family_feature_ids].astype(np.float32)
            if family_feature_ids
            else np.zeros((len(eligible_rows), 1), dtype=np.float32)
        )
        family_dense_vectors[family] = dense_extractor.extract(eligible_rows, layer=int(best_layer), pooling=pooling).astype(np.float32)
        family_profiles[family] = np.stack(
            [task_scores_norm[left_task_id], task_scores_norm[right_task_id]],
            axis=1,
        ).astype(np.float32)
        family_layer_info[family] = {
            "selected_layer": int(best_layer),
            "selected_from_task": best_task_id,
            "selection_metric": None if math.isnan(best_metric) else float(best_metric),
            "feature_ids": family_feature_ids,
            "site": site,
            "pooling": pooling,
        }

    return {
        "method_name": "sparse_sae_feature_bank",
        "task_scores": task_scores,
        "task_scores_norm": task_scores_norm,
        "task_reliability": task_reliability,
        "task_layers": task_layers,
        "selection_metrics": selection_metrics,
        "feature_counts": feature_counts,
        "task_feature_ids": task_feature_ids,
        "task_feature_weights": task_feature_weights,
        "task_bootstrap_means": task_bootstrap_means,
        "task_dense_sparse_agreement": task_dense_sparse_agreement,
        "task_feature_stability": task_feature_stability,
        "task_selected_feature_activations": task_selected_feature_activations,
        "task_banks": task_banks,
        "family_sparse_vectors": family_sparse_vectors,
        "family_dense_vectors": family_dense_vectors,
        "family_profiles": family_profiles,
        "family_layer_info": family_layer_info,
        "site": site,
        "pooling": pooling,
        "robustness_pooling": robustness_pooling,
    }


def _top_sparse_feature_fields(
    artifact: dict[str, Any],
    task_id: str,
    row_index: int,
    *,
    top_n: int = 3,
) -> tuple[list[int], list[float], list[float], float]:
    feature_ids = [int(value) for value in artifact.get("task_feature_ids", {}).get(task_id, [])]
    feature_weights = [float(value) for value in artifact.get("task_feature_weights", {}).get(task_id, [])]
    stability_map = {
        int(feature_id): float(value)
        for feature_id, value in artifact.get("task_feature_stability", {}).get(task_id, {}).items()
    }
    if not feature_ids or not feature_weights:
        return [], [], [], float("nan")

    activation_matrix = artifact.get("task_selected_feature_activations", {}).get(task_id)
    if activation_matrix is not None and row_index < int(activation_matrix.shape[0]):
        activations = np.asarray(activation_matrix[row_index], dtype=np.float64)
        contributions = activations * np.asarray(feature_weights, dtype=np.float64)
        positive_order = [int(i) for i in np.argsort(-contributions) if contributions[int(i)] > 0]
        if positive_order:
            ranked = positive_order[:top_n]
        else:
            ranked = [int(i) for i in np.argsort(-np.abs(contributions))[:top_n]]
    else:
        ranked = [int(i) for i in np.argsort(-np.abs(np.asarray(feature_weights, dtype=np.float64)))[:top_n]]

    top_feature_ids = [int(feature_ids[i]) for i in ranked]
    top_feature_weights = [float(feature_weights[i]) for i in ranked]
    if activation_matrix is not None and row_index < int(activation_matrix.shape[0]):
        activations = np.asarray(activation_matrix[row_index], dtype=np.float64)
        contributions = activations * np.asarray(feature_weights, dtype=np.float64)
        top_feature_contributions = [float(contributions[i]) for i in ranked]
    else:
        top_feature_contributions = [float("nan") for _ in ranked]
    mean_stability = _nanmean([stability_map.get(int(feature_id), float("nan")) for feature_id in top_feature_ids])
    return top_feature_ids, top_feature_weights, top_feature_contributions, mean_stability


def _build_family_cards(
    rows: list[dict[str, Any]],
    family_def: dict[str, Any],
    artifact: dict[str, Any],
    scoring_cfg: dict[str, Any],
    proxy_evidence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    left_task_id = family_def["left_task_id"]
    right_task_id = family_def["right_task_id"]
    left_scores = artifact["task_scores_norm"][left_task_id]
    right_scores = artifact["task_scores_norm"][right_task_id]
    family = family_def["family"]
    sparse_vectors = artifact.get("family_sparse_vectors", {}).get(family)

    concern_weight = float(scoring_cfg["weights"]["concern"])
    related_weight = float(scoring_cfg["weights"]["related_support"])
    reliability_weight = float(scoring_cfg["weights"]["reliability"])
    proxy_causal_badges = {} if proxy_evidence is None else dict(proxy_evidence.get("proxy_causal_badges", {}))

    cards: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        left_score = float(left_scores[index])
        right_score = float(right_scores[index])
        if left_score >= right_score:
            anchor_task_id = left_task_id
            anchor_display = family_def["left_display_name"]
            anchor_score = left_score
            related_score = right_score
        else:
            anchor_task_id = right_task_id
            anchor_display = family_def["right_display_name"]
            anchor_score = right_score
            related_score = left_score

        reliability_components = [artifact["task_reliability"].get(anchor_task_id, float("nan"))]
        if artifact["method_name"] == "sparse_sae_feature_bank":
            reliability_components.append(artifact["task_bootstrap_means"].get(anchor_task_id, float("nan")))
            reliability_components.append(artifact["task_dense_sparse_agreement"].get(anchor_task_id, float("nan")))
        reliability_score = _clip01(_nanmean(reliability_components))
        concern_component = 0.0 if math.isnan(anchor_score) else anchor_score
        related_component = 0.0 if math.isnan(related_score) else related_score
        reliability_component = 0.0 if math.isnan(reliability_score) else reliability_score
        priority_score = (
            concern_weight * concern_component
            + related_weight * related_component
            + reliability_weight * reliability_component
        )

        supporting_feature_count = None
        top_feature_ids: list[int] = []
        top_feature_weights: list[float] = []
        top_feature_contributions: list[float] = []
        mean_feature_stability = float("nan")
        causal_badge = "not_tested"
        if sparse_vectors is not None:
            supporting_feature_count = int(np.sum(sparse_vectors[index] > 0))
        if artifact["method_name"] == "sparse_sae_feature_bank":
            top_feature_ids, top_feature_weights, top_feature_contributions, mean_feature_stability = _top_sparse_feature_fields(
                artifact,
                anchor_task_id,
                index,
            )
            causal_badge = str(proxy_causal_badges.get(anchor_task_id, "not_tested"))
        elif artifact["method_name"] == "dense_residual_logreg":
            supporting_feature_count = int(artifact.get("feature_dims", {}).get(anchor_task_id, 0))

        card = {
            "document_id": str(row["document_id"]),
            "segment_id": str(row["segment_id"]),
            "family": family,
            "family_display_name": family_def["family_display_name"],
            "proxy_anchor": anchor_task_id,
            "proxy_anchor_display_name": anchor_display,
            "segment_text": str(row["text"]),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
            "section_hint": row.get("section_hint"),
            "concern_score": float(anchor_score),
            "related_support_score": float(related_score),
            "reliability_score": float(reliability_score) if not math.isnan(reliability_score) else float("nan"),
            "priority_score": float(priority_score),
            "document_rank": None,
            "document_segment_count": None,
            "supporting_feature_count": supporting_feature_count,
            "selected_layer": artifact.get("task_layers", {}).get(anchor_task_id),
            "top_feature_ids": top_feature_ids,
            "top_feature_weights": top_feature_weights,
            "top_feature_contributions": top_feature_contributions,
            "mean_feature_stability": float(mean_feature_stability) if not math.isnan(mean_feature_stability) else float("nan"),
            "causal_badge": causal_badge,
            "retrieved_examples": [],
            "natural_language_note": "",
        }
        cards.append(card)

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        by_doc.setdefault(card["document_id"], []).append(card)
    for document_cards in by_doc.values():
        document_cards.sort(key=lambda item: (-float(item["priority_score"]), item["segment_id"]))
        total = len(document_cards)
        for rank, card in enumerate(document_cards, start=1):
            card["document_rank"] = int(rank)
            card["document_segment_count"] = int(total)
            card["natural_language_note"] = build_segment_note(card)
    return cards


def _evaluate_highlighting(
    rows: list[dict[str, Any]],
    family_def: dict[str, Any],
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = _labels_for_family(rows, family_def)
    scores = np.asarray([float(card["priority_score"]) for card in cards], dtype=np.float64)
    return {
        "family": family_def["family"],
        "family_display_name": family_def["family_display_name"],
        "segment_auroc": quick_roc_auc(labels, scores),
        "segment_auprc": average_precision(labels, scores),
        "precision_at_3": _mean_document_metric(
            rows,
            labels,
            scores,
            lambda y_doc, s_doc: precision_at_k_binary(y_doc, s_doc, k=3),
        ),
        "recall_at_3": _mean_document_metric(
            rows,
            labels,
            scores,
            lambda y_doc, s_doc: recall_at_k_binary(y_doc, s_doc, k=3),
        ),
        "mean_first_relevant_rank": _mean_document_metric(
            rows,
            labels,
            scores,
            lambda y_doc, s_doc: first_relevant_rank_binary(y_doc, s_doc),
        ),
    }


def _evaluate_triage(
    rows: list[dict[str, Any]],
    family_def: dict[str, Any],
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = _labels_for_family(rows, family_def)
    scores = np.asarray([float(card["priority_score"]) for card in cards], dtype=np.float64)
    first_rank = _mean_document_metric(
        rows,
        labels,
        scores,
        lambda y_doc, s_doc: first_relevant_rank_binary(y_doc, s_doc),
    )
    return {
        "family": family_def["family"],
        "family_display_name": family_def["family_display_name"],
        "recall_at_3": _mean_document_metric(
            rows,
            labels,
            scores,
            lambda y_doc, s_doc: recall_at_k_binary(y_doc, s_doc, k=3),
        ),
        "first_relevant_rank": first_rank,
        "average_review_depth": first_rank,
        "ndcg_at_5": _mean_document_metric(
            rows,
            labels,
            scores,
            lambda y_doc, s_doc: ndcg_at_k_binary(y_doc, s_doc, k=min(5, len(y_doc))),
        ),
    }


def _family_similarity(
    artifact: dict[str, Any],
    family_def: dict[str, Any],
    query_index: int,
    candidate_index: int,
) -> float:
    family = family_def["family"]
    method_name = artifact["method_name"]
    if method_name in {"lexical_tfidf_logreg", "semantic_sentence_embed_logreg", "finetuned_encoder_multilabel"}:
        matrix = np.asarray(artifact["retrieval_matrix"], dtype=np.float32)
        return cosine_similarity(matrix[query_index], matrix[candidate_index])
    if method_name == "dense_residual_logreg":
        matrix = np.asarray(artifact["family_dense_vectors"][family], dtype=np.float32)
        return cosine_similarity(matrix[query_index], matrix[candidate_index])
    sparse_matrix = np.asarray(artifact["family_sparse_vectors"][family], dtype=np.float32)
    dense_matrix = np.asarray(artifact["family_dense_vectors"][family], dtype=np.float32)
    profile_matrix = np.asarray(artifact["family_profiles"][family], dtype=np.float32)
    sparse_cos = cosine_similarity(sparse_matrix[query_index], sparse_matrix[candidate_index])
    dense_cos = cosine_similarity(dense_matrix[query_index], dense_matrix[candidate_index])
    profile_cos = cosine_similarity(profile_matrix[query_index], profile_matrix[candidate_index])
    return float(0.50 * sparse_cos + 0.30 * dense_cos + 0.20 * profile_cos)


def _evaluate_retrieval(
    rows: list[dict[str, Any]],
    family_def: dict[str, Any],
    cards: list[dict[str, Any]],
    artifact: dict[str, Any],
    *,
    top_k: int,
    surfaced_top_k: int,
    all_core_proxy_names: set[str],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    left_proxy = family_def["left_proxy_name"]
    right_proxy = family_def["right_proxy_name"]
    labels = _labels_for_family(rows, family_def)
    card_by_segment = {str(card["segment_id"]): card for card in cards}

    surfaced_cards = sorted(cards, key=lambda item: (-float(item["priority_score"]), item["segment_id"]))[:surfaced_top_k]
    surfaced_segment_ids = {str(card["segment_id"]) for card in surfaced_cards}
    retrieved_examples: dict[str, list[dict[str, Any]]] = {segment_id: [] for segment_id in surfaced_segment_ids}
    surfaced_index_set = {
        index
        for index, row in enumerate(rows)
        if str(row["segment_id"]) in surfaced_segment_ids
    }

    hit_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    within_minus_cross_values: list[float] = []

    for query_index, row in enumerate(rows):
        if int(labels[query_index]) != 1 and query_index not in surfaced_index_set:
            continue
        query_doc_id = str(row["document_id"])
        query_tags = set(str(tag) for tag in row.get("all_tags", []))
        if left_proxy in query_tags:
            query_anchor = left_proxy
        elif right_proxy in query_tags:
            query_anchor = right_proxy
        else:
            query_anchor = left_proxy if float(cards[query_index]["concern_score"]) >= float(cards[query_index]["related_support_score"]) else right_proxy

        candidate_payloads: list[dict[str, Any]] = []
        for candidate_index, candidate_row in enumerate(rows):
            if candidate_index == query_index:
                continue
            if str(candidate_row["document_id"]) == query_doc_id:
                continue
            candidate_tags = set(str(tag) for tag in candidate_row.get("all_tags", []))
            matched_proxy = None
            gain = 0
            if query_anchor in candidate_tags:
                matched_proxy = query_anchor
                gain = 2
            elif left_proxy in candidate_tags or right_proxy in candidate_tags:
                matched_proxy = left_proxy if left_proxy in candidate_tags else right_proxy
                gain = 1
            other_family_positive = int(any(str(tag) in all_core_proxy_names - {left_proxy, right_proxy} for tag in candidate_tags))
            similarity = _family_similarity(artifact, family_def, query_index, candidate_index)
            candidate_payloads.append(
                {
                    "candidate_index": candidate_index,
                    "similarity": float(similarity),
                    "gain": gain,
                    "matched_proxy": matched_proxy,
                    "cross_family_positive": other_family_positive,
                    "row": candidate_row,
                }
            )

        candidate_payloads.sort(key=lambda item: (-float(item["similarity"]), str(item["row"]["segment_id"])))
        top_candidates = candidate_payloads[:top_k]
        gains = [int(item["gain"]) for item in top_candidates]
        if int(labels[query_index]) == 1:
            hit_values.append(1.0 if any(gain > 0 for gain in gains) else 0.0)
            mrr_values.append(reciprocal_rank_from_order([1 if gain > 0 else 0 for gain in gains]))
            if gains:
                dcg = 0.0
                for rank, gain in enumerate(gains, start=1):
                    dcg += (2 ** gain - 1) / math.log2(rank + 1)
                ideal = sorted(gains, reverse=True)
                idcg = 0.0
                for rank, gain in enumerate(ideal, start=1):
                    idcg += (2 ** gain - 1) / math.log2(rank + 1)
                ndcg_values.append(float(dcg / idcg) if not math.isclose(idcg, 0.0) else float("nan"))
            else:
                ndcg_values.append(float("nan"))
            within_rate = float(sum(1 for item in top_candidates if int(item["gain"]) > 0)) / float(top_k) if top_k > 0 else float("nan")
            cross_rate = float(sum(1 for item in top_candidates if int(item["cross_family_positive"]) > 0)) / float(top_k) if top_k > 0 else float("nan")
            within_minus_cross_values.append(within_rate - cross_rate)

        query_segment_id = str(row["segment_id"])
        if query_segment_id in surfaced_segment_ids:
            examples: list[dict[str, Any]] = []
            for item in top_candidates:
                candidate_row = item["row"]
                examples.append(
                    {
                        "retrieved_document_id": str(candidate_row["document_id"]),
                        "retrieved_segment_id": str(candidate_row["segment_id"]),
                        "retrieval_similarity": float(item["similarity"]),
                        "matched_family": family_def["family"] if int(item["gain"]) > 0 else None,
                        "matched_proxy": item["matched_proxy"],
                    }
                )
            retrieved_examples[query_segment_id] = examples

    for card in surfaced_cards:
        card["retrieved_examples"] = retrieved_examples.get(str(card["segment_id"]), [])

    return (
        {
            "family": family_def["family"],
            "family_display_name": family_def["family_display_name"],
            "hit_at_5": _nanmean(hit_values),
            "mrr_at_5": _nanmean(mrr_values),
            "ndcg_at_5": _nanmean(ndcg_values),
            "within_minus_cross_rate": _nanmean(within_minus_cross_values),
        },
        retrieved_examples,
    )


def _build_method_summary(
    assistant_cfg: dict[str, Any],
    benchmark_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    proxy_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    top_k_segments = int(assistant_cfg["analysis"].get("document_top_k_segments", 3))
    surfaced_top_k = int(assistant_cfg["analysis"].get("surfaced_segments_per_family", 5))
    retrieval_top_k = int(assistant_cfg["retrieval"].get("top_k", 5))

    family_summaries: list[dict[str, Any]] = []
    all_cards: list[dict[str, Any]] = []
    surfaced_cards: list[dict[str, Any]] = []
    highlighting_metrics: list[dict[str, Any]] = []
    retrieval_metrics: list[dict[str, Any]] = []
    triage_metrics: list[dict[str, Any]] = []
    retrieval_examples_by_segment: dict[str, list[dict[str, Any]]] = {}

    all_core_proxy_names = {
        family_def["left_proxy_name"]
        for family_def in _family_defs(benchmark_cfg)
    } | {
        family_def["right_proxy_name"]
        for family_def in _family_defs(benchmark_cfg)
    }

    for family_def in _family_defs(benchmark_cfg):
        cards = _build_family_cards(rows, family_def, artifact, assistant_cfg["scoring"], proxy_evidence)
        highlighting = _evaluate_highlighting(rows, family_def, cards)
        retrieval, examples = _evaluate_retrieval(
            rows,
            family_def,
            cards,
            artifact,
            top_k=retrieval_top_k,
            surfaced_top_k=surfaced_top_k,
            all_core_proxy_names=all_core_proxy_names,
        )
        triage = _evaluate_triage(rows, family_def, cards)
        family_rows, document_briefs = _document_family_aggregation(cards, top_k_segments)

        family_summaries.append(
            {
                "family": family_def["family"],
                "family_display_name": family_def["family_display_name"],
                "document_family_scores": family_rows,
                "document_briefs": document_briefs,
            }
        )
        all_cards.extend(cards)
        surfaced_cards.extend(sorted(cards, key=lambda item: (-float(item["priority_score"]), item["segment_id"]))[:surfaced_top_k])
        highlighting_metrics.append(highlighting)
        retrieval_metrics.append(retrieval)
        triage_metrics.append(triage)
        retrieval_examples_by_segment.update(examples)

    all_family_rows, all_document_briefs = _document_family_aggregation(all_cards, top_k_segments)
    for card in surfaced_cards:
        card["retrieved_examples"] = retrieval_examples_by_segment.get(str(card["segment_id"]), card.get("retrieved_examples", []))
        card["natural_language_note"] = build_segment_note(card)

    return {
        "method_name": artifact["method_name"],
        "segment_cards": surfaced_cards,
        "all_segment_cards": all_cards,
        "family_summaries": family_summaries,
        "document_family_scores": all_family_rows,
        "document_briefs": all_document_briefs,
        "highlighting": highlighting_metrics,
        "retrieval": retrieval_metrics,
        "triage": triage_metrics,
    }


def _assistant_feature_card_rows(method_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in method_summaries:
        method_name = str(summary["method_name"])
        for card in summary["segment_cards"]:
            row = dict(card)
            row["method_name"] = method_name
            rows.append(row)
    return rows


def _assistant_feature_usage_rows(method_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, int], int] = {}
    for summary in method_summaries:
        method_name = str(summary["method_name"])
        for card in summary["segment_cards"]:
            proxy_anchor = str(card.get("proxy_anchor", ""))
            family = str(card.get("family", ""))
            for feature_id in card.get("top_feature_ids", []) or []:
                key = (method_name, family, proxy_anchor, int(feature_id))
                counts[key] = counts.get(key, 0) + 1
    rows = [
        {
            "method_name": method_name,
            "family": family,
            "proxy_anchor": proxy_anchor,
            "feature_id": feature_id,
            "assistant_usage_count": usage_count,
        }
        for (method_name, family, proxy_anchor, feature_id), usage_count in counts.items()
    ]
    rows.sort(key=lambda row: (str(row["method_name"]), str(row["family"]), str(row["proxy_anchor"]), -int(row["assistant_usage_count"]), int(row["feature_id"])))
    return rows


def _assistant_card_dossier_link_rows(
    method_summaries: list[dict[str, Any]],
    proxy_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    sparse_dossiers = dict(proxy_evidence.get("dossier_by_proxy_feature", {}))
    rows: list[dict[str, Any]] = []
    for summary in method_summaries:
        if str(summary["method_name"]) != "sparse_sae_feature_bank":
            continue
        for card in summary["segment_cards"]:
            proxy_anchor = str(card.get("proxy_anchor", ""))
            feature_ids = [int(value) for value in card.get("top_feature_ids", [])]
            feature_weights = [float(value) for value in card.get("top_feature_weights", [])]
            feature_contributions = [float(value) for value in card.get("top_feature_contributions", [])]
            for rank, feature_id in enumerate(feature_ids, start=1):
                dossier_row = sparse_dossiers.get((proxy_anchor, int(feature_id)), {})
                rows.append(
                    {
                        "method_name": summary["method_name"],
                        "document_id": card.get("document_id"),
                        "segment_id": card.get("segment_id"),
                        "family": card.get("family"),
                        "proxy_anchor": proxy_anchor,
                        "selected_layer": card.get("selected_layer"),
                        "feature_rank": int(rank),
                        "feature_id": int(feature_id),
                        "feature_weight": feature_weights[rank - 1] if rank - 1 < len(feature_weights) else float("nan"),
                        "feature_contribution": feature_contributions[rank - 1] if rank - 1 < len(feature_contributions) else float("nan"),
                        "mean_feature_stability": card.get("mean_feature_stability"),
                        "causal_badge": card.get("causal_badge"),
                        "dossier_layer": dossier_row.get("layer"),
                        "dossier_feature_priority": dossier_row.get("feature_priority"),
                        "dossier_bootstrap_stability": dossier_row.get("bootstrap_stability"),
                        "dossier_activation_gap": dossier_row.get("activation_gap"),
                        "dossier_contribution_gap": dossier_row.get("contribution_gap"),
                    }
                )
    return rows


def _summary_table_rows(method_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in method_summaries:
        highlight_values = []
        retrieval_values = []
        triage_values = []
        for payload in summary["highlighting"]:
            first_rank = _safe_float(payload["mean_first_relevant_rank"])
            first_rank_score = 0.0 if math.isnan(first_rank) or math.isclose(first_rank, 0.0) else min(1.0, 1.0 / first_rank)
            highlight_values.append(
                _nanmean(
                    [
                        _safe_float(payload["segment_auroc"]),
                        _safe_float(payload["segment_auprc"]),
                        _safe_float(payload["precision_at_3"]),
                        _safe_float(payload["recall_at_3"]),
                        first_rank_score,
                    ]
                )
            )
        for payload in summary["retrieval"]:
            gap = _safe_float(payload["within_minus_cross_rate"])
            gap_score = float("nan") if math.isnan(gap) else _clip01((gap + 1.0) / 2.0)
            retrieval_values.append(
                _nanmean(
                    [
                        _safe_float(payload["hit_at_5"]),
                        _safe_float(payload["mrr_at_5"]),
                        _safe_float(payload["ndcg_at_5"]),
                        gap_score,
                    ]
                )
            )
        for payload in summary["triage"]:
            first_rank = _safe_float(payload["first_relevant_rank"])
            depth = _safe_float(payload["average_review_depth"])
            triage_values.append(
                _nanmean(
                    [
                        _safe_float(payload["recall_at_3"]),
                        0.0 if math.isnan(first_rank) or math.isclose(first_rank, 0.0) else min(1.0, 1.0 / first_rank),
                        0.0 if math.isnan(depth) or math.isclose(depth, 0.0) else min(1.0, 1.0 / depth),
                        _safe_float(payload["ndcg_at_5"]),
                    ]
                )
            )
        highlight_score = _nanmean(highlight_values)
        retrieval_score = _nanmean(retrieval_values)
        triage_score = _nanmean(triage_values)
        core_score = _nanmean([highlight_score, retrieval_score, triage_score])
        rows.append(
            {
                "method_name": summary["method_name"],
                "HighlightScore": highlight_score,
                "RetrievalScore": retrieval_score,
                "TriageScore": triage_score,
                "AssistantCoreScore": core_score,
            }
        )
    rows.sort(key=lambda item: (-_safe_float(item["AssistantCoreScore"]), str(item["method_name"])))
    return rows


def _load_trust_bundle(assistant_cfg: dict[str, Any], benchmark_cfg: dict[str, Any]) -> dict[str, Any]:
    del benchmark_cfg
    proxy_evidence = _load_sparse_proxy_evidence(assistant_cfg)
    return {
        "status": proxy_evidence["status"],
        "benchmark_output_root": proxy_evidence["benchmark_output_root"],
        "summary_root": proxy_evidence["summary_root"],
        "proxy_feature_evidence": proxy_evidence["proxy_feature_rows"],
        "proxy_causal_evidence": proxy_evidence["proxy_causal_rows"],
        "pair_mechanistic_evidence": proxy_evidence["pair_rows"],
    }


def _json_ready_method_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_name": summary["method_name"],
        "segment_cards": summary["segment_cards"],
        "document_family_scores": summary["document_family_scores"],
        "document_briefs": summary["document_briefs"],
        "highlighting": summary["highlighting"],
        "retrieval": summary["retrieval"],
        "triage": summary["triage"],
    }


def run_policy_analysis_experiments(
    config_path: str | Path = ROOT / "configs" / "policy_analysis_assistant.yaml",
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    assistant_cfg = _load_assistant_config(config_path)
    benchmark_cfg = assistant_cfg["__benchmark"]
    if output_root is not None:
        assistant_cfg["output_root"] = str(_resolve_path(output_root))
    out_root = ensure_dir(assistant_cfg["output_root"])
    method_results_root = ensure_dir(out_root / "method_results")
    summary_root = ensure_dir(out_root / "summary")

    task_registry = build_task_registry(benchmark_cfg)
    manifest_root = Path(benchmark_cfg["manifest_root"])
    evaluation_split = str(assistant_cfg.get("splits", {}).get("evaluation", "test"))
    eligible_rows = _eligible_rows(manifest_root, evaluation_split)
    proxy_evidence = _load_sparse_proxy_evidence(assistant_cfg)
    enabled_methods = [
        method_name
        for method_name, method_cfg in benchmark_cfg["methods"].items()
        if bool(method_cfg.get("enabled", False))
    ]
    if not enabled_methods:
        raise ValueError("No enabled methods found in the benchmark config.")

    artifacts: list[dict[str, Any]] = []
    dense_artifact: dict[str, Any] | None = None
    if "lexical_tfidf_logreg" in enabled_methods:
        artifacts.append(_fit_lexical_artifact(task_registry, eligible_rows))
    if "semantic_sentence_embed_logreg" in enabled_methods:
        artifacts.append(_fit_sentence_artifact(benchmark_cfg, task_registry, eligible_rows))
    if "finetuned_encoder_multilabel" in enabled_methods:
        artifacts.append(_fit_finetuned_encoder_artifact(benchmark_cfg, task_registry, eligible_rows))
    if "dense_residual_logreg" in enabled_methods:
        dense_artifact = _fit_dense_artifact(benchmark_cfg, task_registry, eligible_rows)
        artifacts.append(dense_artifact)
    if "sparse_sae_feature_bank" in enabled_methods:
        artifacts.append(
            _fit_sparse_artifact(
                benchmark_cfg,
                task_registry,
                eligible_rows,
                dense_reference_norm=None if dense_artifact is None else dense_artifact["task_scores_norm"],
            )
        )

    method_summaries: list[dict[str, Any]] = []
    method_output_paths: dict[str, str] = {}
    for artifact in artifacts:
        artifact["benchmark_config_path"] = benchmark_cfg["__config_path"]
        summary = _build_method_summary(assistant_cfg, benchmark_cfg, eligible_rows, artifact, proxy_evidence)
        method_summaries.append(summary)
        output_path = method_results_root / f"{artifact['method_name']}.json"
        save_json(output_path, _json_ready_method_summary(summary))
        method_output_paths[artifact["method_name"]] = str(output_path)

    leaderboard = _summary_table_rows(method_summaries)
    highlighting_summary = {summary["method_name"]: summary["highlighting"] for summary in method_summaries}
    retrieval_summary = {summary["method_name"]: summary["retrieval"] for summary in method_summaries}
    triage_summary = {summary["method_name"]: summary["triage"] for summary in method_summaries}
    trust_bundle = _load_trust_bundle(assistant_cfg, benchmark_cfg)
    assistant_feature_cards = _assistant_feature_card_rows(method_summaries)
    assistant_feature_usage = _assistant_feature_usage_rows(method_summaries)
    assistant_card_dossier_links = _assistant_card_dossier_link_rows(method_summaries, proxy_evidence)
    assistant_report = {
        "assistant_name": "Policy Feature Mechanistic Rerun v2 Assistant Layer",
        "config_path": assistant_cfg["__config_path"],
        "benchmark_config_path": benchmark_cfg["__config_path"],
        "benchmark_output_root": assistant_cfg["benchmark_output_root"],
        "evaluation_split": evaluation_split,
        "n_segments": len(eligible_rows),
        "methods": [summary["method_name"] for summary in method_summaries],
        "leaderboard": leaderboard,
        "trust_bundle_status": trust_bundle["status"],
        "method_output_paths": method_output_paths,
        "assistant_feature_cards_path": str(summary_root / "assistant_feature_cards.jsonl"),
        "assistant_feature_usage_path": str(summary_root / "assistant_feature_usage.csv"),
        "assistant_card_dossier_links_path": str(summary_root / "assistant_card_dossier_links.jsonl"),
    }

    save_json(summary_root / "assistant_leaderboard.json", leaderboard)
    save_json(summary_root / "highlighting_summary.json", highlighting_summary)
    save_json(summary_root / "retrieval_summary.json", retrieval_summary)
    save_json(summary_root / "triage_summary.json", triage_summary)
    save_json(summary_root / "trust_bundle.json", trust_bundle)
    save_json(summary_root / "assistant_report.json", assistant_report)
    _write_jsonl(summary_root / "assistant_feature_cards.jsonl", assistant_feature_cards)
    _write_jsonl(summary_root / "assistant_card_dossier_links.jsonl", assistant_card_dossier_links)
    with (summary_root / "assistant_feature_usage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method_name", "family", "proxy_anchor", "feature_id", "assistant_usage_count"])
        writer.writeheader()
        for row in assistant_feature_usage:
            writer.writerow(row)
    return {
        "output_root": str(out_root),
        "method_output_paths": method_output_paths,
        "leaderboard_path": str(summary_root / "assistant_leaderboard.json"),
        "report_path": str(summary_root / "assistant_report.json"),
        "assistant_feature_cards_path": str(summary_root / "assistant_feature_cards.jsonl"),
        "assistant_feature_usage_path": str(summary_root / "assistant_feature_usage.csv"),
        "assistant_card_dossier_links_path": str(summary_root / "assistant_card_dossier_links.jsonl"),
    }


def analyze_document_with_sparse_assistant(
    payload: str | list[str] | list[dict[str, Any]] | dict[str, Any],
    *,
    config_path: str | Path = ROOT / "configs" / "policy_analysis_assistant.yaml",
    output_path: str | Path | None = None,
    document_id: str = "document_1",
    title: str = "Untitled document",
    source_type: str = "user_text",
) -> dict[str, Any]:
    assistant_cfg = _load_assistant_config(config_path)
    benchmark_cfg = assistant_cfg["__benchmark"]
    task_registry = build_task_registry(benchmark_cfg)

    normalized_document = normalize_document_input(
        payload,
        document_id=document_id,
        title=title,
        source_type=source_type,
    )
    segments = segment_document(
        normalized_document,
        chunk_chars=int(assistant_cfg["segmentation"].get("chunk_chars", 1200)),
        overlap_chars=int(assistant_cfg["segmentation"].get("overlap_chars", 200)),
    )
    prototype_rows = [
        {
            "document_id": normalized_document["document_id"],
            "segment_id": str(segment["segment_id"]),
            "text": str(segment["segment_text"]),
            "all_tags": [],
            "char_start": segment.get("char_start"),
            "char_end": segment.get("char_end"),
            "section_hint": segment.get("section_hint"),
            "metadata": {
                "title": normalized_document["title"],
                "source_type": normalized_document["source_type"],
            },
        }
        for segment in segments
    ]

    corpus_rows = _eligible_rows(Path(benchmark_cfg["manifest_root"]), str(assistant_cfg.get("splits", {}).get("evaluation", "test")))
    combined_rows = corpus_rows + prototype_rows
    proxy_evidence = _load_sparse_proxy_evidence(assistant_cfg)
    dense_artifact = _fit_dense_artifact(benchmark_cfg, task_registry, combined_rows)
    sparse_artifact = _fit_sparse_artifact(
        benchmark_cfg,
        task_registry,
        combined_rows,
        dense_reference_norm=dense_artifact["task_scores_norm"],
    )
    summary = _build_method_summary(assistant_cfg, benchmark_cfg, combined_rows, sparse_artifact, proxy_evidence)

    prototype_cards = [
        card
        for card in summary["all_segment_cards"]
        if str(card["document_id"]) == str(normalized_document["document_id"])
    ]
    prototype_cards.sort(key=lambda item: (-float(item["priority_score"]), item["family"], item["segment_id"]))
    family_rows, document_briefs = _document_family_aggregation(
        prototype_cards,
        int(assistant_cfg["analysis"].get("document_top_k_segments", 3)),
    )
    result = {
        "normalized_document": normalized_document,
        "segments": segments,
        "segment_cards": prototype_cards,
        "document_family_scores": family_rows,
        "document_briefs": document_briefs,
    }
    if output_path is not None:
        save_json(output_path, result)
    return result
