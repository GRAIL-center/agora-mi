"""Baseline runners for sentence embeddings, dense residuals, and sparse module probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from policy_interp.discovery import DiscoveryArtifacts
from policy_interp.feature_matrix import build_module_score_matrix, load_residual_matrix, matrix_from_vector_frame
from policy_interp.io import read_parquet, write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.text_models import SentenceEncoder


@dataclass(slots=True)
class BaselineArtifacts:
    comparison_path: str
    dense_model_selection_path: str
    sparse_feature_selection_path: str
    sparse_feature_module_overlap_path: str


def run_baselines(config: ExperimentConfig, discovery: DiscoveryArtifacts | None = None) -> BaselineArtifacts:
    segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    stable_modules = _read_stable_modules(config.run_root / "discovery" / "module_stability.parquet")
    extraction_root = config.run_root / "extraction"
    baseline_root = config.run_root / "baselines"
    baseline_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    dense_selection_rows: list[dict[str, object]] = []
    sparse_feature_selection_rows: list[dict[str, object]] = []

    sentence_rows = _run_sentence_embedding_baseline(segments, config)
    rows.extend(sentence_rows)

    for proxy_key in config.dataset.proxy_columns.keys():
        dense_result = _run_dense_residual_baseline(segments, extraction_root, proxy_key, config)
        rows.append(dense_result["summary"])
        dense_selection_rows.append(dense_result["selection"])

        sparse_result = _run_sparse_module_baseline(segments, stable_modules, extraction_root, proxy_key, config)
        rows.append(sparse_result)

        individual_result = _run_sparse_individual_feature_baseline(segments, extraction_root, proxy_key, config)
        rows.append(individual_result["summary"])
        sparse_feature_selection_rows.append(individual_result["selection"])

    comparison_frame = pd.DataFrame(rows)
    dense_selection_frame = pd.DataFrame(dense_selection_rows)
    sparse_feature_selection_frame = pd.DataFrame(sparse_feature_selection_rows)
    overlap_frame = _build_feature_module_overlap_frame(
        sparse_feature_selection=sparse_feature_selection_frame,
        stable_modules=stable_modules,
        enabled=config.baselines.compute_feature_module_overlap,
    )
    comparison_path = baseline_root / "baseline_comparison.parquet"
    dense_selection_path = baseline_root / "dense_model_selection.parquet"
    sparse_feature_selection_path = baseline_root / "sparse_feature_selection.parquet"
    overlap_path = baseline_root / "sparse_feature_module_overlap.parquet"
    write_parquet(comparison_frame, comparison_path)
    write_parquet(dense_selection_frame, dense_selection_path)
    write_parquet(sparse_feature_selection_frame, sparse_feature_selection_path)
    write_parquet(overlap_frame, overlap_path)
    return BaselineArtifacts(
        str(comparison_path),
        str(dense_selection_path),
        str(sparse_feature_selection_path),
        str(overlap_path),
    )


def _read_stable_modules(path: object) -> pd.DataFrame:
    parquet_path = pd.io.common.stringify_path(path)
    if not Path(parquet_path).exists():
        return pd.DataFrame(columns=["stable_module_id", "layer", "stable", "feature_ids", "module_size"])
    return read_parquet(path)


def _run_sentence_embedding_baseline(segments: pd.DataFrame, config: ExperimentConfig) -> list[dict[str, object]]:
    encoder = SentenceEncoder(
        model_name=config.baselines.sentence_embedding_model,
        device=config.backbone.device,
        max_length=config.matching.max_length,
    )
    embeddings = encoder.encode(segments["text"].tolist(), batch_size=config.matching.batch_size)
    rows: list[dict[str, object]] = []
    for proxy_key in config.dataset.proxy_columns.keys():
        metric = _fit_and_score_by_split(
            features=embeddings,
            labels=segments[proxy_key].astype(int).to_numpy(),
            splits=segments["split"].tolist(),
            logistic_c=config.baselines.logistic_c,
            max_iter=config.baselines.max_iter,
        )
        rows.append(
            {
                "method": "sentence_embedding_lr",
                "proxy": proxy_key,
                "selected_layer": None,
                **metric,
            }
        )
    return rows


def _run_dense_residual_baseline(
    segments: pd.DataFrame,
    extraction_root: object,
    proxy_key: str,
    config: ExperimentConfig,
) -> dict[str, dict[str, object]]:
    best = None
    for layer in config.extract.layers:
        manifest = extraction_root / f"residual_pool_manifest_layer_{layer}.parquet"
        vectors = load_residual_matrix(manifest)
        ordered_ids, matrix = matrix_from_vector_frame(vectors)
        aligned = segments.set_index("segment_id").loc[ordered_ids].reset_index()
        metrics = _fit_and_score_by_split(
            features=matrix,
            labels=aligned[proxy_key].astype(int).to_numpy(),
            splits=aligned["split"].tolist(),
            logistic_c=config.baselines.logistic_c,
            max_iter=config.baselines.max_iter,
        )
        candidate = {"layer": layer, **metrics}
        if best is None or candidate["dev_auc"] > best["dev_auc"]:
            best = candidate
    assert best is not None
    return {
        "summary": {
            "method": "dense_residual_lr",
            "proxy": proxy_key,
            "selected_layer": best["layer"],
            "train_auc": best["train_auc"],
            "dev_auc": best["dev_auc"],
            "test_auc": best["test_auc"],
        },
        "selection": {
            "proxy": proxy_key,
            "selected_layer": best["layer"],
            "dev_auc": best["dev_auc"],
        },
    }


def _run_sparse_module_baseline(
    segments: pd.DataFrame,
    stable_modules: pd.DataFrame,
    extraction_root: object,
    proxy_key: str,
    config: ExperimentConfig,
) -> dict[str, object]:
    best = None
    for layer in config.extract.layers:
        top_features = read_parquet(extraction_root / f"segment_top_features_layer_{layer}.parquet")
        score_frame = build_module_score_matrix(stable_modules, top_features, segments, layer)
        feature_columns = [column for column in score_frame.columns if column not in {"segment_id", "split"}]
        if not feature_columns:
            continue
        matrix = score_frame[feature_columns].to_numpy(dtype=float)
        binarized = (matrix > config.baselines.sparse_binarize_threshold).astype(float)
        aligned = segments.set_index("segment_id").loc[score_frame["segment_id"].tolist()].reset_index()
        metric = _fit_and_score_by_split(
            features=binarized,
            labels=aligned[proxy_key].astype(int).to_numpy(),
            splits=aligned["split"].tolist(),
            logistic_c=config.baselines.logistic_c,
            max_iter=config.baselines.max_iter,
        )
        candidate = {"layer": layer, **metric}
        if best is None or candidate["dev_auc"] > best["dev_auc"]:
            best = candidate
    if best is None:
        return {
            "method": "sparse_module_probe",
            "proxy": proxy_key,
            "selected_layer": None,
            "train_auc": np.nan,
            "dev_auc": np.nan,
            "test_auc": np.nan,
        }
    return {
        "method": "sparse_module_probe",
        "proxy": proxy_key,
        "selected_layer": best["layer"],
        "train_auc": best["train_auc"],
        "dev_auc": best["dev_auc"],
        "test_auc": best["test_auc"],
    }


def _run_sparse_individual_feature_baseline(
    segments: pd.DataFrame,
    extraction_root: object,
    proxy_key: str,
    config: ExperimentConfig,
) -> dict[str, dict[str, object]]:
    best = None
    for layer in config.extract.layers:
        top_features = read_parquet(extraction_root / f"segment_top_features_layer_{layer}.parquet")
        feature_matrix = _build_individual_feature_matrix(top_features, segments)
        if feature_matrix["matrix"].shape[1] == 0:
            continue
        aligned = segments.set_index("segment_id").loc[feature_matrix["segment_ids"]].reset_index()
        labels = aligned[proxy_key].astype(int).to_numpy()
        splits = aligned["split"].tolist()
        selected_ids = _select_top_individual_feature_ids(
            feature_ids=feature_matrix["feature_ids"],
            matrix=feature_matrix["matrix"],
            labels=labels,
            splits=splits,
            top_k=config.baselines.individual_feature_top_k,
        )
        if not selected_ids:
            continue
        feature_index = {
            feature_id: column_index
            for column_index, feature_id in enumerate(feature_matrix["feature_ids"])
        }
        selected_columns = [feature_index[feature_id] for feature_id in selected_ids]
        selected_matrix = feature_matrix["matrix"][:, selected_columns]
        metric = _fit_and_score_by_split(
            features=selected_matrix,
            labels=labels,
            splits=splits,
            logistic_c=config.baselines.logistic_c,
            max_iter=config.baselines.max_iter,
        )
        candidate = {
            "layer": layer,
            "selected_feature_ids": selected_ids,
            **metric,
        }
        if best is None or candidate["dev_auc"] > best["dev_auc"]:
            best = candidate
    if best is None:
        return {
            "summary": {
                "method": "sparse_individual_feature_probe",
                "proxy": proxy_key,
                "selected_layer": None,
                "train_auc": np.nan,
                "dev_auc": np.nan,
                "test_auc": np.nan,
            },
            "selection": {
                "proxy": proxy_key,
                "selected_layer": None,
                "selected_feature_ids": [],
                "selected_feature_count": 0,
                "dev_auc": np.nan,
            },
        }
    return {
        "summary": {
            "method": "sparse_individual_feature_probe",
            "proxy": proxy_key,
            "selected_layer": best["layer"],
            "train_auc": best["train_auc"],
            "dev_auc": best["dev_auc"],
            "test_auc": best["test_auc"],
        },
        "selection": {
            "proxy": proxy_key,
            "selected_layer": best["layer"],
            "selected_feature_ids": best["selected_feature_ids"],
            "selected_feature_count": len(best["selected_feature_ids"]),
            "dev_auc": best["dev_auc"],
        },
    }


def _build_individual_feature_matrix(
    top_features: pd.DataFrame,
    segments: pd.DataFrame,
) -> dict[str, object]:
    grouped = (
        top_features.groupby(["segment_id", "feature_id"])["pooled_activation"]
        .max()
        .reset_index()
    )
    if grouped.empty:
        return {"segment_ids": [], "feature_ids": [], "matrix": np.zeros((0, 0), dtype=np.float32)}
    wide = (
        grouped.pivot(index="segment_id", columns="feature_id", values="pooled_activation")
        .fillna(0.0)
        .sort_index(axis=1)
    )
    ordered_segments = segments[["segment_id", "split"]].drop_duplicates().set_index("segment_id")
    aligned = ordered_segments.join(wide, how="left").fillna(0.0).reset_index()
    feature_ids = [int(column) for column in wide.columns.tolist()]
    matrix = aligned[feature_ids].to_numpy(dtype=np.float32)
    return {
        "segment_ids": aligned["segment_id"].tolist(),
        "feature_ids": feature_ids,
        "matrix": matrix,
    }


def _select_top_individual_feature_ids(
    feature_ids: list[int],
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: list[str],
    top_k: int,
) -> list[int]:
    if not feature_ids:
        return []
    matrix = np.asarray(matrix, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    split_array = np.asarray(splits)
    train_mask = split_array == "train"
    train_labels = labels[train_mask]
    train_matrix = matrix[train_mask]
    if train_matrix.size == 0 or np.unique(train_labels).size < 2:
        return []

    ranked: list[tuple[float, float, int]] = []
    for column_index, feature_id in enumerate(feature_ids):
        values = train_matrix[:, column_index]
        if np.allclose(values, values[0]):
            score = 0.5
        else:
            auc = roc_auc_score(train_labels, values)
            score = max(float(auc), float(1.0 - auc))
        mean_gap = float(
            values[train_labels == 1].mean() - values[train_labels == 0].mean()
        )
        ranked.append((score, abs(mean_gap), int(feature_id)))
    ranked.sort(reverse=True)
    return [feature_id for _, _, feature_id in ranked[:top_k]]


def _build_feature_module_overlap_frame(
    sparse_feature_selection: pd.DataFrame,
    stable_modules: pd.DataFrame,
    enabled: bool,
) -> pd.DataFrame:
    if not enabled or sparse_feature_selection.empty or stable_modules.empty:
        return pd.DataFrame(
            columns=[
                "proxy",
                "selected_layer",
                "stable_module_id",
                "module_layer",
                "module_size",
                "selected_feature_count",
                "overlap_count",
                "jaccard",
                "module_coverage",
                "probe_coverage",
            ]
        )

    stable_only = stable_modules.loc[stable_modules["stable"]].copy()
    rows: list[dict[str, object]] = []
    for selection in sparse_feature_selection.itertuples(index=False):
        if selection.selected_layer is None or not selection.selected_feature_ids:
            continue
        selected_set = {int(feature_id) for feature_id in selection.selected_feature_ids}
        layer_modules = stable_only.loc[stable_only["layer"] == int(selection.selected_layer)].copy()
        for module in layer_modules.itertuples(index=False):
            module_set = {int(feature_id) for feature_id in module.feature_ids}
            overlap = selected_set & module_set
            union = selected_set | module_set
            rows.append(
                {
                    "proxy": selection.proxy,
                    "selected_layer": int(selection.selected_layer),
                    "stable_module_id": module.stable_module_id,
                    "module_layer": int(module.layer),
                    "module_size": int(module.module_size),
                    "selected_feature_count": int(selection.selected_feature_count),
                    "overlap_count": len(overlap),
                    "jaccard": (len(overlap) / len(union)) if union else 0.0,
                    "module_coverage": (len(overlap) / len(module_set)) if module_set else 0.0,
                    "probe_coverage": (len(overlap) / len(selected_set)) if selected_set else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _fit_and_score_by_split(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[str],
    logistic_c: float,
    max_iter: int,
) -> dict[str, float]:
    features = np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    split_array = np.asarray(splits)
    train_mask = split_array == "train"
    dev_mask = split_array == "dev"
    test_mask = split_array == "test"
    model = LogisticRegression(C=logistic_c, max_iter=max_iter, solver="liblinear")
    model.fit(features[train_mask], labels[train_mask])
    output = {}
    for name, mask in [("train", train_mask), ("dev", dev_mask), ("test", test_mask)]:
        if labels[mask].size == 0 or np.unique(labels[mask]).size < 2:
            output[f"{name}_auc"] = np.nan
            continue
        probs = model.predict_proba(features[mask])[:, 1]
        output[f"{name}_auc"] = roc_auc_score(labels[mask], probs)
    return output
