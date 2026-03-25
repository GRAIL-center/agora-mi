from __future__ import annotations

import numpy as np


def _cluster_index_map(cluster_ids: np.ndarray) -> dict[str, np.ndarray]:
    mapping: dict[str, list[int]] = {}
    for idx, cluster_id in enumerate(cluster_ids.astype(str).tolist()):
        mapping.setdefault(cluster_id, []).append(idx)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in mapping.items()}


def cluster_permutation_pvalues(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    *,
    pos_cluster_ids: np.ndarray,
    neg_cluster_ids: np.ndarray,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    if features_pos.ndim != 2 or features_neg.ndim != 2:
        raise ValueError("Feature arrays must be 2D.")
    pooled = np.concatenate([features_pos, features_neg], axis=0)
    pooled_cluster_ids = np.concatenate([pos_cluster_ids.astype(str), neg_cluster_ids.astype(str)], axis=0)
    observed = np.abs(features_pos.mean(axis=0) - features_neg.mean(axis=0))
    cluster_map = _cluster_index_map(pooled_cluster_ids)
    unique_clusters = np.array(sorted(cluster_map.keys()), dtype=object)
    n_pos_clusters = len(np.unique(pos_cluster_ids.astype(str)))
    rng = np.random.default_rng(seed)
    ge_count = np.zeros(pooled.shape[1], dtype=np.int64)

    for _ in range(n_perm):
        perm_clusters = rng.permutation(unique_clusters)
        pos_clusters = set(perm_clusters[:n_pos_clusters].tolist())
        pos_indices = np.concatenate([cluster_map[cid] for cid in unique_clusters if cid in pos_clusters], axis=0)
        neg_indices = np.concatenate([cluster_map[cid] for cid in unique_clusters if cid not in pos_clusters], axis=0)
        perm_delta = np.abs(pooled[pos_indices].mean(axis=0) - pooled[neg_indices].mean(axis=0))
        ge_count += (perm_delta >= observed).astype(np.int64)
    return (ge_count + 1).astype(np.float64) / (n_perm + 1)


def cluster_bootstrap_selection_frequency(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    *,
    pos_cluster_ids: np.ndarray,
    neg_cluster_ids: np.ndarray,
    topk: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    pos_clusters = np.unique(pos_cluster_ids.astype(str))
    neg_clusters = np.unique(neg_cluster_ids.astype(str))
    pos_map = _cluster_index_map(pos_cluster_ids.astype(str))
    neg_map = _cluster_index_map(neg_cluster_ids.astype(str))
    rng = np.random.default_rng(seed)
    counts = np.zeros(features_pos.shape[1], dtype=np.int64)

    for _ in range(n_boot):
        pos_sample = rng.choice(pos_clusters, size=len(pos_clusters), replace=True)
        neg_sample = rng.choice(neg_clusters, size=len(neg_clusters), replace=True)
        pos_indices = np.concatenate([pos_map[cid] for cid in pos_sample], axis=0)
        neg_indices = np.concatenate([neg_map[cid] for cid in neg_sample], axis=0)
        delta = features_pos[pos_indices].mean(axis=0) - features_neg[neg_indices].mean(axis=0)
        selected = np.argsort(delta)[::-1][:topk]
        counts[selected] += 1
    return counts.astype(np.float64) / max(n_boot, 1)
