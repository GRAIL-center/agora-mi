from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import rankdata


def quick_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    # Mann-Whitney U equivalent for ROC-AUC without sklearn dependency.
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def length_stats(lengths_a: np.ndarray, lengths_b: np.ndarray) -> dict[str, float]:
    a = np.asarray(lengths_a, dtype=np.float64)
    b = np.asarray(lengths_b, dtype=np.float64)
    return {
        "mean_a": float(a.mean()) if a.size else float("nan"),
        "var_a": float(a.var(ddof=1)) if a.size > 1 else float("nan"),
        "mean_b": float(b.mean()) if b.size else float("nan"),
        "var_b": float(b.var(ddof=1)) if b.size > 1 else float("nan"),
        "mean_diff": float(a.mean() - b.mean()) if a.size and b.size else float("nan"),
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def distinct_n(texts: list[str], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    grams: Counter[tuple[str, ...]] = Counter()
    total = 0
    for t in texts:
        toks = t.split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            grams[tuple(toks[i : i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return float(len(grams) / total)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    if y.size == 0 or s.size == 0 or y.shape[0] != s.shape[0] or n_pos == 0:
        return float("nan")
    order = np.argsort(s)[::-1]
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float((precision * ranked).sum() / n_pos)


def precision_at_k_binary(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or s.size == 0 or y.shape[0] != s.shape[0] or k <= 0:
        return float("nan")
    topk = min(int(k), y.size)
    order = np.argsort(s)[::-1][:topk]
    return float(y[order].sum() / topk)


def recall_at_k_binary(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    if y.size == 0 or s.size == 0 or y.shape[0] != s.shape[0] or k <= 0 or n_pos == 0:
        return float("nan")
    topk = min(int(k), y.size)
    order = np.argsort(s)[::-1][:topk]
    return float(y[order].sum() / n_pos)


def first_relevant_rank_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or s.size == 0 or y.shape[0] != s.shape[0]:
        return float("nan")
    order = np.argsort(s)[::-1]
    ranked = y[order]
    hits = np.flatnonzero(ranked == 1)
    if hits.size == 0:
        return float("nan")
    return float(hits[0] + 1)


def ndcg_at_k_binary(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or s.size == 0 or y.shape[0] != s.shape[0] or k <= 0:
        return float("nan")
    topk = min(int(k), y.size)
    order = np.argsort(s)[::-1][:topk]
    ranked = y[order].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, topk + 2))
    dcg = float(((2.0**ranked - 1.0) * discounts).sum())
    ideal = np.sort(y.astype(np.float64))[::-1][:topk]
    idcg = float(((2.0**ideal - 1.0) * discounts).sum())
    if idcg == 0.0:
        return float("nan")
    return float(dcg / idcg)


def reciprocal_rank_from_order(relevant_mask: np.ndarray) -> float:
    hits = np.flatnonzero(np.asarray(relevant_mask).astype(bool))
    if hits.size == 0:
        return 0.0
    return float(1.0 / float(hits[0] + 1))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
