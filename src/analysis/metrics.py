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
