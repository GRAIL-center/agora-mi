from __future__ import annotations

import numpy as np


def percentile_interval(
    values: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - float(confidence)
    low = float(np.quantile(arr, alpha / 2.0))
    high = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return low, high


def bootstrap_ci(
    values: np.ndarray | list[float],
    B: int,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=np.float64)
    n = arr.size
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    alpha = 1.0 - float(confidence)
    return bootstrap_ci(values, B=int(n_boot), alpha=alpha, seed=seed)


def paired_bootstrap_metric(
    labels: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    metric_fn,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or s.size == 0 or y.size != s.size:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    estimate = float(metric_fn(y, s))
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, y.size, size=y.size)
        stats[i] = float(metric_fn(y[idx], s[idx]))
    ci_low, ci_high = percentile_interval(stats, confidence=float(confidence))
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def paired_bootstrap_metric_delta(
    labels: np.ndarray | list[int],
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    *,
    metric_fn,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if y.size == 0 or a.size == 0 or b.size == 0 or y.size != a.size or y.size != b.size:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    estimate = float(metric_fn(y, a) - metric_fn(y, b))
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, y.size, size=y.size)
        stats[i] = float(metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx]))
    ci_low, ci_high = percentile_interval(stats, confidence=float(confidence))
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
