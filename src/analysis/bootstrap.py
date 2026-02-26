from __future__ import annotations

import numpy as np


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
