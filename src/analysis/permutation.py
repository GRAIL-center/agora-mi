from __future__ import annotations

import numpy as np


def permutation_test(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    N: int,
    seed: int = 0,
) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")

    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)

    ge = 0
    for _ in range(N):
        perm = rng.permutation(pooled)
        diff = abs(perm[:n_a].mean() - perm[n_a:].mean())
        if diff >= observed:
            ge += 1
    # add-one smoothing
    return float((ge + 1) / (N + 1))


def paired_permutation_sign_flip_test(
    deltas: np.ndarray | list[float],
    N: int,
    seed: int = 0,
) -> float:
    """
    Paired permutation test using random sign flips.
    Null hypothesis: mean(deltas) == 0.
    """
    d = np.asarray(deltas, dtype=np.float64)
    if d.size == 0:
        return float("nan")

    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(N):
        signs = rng.choice([-1.0, 1.0], size=d.size)
        stat = abs((d * signs).mean())
        if stat >= observed:
            ge += 1
    return float((ge + 1) / (N + 1))
