from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def compute_effect_size_d(
    mean_a: np.ndarray,
    std_a: np.ndarray,
    n_a: int,
    mean_b: np.ndarray,
    std_b: np.ndarray,
    n_b: int,
) -> np.ndarray:
    # Cohen's d with pooled standard deviation.
    pooled_var = ((n_a - 1) * (std_a**2) + (n_b - 1) * (std_b**2)) / max((n_a + n_b - 2), 1)
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-12))
    return (mean_a - mean_b) / pooled_std


def _vectorized_permutation_pvalues(
    features_safe: np.ndarray,
    features_innov: np.ndarray,
    observed_delta: np.ndarray,
    N: int = 10000,
    seed: int = 0,
) -> np.ndarray:
    """Vectorized permutation test across all features simultaneously.

    For each permutation, randomly shuffles group labels and computes
    |mean_a - mean_b| for every feature at once. Returns two-sided
    p-values with add-one smoothing.
    """
    n_a = features_safe.shape[0]
    pooled = np.concatenate([features_safe, features_innov], axis=0)  # (n_a+n_b, D)
    n_total = pooled.shape[0]
    abs_observed = np.abs(observed_delta)

    rng = np.random.default_rng(seed)
    ge_count = np.zeros(features_safe.shape[1], dtype=np.int64)

    for i in range(N):
        perm = rng.permutation(n_total)
        perm_a = pooled[perm[:n_a]]
        perm_b = pooled[perm[n_a:]]
        perm_delta = np.abs(perm_a.mean(axis=0) - perm_b.mean(axis=0))
        ge_count += (perm_delta >= abs_observed).astype(np.int64)
        if (i + 1) % 2000 == 0:
            logging.info("  permutation %d / %d", i + 1, N)

    # add-one smoothing
    return (ge_count + 1).astype(np.float64) / (N + 1)


def polarization_table(
    features_safe: np.ndarray,
    features_innov: np.ndarray,
    *,
    perm_N: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute per-feature polarization statistics.

    Args:
        features_safe: (n_safe, D) array of SAE feature activations for Dsafe.
        features_innov: (n_innov, D) array of SAE feature activations for Dinnov.
        perm_N: Number of permutations for per-feature p-values. 0 = skip.
        seed: RNG seed for permutation test reproducibility.
    """
    if features_safe.ndim != 2 or features_innov.ndim != 2:
        raise ValueError("features_safe and features_innov must be 2D arrays.")
    if features_safe.shape[1] != features_innov.shape[1]:
        raise ValueError("safe and innov feature arrays must have same feature dimension.")

    mean_safe = features_safe.mean(axis=0)
    mean_innov = features_innov.mean(axis=0)
    std_safe = features_safe.std(axis=0, ddof=1)
    std_innov = features_innov.std(axis=0, ddof=1)
    delta = mean_safe - mean_innov
    d = compute_effect_size_d(
        mean_safe,
        std_safe,
        features_safe.shape[0],
        mean_innov,
        std_innov,
        features_innov.shape[0],
    )

    data = {
        "feature_id": np.arange(features_safe.shape[1], dtype=np.int64),
        "delta": delta,
        "mean_safe": mean_safe,
        "mean_innov": mean_innov,
        "std_safe": std_safe,
        "std_innov": std_innov,
        "effect_size_d": d,
    }

    if perm_N > 0:
        logging.info(
            "Running vectorized permutation test (N=%d) across %d features...",
            perm_N, features_safe.shape[1],
        )
        p_values = _vectorized_permutation_pvalues(
            features_safe, features_innov, delta, N=perm_N, seed=seed,
        )
        data["p_value"] = p_values

    df = pd.DataFrame(data)
    return df.sort_values("delta", ascending=False).reset_index(drop=True)


def topk_feature_lists(
    delta_df: pd.DataFrame,
    *,
    topk: int,
    delta_thresh: float | None = None,
) -> tuple[list[int], list[int]]:
    if delta_thresh is not None:
        safe_ids = delta_df.loc[delta_df["delta"] >= delta_thresh, "feature_id"].astype(int).tolist()
        innov_ids = delta_df.loc[delta_df["delta"] <= -delta_thresh, "feature_id"].astype(int).tolist()
        return safe_ids, innov_ids

    safe_ids = delta_df.sort_values("delta", ascending=False)["feature_id"].astype(int).head(topk).tolist()
    innov_ids = delta_df.sort_values("delta", ascending=True)["feature_id"].astype(int).head(topk).tolist()
    return safe_ids, innov_ids
