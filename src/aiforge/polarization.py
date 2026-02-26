from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=np.float64)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        running_min = min(running_min, candidate)
        adjusted[i] = running_min
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def compute_polarization_table(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatched rows: features={features.shape[0]}, labels={labels.shape[0]}"
        )

    labels = labels.astype(str)
    safe_mask = labels == "safe"
    innov_mask = labels == "innov"
    if not np.any(safe_mask):
        raise ValueError("No safe examples found in labels.")
    if not np.any(innov_mask):
        raise ValueError("No innov examples found in labels.")

    safe_feats = features[safe_mask]
    innov_feats = features[innov_mask]
    mean_safe = safe_feats.mean(axis=0)
    mean_innov = innov_feats.mean(axis=0)
    delta = mean_safe - mean_innov

    # Welch t-test per feature to provide a quick significance signal.
    t_stat, p_values = stats.ttest_ind(
        safe_feats,
        innov_feats,
        equal_var=False,
        axis=0,
        nan_policy="omit",
    )
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    q_values = _bh_fdr(p_values)

    df = pd.DataFrame(
        {
            "feature_index": np.arange(features.shape[1], dtype=np.int64),
            "mean_safe": mean_safe,
            "mean_innov": mean_innov,
            "delta": delta,
            "abs_delta": np.abs(delta),
            "t_stat": t_stat,
            "p_value": p_values,
            "q_value_bh": q_values,
        }
    )
    return df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
