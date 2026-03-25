from __future__ import annotations

import torch


def clamp_features(
    features: torch.Tensor,
    feature_ids: list[int],
    q95: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    out = features.clone()
    if not feature_ids:
        return out
    idx = torch.tensor(feature_ids, dtype=torch.long, device=out.device)
    thresh = alpha * q95[idx]
    out[:, idx] = torch.minimum(out[:, idx], thresh.unsqueeze(0))
    return out
