from __future__ import annotations

import torch

from sae.load_sae import LoadedSAE


def encode_features(sae: LoadedSAE, activations: torch.Tensor) -> torch.Tensor:
    return sae.encode(activations)


def topk_features(features: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"features must be [batch, d_sae], got shape {tuple(features.shape)}")
    vals, idx = torch.topk(features, k=min(k, features.size(1)), dim=1)
    return idx, vals
