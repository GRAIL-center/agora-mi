from __future__ import annotations

import torch

from sae.load_sae import LoadedSAE


def reconstruct_activations(sae: LoadedSAE, features: torch.Tensor) -> torch.Tensor:
    return sae.decode(features)
