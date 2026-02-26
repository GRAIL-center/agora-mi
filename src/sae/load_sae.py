from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class LoadedSAE:
    backend: str
    sae: Any
    device: torch.device
    d_in: int
    d_sae: int

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        if self.backend == "sae_lens":
            return self.sae.encode(activations)
        if self.backend == "npz":
            x = activations.to(torch.float32)
            pre = (x - self.sae["b_dec"]) @ self.sae["W_enc"] + self.sae["b_enc"]
            return torch.relu(pre)
        raise RuntimeError(f"Unknown SAE backend: {self.backend}")

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if self.backend == "sae_lens":
            return self.sae.decode(features)
        if self.backend == "npz":
            f = features.to(torch.float32)
            return f @ self.sae["W_dec"] + self.sae["b_dec"]
        raise RuntimeError(f"Unknown SAE backend: {self.backend}")


def _load_npz_sae(npz_path: str | Path, device: torch.device) -> LoadedSAE:
    arr = np.load(npz_path)
    if "W_enc" not in arr.files or "W_dec" not in arr.files:
        raise ValueError(f"NPZ SAE must contain W_enc and W_dec arrays: {npz_path}")
    w_enc = np.asarray(arr["W_enc"], dtype=np.float32)
    w_dec = np.asarray(arr["W_dec"], dtype=np.float32)

    # Normalize to [d_in, d_sae] and [d_sae, d_in]
    if w_enc.shape[0] == w_dec.shape[1]:
        pass
    elif w_enc.shape[1] == w_dec.shape[1]:
        w_enc = w_enc.T
    else:
        raise ValueError(f"Incompatible NPZ SAE shapes: W_enc={w_enc.shape}, W_dec={w_dec.shape}")
    d_in, d_sae = w_enc.shape
    if w_dec.shape == (d_in, d_sae):
        w_dec = w_dec.T
    if w_dec.shape != (d_sae, d_in):
        raise ValueError(f"Unexpected W_dec shape after normalization: {w_dec.shape}")

    b_enc = np.asarray(arr["b_enc"], dtype=np.float32) if "b_enc" in arr.files else np.zeros(d_sae, dtype=np.float32)
    b_dec = np.asarray(arr["b_dec"], dtype=np.float32) if "b_dec" in arr.files else np.zeros(d_in, dtype=np.float32)

    sae_t = {
        "W_enc": torch.from_numpy(w_enc).to(device),
        "W_dec": torch.from_numpy(w_dec).to(device),
        "b_enc": torch.from_numpy(b_enc).to(device),
        "b_dec": torch.from_numpy(b_dec).to(device),
    }
    return LoadedSAE(backend="npz", sae=sae_t, device=device, d_in=d_in, d_sae=d_sae)


def load_sae(
    *,
    release: str,
    sae_id: str,
    device: torch.device,
    npz_path: str | None = None,
) -> LoadedSAE:
    if npz_path:
        return _load_npz_sae(npz_path, device)

    try:
        from sae_lens import SAE
    except Exception as exc:
        raise RuntimeError(
            "sae-lens is required for Gemma Scope loading. Install requirements.txt or pass --sae_npz."
        ) from exc

    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    d_in = getattr(sae, "d_in", None)
    d_sae = getattr(sae, "d_sae", None)
    if d_in is None or d_sae is None:
        cfg = getattr(sae, "cfg", None)
        if cfg is not None:
            d_in = getattr(cfg, "d_in", d_in)
            d_sae = getattr(cfg, "d_sae", d_sae)
    if d_in is None or d_sae is None:
        # Fallback for newer sae-lens objects where dims are only implied by parameters.
        w_enc = getattr(sae, "W_enc", None)
        if w_enc is not None and hasattr(w_enc, "shape") and len(w_enc.shape) == 2:
            d_in = int(w_enc.shape[0]) if d_in is None else int(d_in)
            d_sae = int(w_enc.shape[1]) if d_sae is None else int(d_sae)
    if d_in is None or d_sae is None:
        raise RuntimeError("Unable to infer SAE dimensions from sae-lens object.")
    d_in = int(d_in)
    d_sae = int(d_sae)
    return LoadedSAE(backend="sae_lens", sae=sae, device=device, d_in=d_in, d_sae=d_sae)


def load_sae_for_layer(
    run_config: dict[str, Any],
    *,
    layer: int,
    device: torch.device,
    npz_path: str | None = None,
) -> LoadedSAE:
    release = str(run_config["sae_release"])
    template = str(run_config["sae_id_template"])
    sae_id = template.replace("{L}", str(layer))
    return load_sae(release=release, sae_id=sae_id, device=device, npz_path=npz_path)
