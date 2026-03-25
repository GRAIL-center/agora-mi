from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class SAEBackend(Protocol):
    d_model: int
    d_features: int

    def encode(self, residuals: np.ndarray) -> np.ndarray:
        ...

    def decode(self, features: np.ndarray) -> np.ndarray:
        ...

    def ablate_features(self, residuals: np.ndarray, feature_idx: np.ndarray) -> np.ndarray:
        ...


@dataclass
class NpzSAE:
    w_enc: np.ndarray
    b_enc: np.ndarray
    w_dec: np.ndarray
    b_dec: np.ndarray

    @property
    def d_model(self) -> int:
        return int(self.b_dec.shape[0])

    @property
    def d_features(self) -> int:
        return int(self.b_enc.shape[0])

    @classmethod
    def load(cls, path: str) -> "NpzSAE":
        arr = np.load(path)
        required = {"W_enc", "W_dec"}
        missing = required.difference(set(arr.files))
        if missing:
            raise ValueError(f"Missing SAE arrays in {path}: {sorted(missing)}")

        w_enc = np.asarray(arr["W_enc"], dtype=np.float32)
        w_dec = np.asarray(arr["W_dec"], dtype=np.float32)

        # Infer dimensions and normalize orientation.
        enc_shapes = w_enc.shape
        dec_shapes = w_dec.shape
        if len(enc_shapes) != 2 or len(dec_shapes) != 2:
            raise ValueError("W_enc and W_dec must be rank-2 arrays")

        if w_enc.shape[0] == w_dec.shape[1]:
            d_model = w_enc.shape[0]
            d_feat = w_enc.shape[1]
            w_enc_norm = w_enc
            if w_dec.shape != (d_feat, d_model):
                raise ValueError(
                    f"Incompatible W_dec shape {w_dec.shape}; expected {(d_feat, d_model)}"
                )
            w_dec_norm = w_dec
        elif w_enc.shape[1] == w_dec.shape[1]:
            d_model = w_enc.shape[1]
            d_feat = w_enc.shape[0]
            w_enc_norm = w_enc.T
            if w_dec.shape == (d_feat, d_model):
                w_dec_norm = w_dec
            elif w_dec.shape == (d_model, d_feat):
                w_dec_norm = w_dec.T
            else:
                raise ValueError(
                    f"Incompatible W_dec shape {w_dec.shape} for inferred dims "
                    f"(d_model={d_model}, d_feat={d_feat})"
                )
        else:
            raise ValueError(
                f"Could not infer SAE dimensions from W_enc {w_enc.shape} and W_dec {w_dec.shape}"
            )

        b_enc = np.asarray(arr["b_enc"], dtype=np.float32) if "b_enc" in arr.files else np.zeros(d_feat, dtype=np.float32)
        b_dec = np.asarray(arr["b_dec"], dtype=np.float32) if "b_dec" in arr.files else np.zeros(d_model, dtype=np.float32)
        if b_enc.shape[0] != d_feat:
            raise ValueError(f"b_enc has shape {b_enc.shape}, expected ({d_feat},)")
        if b_dec.shape[0] != d_model:
            raise ValueError(f"b_dec has shape {b_dec.shape}, expected ({d_model},)")
        return cls(w_enc=w_enc_norm, b_enc=b_enc, w_dec=w_dec_norm, b_dec=b_dec)

    def encode(self, residuals: np.ndarray) -> np.ndarray:
        x = np.asarray(residuals, dtype=np.float32)
        pre = (x - self.b_dec) @ self.w_enc + self.b_enc
        return np.maximum(pre, 0.0)

    def decode(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=np.float32)
        return f @ self.w_dec + self.b_dec

    def ablate_features(self, residuals: np.ndarray, feature_idx: np.ndarray) -> np.ndarray:
        x = np.asarray(residuals, dtype=np.float32)
        f = self.encode(x)
        f_ab = f.copy()
        f_ab[:, feature_idx] = 0.0
        delta = self.decode(f_ab - f)
        return x + delta


@dataclass
class SaeLensAdapter:
    sae: object
    d_model: int
    d_features: int

    @classmethod
    def load(cls, release: str, sae_id: str, device: str) -> "SaeLensAdapter":
        try:
            from sae_lens import SAE
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sae-lens is not installed. Install with `pip install '.[sae]'`."
            ) from exc

        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded

        d_model = getattr(sae, "d_in", None)
        d_features = getattr(sae, "d_sae", None)
        if d_model is None or d_features is None:
            raise RuntimeError("Unable to read d_in/d_sae from sae-lens object.")
        return cls(sae=sae, d_model=int(d_model), d_features=int(d_features))

    def encode(self, residuals: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.asarray(residuals, dtype=np.float32)).to(getattr(self.sae, "device", "cpu"))
        with torch.no_grad():
            feats = self.sae.encode(x)
        return feats.detach().cpu().numpy()

    def decode(self, features: np.ndarray) -> np.ndarray:
        import torch

        f = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(getattr(self.sae, "device", "cpu"))
        with torch.no_grad():
            recon = self.sae.decode(f)
        return recon.detach().cpu().numpy()

    def ablate_features(self, residuals: np.ndarray, feature_idx: np.ndarray) -> np.ndarray:
        x = np.asarray(residuals, dtype=np.float32)
        f = self.encode(x)
        f_ab = f.copy()
        f_ab[:, feature_idx] = 0.0
        delta = self.decode(f_ab - f)
        return x + delta


def load_sae_backend(
    *,
    sae_npz: str | None,
    sae_release: str | None,
    sae_id: str | None,
    device: str,
) -> SAEBackend:
    if sae_npz:
        return NpzSAE.load(sae_npz)
    if sae_release and sae_id:
        return SaeLensAdapter.load(release=sae_release, sae_id=sae_id, device=device)
    raise ValueError("Provide either --sae-npz or both --sae-release and --sae-id.")


def save_features_npz(
    *,
    output_path: str,
    features: np.ndarray,
    labels: np.ndarray,
    prompts: np.ndarray | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": features, "labels": labels}
    if prompts is not None:
        payload["prompts"] = prompts
    np.savez_compressed(path, **payload)
