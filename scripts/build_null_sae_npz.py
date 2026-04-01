from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, setup_logging
from sae.load_sae import load_sae_for_layer


def _row_norms(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1.0e-8)


def _tied_encoder_from_decoder(w_dec: np.ndarray) -> np.ndarray:
    row_norm_sq = np.square(_row_norms(w_dec)).reshape(-1)
    return (w_dec / row_norm_sq[:, None]).T.astype(np.float32)


def _random_dictionary_like(w_dec: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_w_dec = rng.standard_normal(size=w_dec.shape).astype(np.float32)
    target_norms = _row_norms(w_dec)
    random_norms = _row_norms(random_w_dec)
    return (random_w_dec / random_norms) * target_norms


def _build_null_payload(
    *,
    params: dict[str, np.ndarray],
    mode: str,
    seed: int,
    encoder_blend_alpha: float,
) -> dict[str, np.ndarray]:
    w_enc = np.asarray(params["W_enc"], dtype=np.float32)
    w_dec = np.asarray(params["W_dec"], dtype=np.float32)
    b_enc = np.asarray(params["b_enc"], dtype=np.float32)
    b_dec = np.asarray(params["b_dec"], dtype=np.float32)
    tied_w_enc = _tied_encoder_from_decoder(w_dec)

    if mode == "soft_frozen_decoder":
        mode = "encoder_blend"

    if mode == "tied_encoder_null":
        out_w_dec = w_dec
        out_w_enc = tied_w_enc
        out_b_enc = np.zeros_like(b_enc, dtype=np.float32)
        out_b_dec = b_dec
    elif mode == "random_dictionary":
        out_w_dec = _random_dictionary_like(w_dec, seed=seed)
        out_w_enc = _tied_encoder_from_decoder(out_w_dec)
        out_b_enc = np.zeros_like(b_enc, dtype=np.float32)
        out_b_dec = b_dec
    elif mode == "encoder_blend":
        alpha = float(np.clip(encoder_blend_alpha, 0.0, 1.0))
        out_w_dec = w_dec
        out_w_enc = (alpha * w_enc + (1.0 - alpha) * tied_w_enc).astype(np.float32)
        out_b_enc = (alpha * b_enc).astype(np.float32)
        out_b_dec = b_dec
    else:
        raise ValueError(f"Unsupported null SAE mode: {mode}")

    return {
        "W_enc": out_w_enc.astype(np.float32),
        "W_dec": out_w_dec.astype(np.float32),
        "b_enc": out_b_enc.astype(np.float32),
        "b_dec": out_b_dec.astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_mech_interp.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument(
        "--mode",
        choices=["tied_encoder_null", "random_dictionary", "encoder_blend", "soft_frozen_decoder"],
        required=True,
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder_blend_alpha", type=float, default=0.5)
    args = parser.parse_args()

    setup_logging("build_null_sae_npz")
    cfg = read_config(args.config)
    device_name = str(cfg.get("device", "cuda"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    sae = load_sae_for_layer(cfg, layer=int(args.layer), site=str(args.site), device=device)
    params = sae.to_numpy()
    payload = _build_null_payload(
        params=params,
        mode=str(args.mode),
        seed=int(args.seed),
        encoder_blend_alpha=float(args.encoder_blend_alpha),
    )

    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)
    np.savez(out_path, **payload)
    meta = {
        "mode": str(args.mode),
        "layer": int(args.layer),
        "site": str(args.site),
        "seed": int(args.seed),
        "encoder_blend_alpha": float(args.encoder_blend_alpha),
        "source_config": str(args.config),
        "shape_W_enc": list(payload["W_enc"].shape),
        "shape_W_dec": list(payload["W_dec"].shape),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


