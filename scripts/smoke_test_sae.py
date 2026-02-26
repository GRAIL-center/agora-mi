from __future__ import annotations

import argparse
import logging

import torch

from _common import read_config, setup_logging
from model.hooks import capture_residual_stream
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt
from sae.encode import topk_features
from sae.load_sae import load_sae_for_layer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--sae_npz", default=None)
    args = parser.parse_args()

    setup_logging("smoke_test_sae")
    cfg = read_config(args.config)
    layer = int(args.layer if args.layer is not None else cfg.get("layers", [0])[0])

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    p_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    prompt = render_prompt(
        str(p_cfg["template_v1"]),
        "The bill imposes external audits and strict compliance requirements.",
    )

    enc = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg.get("max_length", 1024)),
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        h = capture_residual_stream(model, layer=layer, token_index="last", inputs=enc)

    sae = load_sae_for_layer(cfg, layer=layer, device=device, npz_path=args.sae_npz)
    with torch.no_grad():
        f = sae.encode(h.to(torch.float32))
        nnz = int((f[0] > 0).sum().item())
        idx, vals = topk_features(f, k=10)

    logging.info("Layer %d activation shape: %s", layer, tuple(h.shape))
    logging.info("SAE dims d_in=%d d_sae=%d", sae.d_in, sae.d_sae)
    print(f"nonzero features: {nnz}")
    print("top-10 feature indices + values:")
    for i in range(idx.size(1)):
        print(f"{i+1}. feature={int(idx[0, i].item())} value={float(vals[0, i].item()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
