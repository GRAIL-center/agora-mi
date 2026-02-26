from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from _common import read_config, setup_logging
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    args = parser.parse_args()

    setup_logging("smoke_test_model")
    cfg = read_config(args.config)
    model_bundle = load_model_bundle(cfg)
    model = model_bundle.model
    tokenizer = model_bundle.tokenizer
    device = model_bundle.device

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])
    sample_text = "The policy establishes government grants to support AI research and pilot programs."
    prompt = render_prompt(template, sample_text)

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg.get("max_length", 1024)))
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded, use_cache=False)
        logits = out.logits
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        top_vals, top_idx = torch.topk(probs[0], k=5)

    logging.info("logits shape: %s", tuple(logits.shape))
    print(f"logits shape: {tuple(logits.shape)}")
    print("top-5 next-token candidates after Answer:")
    for rank, (tok_id, score) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), start=1):
        tok = tokenizer.decode([tok_id])
        print(f"{rank}. id={tok_id} token={tok!r} prob={score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
