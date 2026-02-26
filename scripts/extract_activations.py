from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from _common import ensure_dir, read_config, setup_logging
from data.io import read_jsonl
from model.hooks import capture_residual_stream
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt


def _infer_split_name(path: str) -> str:
    stem = Path(path).stem.lower()
    if "train" in stem:
        return "train"
    if "test" in stem:
        return "test"
    return "dev"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--split_name", default=None)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    setup_logging("extract_activations")
    cfg = read_config(args.config)
    split_name = args.split_name or _infer_split_name(args.input_jsonl)

    rows = read_jsonl(args.input_jsonl)
    if not rows:
        raise ValueError(f"No rows in input jsonl: {args.input_jsonl}")

    p_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(p_cfg["template_v1"])
    prompts = [render_prompt(template, str(r["text"])) for r in rows]
    ids = [str(r.get("id", i)) for i, r in enumerate(rows)]

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    batch_size = int(cfg.get("batch_size", 4))
    max_length = int(cfg.get("max_length", 1024))

    all_h = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = capture_residual_stream(model, layer=args.layer, token_index="last", inputs=enc)
        all_h.append(h.cpu())

    acts = torch.cat(all_h, dim=0)
    out_dir = ensure_dir(args.out_dir)
    act_path = out_dir / f"activations_{split_name}_layer{args.layer}.pt"
    ids_path = out_dir / f"ids_{split_name}.json"
    torch.save(acts, act_path)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    logging.info("Saved activations: %s shape=%s", act_path, tuple(acts.shape))
    logging.info("Saved ids: %s count=%d", ids_path, len(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
