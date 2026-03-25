from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.fdr import benjamini_hochberg
from analysis.polarization import polarization_table, topk_feature_lists
from data.io import read_jsonl
from model.hooks import capture_residual_stream
from model.load_model import load_model_bundle
from model.prompt import build_prompts, load_prompt_config, render_prompt
from sae.encode import encode_features
from sae.load_sae import load_sae_for_layer


def _extract_features(
    *,
    model,
    tokenizer,
    device,
    sae,
    prompts: list[str],
    layer: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    all_feats = []
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
            h = capture_residual_stream(model, layer=layer, token_index="last", inputs=enc)
            f = encode_features(sae, h.to(torch.float32))
        all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--safe_jsonl", default="data/processed/dsafe_dev.jsonl")
    parser.add_argument("--innov_jsonl", default="data/processed/dinnov_dev.jsonl")
    parser.add_argument("--sae_npz", default=None)
    parser.add_argument("--tag", default=None, help="Optional tag suffix for output files, e.g. train/dev/test.")
    parser.add_argument("--perm_N", type=int, default=None, help="Permutations for per-feature p-values. Overrides config perm_N.")
    args = parser.parse_args()

    setup_logging("compute_polarization")
    cfg = read_config(args.config)
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])

    safe_rows = read_jsonl(args.safe_jsonl)
    innov_rows = read_jsonl(args.innov_jsonl)

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device

    use_chat_template = cfg.get("use_chat_template", False)
    safe_prompts = build_prompts(safe_rows, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    innov_prompts = build_prompts(innov_rows, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    if not safe_prompts or not innov_prompts:
        raise ValueError("Safe/innov dev prompts are empty. Check processed dataset paths.")
    sae = load_sae_for_layer(cfg, layer=args.layer, device=device, npz_path=args.sae_npz)
    batch_size = int(cfg.get("batch_size", 4))
    max_length = int(cfg.get("max_length", 1024))

    feats_safe = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=safe_prompts,
        layer=args.layer,
        batch_size=batch_size,
        max_length=max_length,
    )
    feats_innov = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=innov_prompts,
        layer=args.layer,
        batch_size=batch_size,
        max_length=max_length,
    )

    perm_N = args.perm_N if args.perm_N is not None else int(cfg.get("perm_N", 10000))
    seed = int(cfg.get("seed", 0))
    delta_df = polarization_table(feats_safe, feats_innov, perm_N=perm_N, seed=seed)

    # Apply FDR correction if permutation p-values were computed.
    fdr_q = float(cfg.get("fdr_q", 0.05))
    if "p_value" in delta_df.columns:
        fdr_result = benjamini_hochberg(delta_df["p_value"].values, q=fdr_q)
        delta_df["q_value"] = fdr_result["q_values"]
        delta_df["fdr_reject"] = fdr_result["reject"]
        n_survivors = int(delta_df["fdr_reject"].sum())
        logging.info("FDR correction (q<%.2f): %d / %d features survive", fdr_q, n_survivors, len(delta_df))

    topk = int(cfg.get("topk", 64))
    delta_thresh = cfg.get("delta_thresh")
    safe_ids, innov_ids = topk_feature_lists(delta_df, topk=topk, delta_thresh=delta_thresh)

    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "polarization")
    suffix = f"_{args.tag}" if args.tag else ""
    delta_path = out_dir / f"layer{args.layer}{suffix}_delta.csv"
    safe_path = out_dir / f"layer{args.layer}{suffix}_topk_safe.json"
    innov_path = out_dir / f"layer{args.layer}{suffix}_topk_innov.json"
    feats_path = out_dir / f"layer{args.layer}{suffix}_features.npz"

    delta_df.to_csv(delta_path, index=False)
    with safe_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_ids": safe_ids}, f, ensure_ascii=False, indent=2)
    with innov_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_ids": innov_ids}, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        feats_path,
        safe_features=feats_safe,
        innov_features=feats_innov,
        safe_ids=np.array([r.get("id", i) for i, r in enumerate(safe_rows)], dtype=object),
        innov_ids=np.array([r.get("id", i) for i, r in enumerate(innov_rows)], dtype=object),
        safe_text=np.array([r["text"] for r in safe_rows], dtype=object),
        innov_text=np.array([r["text"] for r in innov_rows], dtype=object),
    )

    meta_payload = {
        "delta_csv": str(delta_path),
        "topk_safe_json": str(safe_path),
        "topk_innov_json": str(innov_path),
        "safe_count": len(safe_rows),
        "innov_count": len(innov_rows),
        "perm_N": perm_N,
        "fdr_q": fdr_q,
    }
    if "fdr_reject" in delta_df.columns:
        meta_payload["n_fdr_survivors"] = int(delta_df["fdr_reject"].sum())
    save_with_metadata(
        output_path=out_dir / f"layer{args.layer}{suffix}_run_meta.json",
        payload=meta_payload,
        config={"run": cfg, "args": vars(args)},
    )
    logging.info("Saved: %s", delta_path)
    logging.info("Saved: %s", safe_path)
    logging.info("Saved: %s", innov_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
