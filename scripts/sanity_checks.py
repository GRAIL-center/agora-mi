from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.metrics import length_stats, pearson_corr, quick_roc_auc
from data.io import read_jsonl
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--safe_jsonl", default="data/processed/dsafe_dev.jsonl")
    parser.add_argument("--innov_jsonl", default="data/processed/dinnov_dev.jsonl")
    parser.add_argument("--features_npz", default=None)
    parser.add_argument("--topk_safe_json", default=None)
    parser.add_argument("--topk_innov_json", default=None)
    args = parser.parse_args()

    setup_logging("sanity_checks")
    cfg = read_config(args.config)

    pol_dir = Path(cfg.get("results_dir", "results")) / "polarization"
    features_npz = args.features_npz or str(pol_dir / f"layer{args.layer}_dev_features.npz")
    topk_safe_json = args.topk_safe_json or str(pol_dir / f"layer{args.layer}_topk_safe.json")
    topk_innov_json = args.topk_innov_json or str(pol_dir / f"layer{args.layer}_topk_innov.json")

    arr = np.load(features_npz, allow_pickle=True)
    safe_features = np.asarray(arr["safe_features"], dtype=np.float32)
    innov_features = np.asarray(arr["innov_features"], dtype=np.float32)
    safe_text = np.asarray(arr["safe_text"], dtype=object).tolist()
    innov_text = np.asarray(arr["innov_text"], dtype=object).tolist()

    with Path(topk_safe_json).open("r", encoding="utf-8") as f:
        top_safe = [int(x) for x in json.load(f)["feature_ids"]]
    with Path(topk_innov_json).open("r", encoding="utf-8") as f:
        top_innov = [int(x) for x in json.load(f)["feature_ids"]]
    topk = int(cfg.get("topk", 64))
    top_safe = top_safe[:topk]
    top_innov = top_innov[:topk]

    # 1) Length confound
    safe_rows = read_jsonl(args.safe_jsonl)
    innov_rows = read_jsonl(args.innov_jsonl)
    bundle = load_model_bundle(cfg)
    tokenizer = bundle.tokenizer
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])
    safe_prompts = [render_prompt(template, str(r["text"])) for r in safe_rows]
    innov_prompts = [render_prompt(template, str(r["text"])) for r in innov_rows]
    max_length = int(cfg.get("max_length", 1024))
    safe_lens = np.array(
        [
            len(
                tokenizer(
                    p,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
            )
            for p in safe_prompts
        ],
        dtype=np.int64,
    )
    innov_lens = np.array(
        [
            len(
                tokenizer(
                    p,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
            )
            for p in innov_prompts
        ],
        dtype=np.int64,
    )
    len_stats = length_stats(safe_lens, innov_lens)

    # Correlation between feature activations and length for top polarized features
    corr_items = []
    combined_feats = np.concatenate([safe_features, innov_features], axis=0)
    combined_lens = np.concatenate([safe_lens, innov_lens], axis=0)
    tracked = top_safe[:10] + top_innov[:10]
    seen = set()
    for fid in tracked:
        if fid in seen:
            continue
        seen.add(fid)
        corr = pearson_corr(combined_feats[:, fid], combined_lens)
        corr_items.append({"feature_id": int(fid), "pearson_corr_with_length": float(corr)})

    # 2) Quick separation score / AUC
    safe_score = safe_features[:, top_safe].sum(axis=1) - safe_features[:, top_innov].sum(axis=1)
    innov_score = innov_features[:, top_safe].sum(axis=1) - innov_features[:, top_innov].sum(axis=1)
    y_true = np.concatenate([np.ones_like(safe_score), np.zeros_like(innov_score)])
    scores = np.concatenate([safe_score, innov_score])
    auc = quick_roc_auc(y_true, scores)
    recommendation = None
    if np.isfinite(auc) and auc < 0.60:
        recommendation = "AUC < 0.60: try different layer index or wider SAE."

    # 3) Qualitative top examples
    qual_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "qualitative")
    all_texts = safe_text + innov_text
    all_feats = np.concatenate([safe_features, innov_features], axis=0)
    for fid in top_safe[:5]:
        vals = all_feats[:, fid]
        top_idx = np.argsort(vals)[::-1][:5]
        out_path = qual_dir / f"feature_{fid}_top_examples.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"Feature {fid} top examples\n\n")
            for rank, i in enumerate(top_idx, start=1):
                f.write(f"[{rank}] activation={vals[i]:.6f}\n")
                f.write(str(all_texts[i]).strip() + "\n")
                f.write("\n---\n\n")

    sanity_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "sanity")
    summary_path = sanity_dir / f"layer{args.layer}_sanity.json"
    payload = {
        "layer": args.layer,
        "length_confound": len_stats,
        "top_feature_length_correlations": corr_items,
        "quick_auc": float(auc),
        "recommendation": recommendation,
    }
    save_with_metadata(
        output_path=summary_path,
        payload={"sanity": payload},
        config={"run": cfg, "args": vars(args)},
    )
    logging.info("Saved sanity summary: %s", summary_path)
    logging.info("Saved qualitative examples: %s", qual_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
