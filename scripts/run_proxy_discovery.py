from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.cluster_stats import cluster_bootstrap_selection_frequency, cluster_permutation_pvalues
from analysis.fdr import benjamini_hochberg
from analysis.metrics import quick_roc_auc
from analysis.polarization import polarization_table
from data.io import read_jsonl
from model.hooks import capture_layer_site_sequence, pool_sequence_activations
from model.load_model import load_model_bundle
from model.prompt import build_prompts, load_prompt_config
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
    site: str,
    batch_size: int,
    max_length: int,
    pooling: str,
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
            h_seq = capture_layer_site_sequence(model, layer=layer, site=site, inputs=enc)
            pooled = pool_sequence_activations(h_seq, attention_mask=enc["attention_mask"], pooling=pooling)
            f = encode_features(sae, pooled.to(torch.float32))
        all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def _resolve_proxy_dir(manifest_root: Path, family: str, proxy_slug: str) -> Path:
    proxy_dir = manifest_root / family / "proxies" / proxy_slug
    if not proxy_dir.exists():
        raise FileNotFoundError(f"Missing proxy manifest directory: {proxy_dir}")
    return proxy_dir


def _resolve_negative_dir(manifest_root: Path, family: str, proxy_slug: str) -> Path:
    neg_dir = manifest_root / family / "negatives" / proxy_slug
    if not neg_dir.exists():
        raise FileNotFoundError(f"Missing negative manifest directory: {neg_dir}")
    return neg_dir


def _load_split_pair(proxy_dir: Path, neg_dir: Path, split: str) -> tuple[list[dict], list[dict]]:
    return read_jsonl(proxy_dir / f"{split}.jsonl"), read_jsonl(neg_dir / f"{split}.jsonl")


def _weighted_score(features: np.ndarray, feature_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if feature_ids.size == 0:
        return np.zeros(features.shape[0], dtype=np.float64)
    return features[:, feature_ids] @ weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_mech_interp.yaml")
    parser.add_argument("--manifest_root", default="data/processed/public_values")
    parser.add_argument("--family", required=True)
    parser.add_argument("--proxy_slug", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument("--pooling", default=None)
    parser.add_argument("--discovery_split", default=None)
    parser.add_argument("--evaluation_split", default=None)
    parser.add_argument("--sae_npz", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--perm_N", type=int, default=None)
    parser.add_argument("--bootstrap_B", type=int, default=None)
    args = parser.parse_args()

    setup_logging("run_proxy_discovery")
    cfg = read_config(args.config)
    discovery_split = str(args.discovery_split or cfg.get("discovery", {}).get("discovery_split", "train"))
    evaluation_split = str(args.evaluation_split or cfg.get("discovery", {}).get("evaluation_split", "test"))
    pooling = str(args.pooling or cfg.get("pooling", "mean"))
    topk = int(args.topk if args.topk is not None else cfg.get("topk", 64))
    perm_N = int(args.perm_N if args.perm_N is not None else cfg.get("perm_N", 2000))
    bootstrap_B = int(args.bootstrap_B if args.bootstrap_B is not None else cfg.get("bootstrap_B", 500))
    seed = int(cfg.get("seed", 0))

    manifest_root = Path(args.manifest_root)
    proxy_dir = _resolve_proxy_dir(manifest_root, args.family, args.proxy_slug)
    neg_dir = _resolve_negative_dir(manifest_root, args.family, args.proxy_slug)
    pos_train, neg_train = _load_split_pair(proxy_dir, neg_dir, discovery_split)
    pos_eval, neg_eval = _load_split_pair(proxy_dir, neg_dir, evaluation_split)
    if not pos_train or not neg_train:
        raise ValueError("Discovery split is empty for positive or negative rows.")
    if not pos_eval or not neg_eval:
        raise ValueError("Evaluation split is empty for positive or negative rows.")

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    template = str(prompt_cfg["template_v1"])
    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    sae = load_sae_for_layer(cfg, layer=args.layer, site=args.site, device=device, npz_path=args.sae_npz)
    batch_size = int(cfg.get("batch_size", 4))
    max_length = int(cfg.get("max_length", 1024))
    use_chat_template = bool(cfg.get("use_chat_template", False))

    pos_train_prompts = build_prompts(pos_train, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    neg_train_prompts = build_prompts(neg_train, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    pos_eval_prompts = build_prompts(pos_eval, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    neg_eval_prompts = build_prompts(neg_eval, template, tokenizer=tokenizer, use_chat_template=use_chat_template)

    feats_pos_train = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=pos_train_prompts,
        layer=args.layer,
        site=args.site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    feats_neg_train = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=neg_train_prompts,
        layer=args.layer,
        site=args.site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    feats_pos_eval = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=pos_eval_prompts,
        layer=args.layer,
        site=args.site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    feats_neg_eval = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=neg_eval_prompts,
        layer=args.layer,
        site=args.site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )

    train_df = polarization_table(feats_pos_train, feats_neg_train, perm_N=0, seed=seed)
    p_values = cluster_permutation_pvalues(
        feats_pos_train,
        feats_neg_train,
        pos_cluster_ids=np.asarray([row["document_id"] for row in pos_train], dtype=object),
        neg_cluster_ids=np.asarray([row["document_id"] for row in neg_train], dtype=object),
        n_perm=perm_N,
        seed=seed,
    )
    train_df["p_value"] = p_values
    fdr = benjamini_hochberg(train_df["p_value"].values, q=float(cfg.get("fdr_q", 0.05)))
    train_df["q_value"] = fdr["q_values"]
    train_df["fdr_reject"] = fdr["reject"]

    positive_candidates = train_df[train_df["delta"] > 0].copy()
    if bool(cfg.get("discovery", {}).get("use_fdr_survivors_first", True)):
        positive_candidates = positive_candidates[positive_candidates["fdr_reject"]]
    if positive_candidates.empty:
        positive_candidates = train_df[train_df["delta"] > 0].copy()
    selected = positive_candidates.sort_values("delta", ascending=False).head(topk)
    feature_ids = selected["feature_id"].astype(int).to_numpy()
    raw_weights = selected["delta"].to_numpy(dtype=np.float64)
    weight_denom = np.abs(raw_weights).sum()
    weights = raw_weights / weight_denom if weight_denom > 0 else np.zeros_like(raw_weights)

    stability = cluster_bootstrap_selection_frequency(
        feats_pos_train,
        feats_neg_train,
        pos_cluster_ids=np.asarray([row["document_id"] for row in pos_train], dtype=object),
        neg_cluster_ids=np.asarray([row["document_id"] for row in neg_train], dtype=object),
        topk=topk,
        n_boot=bootstrap_B,
        seed=seed + 1,
    )
    selected_stability = {
        int(fid): float(stability[int(fid)])
        for fid in feature_ids.tolist()
    }

    eval_df = polarization_table(feats_pos_eval, feats_neg_eval, perm_N=0, seed=seed)
    y_true = np.concatenate(
        [
            np.ones(feats_pos_eval.shape[0], dtype=np.int64),
            np.zeros(feats_neg_eval.shape[0], dtype=np.int64),
        ]
    )
    eval_features = np.concatenate([feats_pos_eval, feats_neg_eval], axis=0)
    eval_scores = _weighted_score(eval_features, feature_ids, weights)
    eval_auc = quick_roc_auc(y_true, eval_scores)

    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "policy_discovery"
        / args.family
        / args.proxy_slug
        / f"layer{args.layer}_{args.site}"
    )
    train_df.to_csv(out_dir / "train_delta.csv", index=False)
    eval_df.to_csv(out_dir / "eval_delta.csv", index=False)
    np.savez_compressed(
        out_dir / "eval_features.npz",
        positive_features=feats_pos_eval,
        negative_features=feats_neg_eval,
        positive_ids=np.asarray([row["segment_id"] for row in pos_eval], dtype=object),
        negative_ids=np.asarray([row["segment_id"] for row in neg_eval], dtype=object),
    )

    feature_bank = {
        "family_name": args.family,
        "proxy_slug": args.proxy_slug,
        "proxy_name": pos_train[0].get("proxy_name"),
        "layer": args.layer,
        "site": args.site,
        "pooling": pooling,
        "discovery_split": discovery_split,
        "evaluation_split": evaluation_split,
        "feature_ids": feature_ids.astype(int).tolist(),
        "feature_weights": weights.tolist(),
        "train_top_deltas": raw_weights.tolist(),
        "train_q_values": selected["q_value"].tolist(),
        "bootstrap_stability": selected_stability,
    }
    (out_dir / "feature_bank.json").write_text(json.dumps(feature_bank, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "bootstrap_stability.json").write_text(
        json.dumps({"feature_stability": selected_stability}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_with_metadata(
        output_path=out_dir / "eval_summary.json",
        payload={
            "feature_bank_path": str(out_dir / "feature_bank.json"),
            "train_delta_csv": str(out_dir / "train_delta.csv"),
            "eval_delta_csv": str(out_dir / "eval_delta.csv"),
            "eval_auc": eval_auc,
            "n_positive_train": len(pos_train),
            "n_negative_train": len(neg_train),
            "n_positive_eval": len(pos_eval),
            "n_negative_eval": len(neg_eval),
            "n_selected_features": int(feature_ids.size),
            "n_fdr_survivors": int(train_df["fdr_reject"].sum()),
        },
        config={"run": cfg, "args": vars(args)},
    )
    logging.info("Saved discovery outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
