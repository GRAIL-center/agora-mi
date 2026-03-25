from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.metrics import quick_roc_auc
from data.io import read_jsonl
from data.matching import fit_tfidf_logistic, masked_texts, proxy_keywords, score_tfidf_logistic
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


def _weighted_score(features: np.ndarray, feature_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if feature_ids.size == 0:
        return np.zeros(features.shape[0], dtype=np.float64)
    return features[:, feature_ids] @ weights


def _resolve_transfer_spec(args, cfg: dict) -> dict:
    if args.transfer_manifest:
        spec = json.loads(Path(args.transfer_manifest).read_text(encoding="utf-8"))
    else:
        if not all([args.source_family, args.source_proxy_slug, args.target_family, args.target_proxy_slug]):
            raise ValueError("Provide either --transfer_manifest or explicit source/target family and proxy slugs.")
        spec = {
            "source_family": args.source_family,
            "source_proxy_slug": args.source_proxy_slug,
            "target_family": args.target_family,
            "target_proxy_slug": args.target_proxy_slug,
            "family_relation": "within_family" if args.source_family == args.target_family else "cross_family",
            "feature_bank_path": None,
        }

    if spec.get("feature_bank_path"):
        return spec

    search_root = Path(cfg.get("results_dir", "results/policy_mech_interp")) / "policy_discovery" / spec["source_family"] / spec["source_proxy_slug"]
    candidates = sorted(search_root.rglob("feature_bank.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No feature_bank.json found under {search_root}")
    spec["feature_bank_path"] = str(candidates[0])
    return spec


def _load_pair_rows(manifest_root: Path, family: str, proxy_slug: str, split: str) -> tuple[list[dict], list[dict]]:
    pos_rows = read_jsonl(manifest_root / family / "proxies" / proxy_slug / f"{split}.jsonl")
    neg_rows = read_jsonl(manifest_root / family / "negatives" / proxy_slug / f"{split}.jsonl")
    return pos_rows, neg_rows


def _copy_rows_with_text(rows: list[dict], texts: list[str]) -> list[dict]:
    return [{**row, "text": text} for row, text in zip(rows, texts)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_mech_interp.yaml")
    parser.add_argument("--manifest_root", default="data/processed/public_values")
    parser.add_argument("--transfer_manifest", default=None)
    parser.add_argument("--source_family", default=None)
    parser.add_argument("--source_proxy_slug", default=None)
    parser.add_argument("--target_family", default=None)
    parser.add_argument("--target_proxy_slug", default=None)
    parser.add_argument("--sae_npz", default=None)
    args = parser.parse_args()

    setup_logging("run_proxy_transfer")
    cfg = read_config(args.config)
    spec = _resolve_transfer_spec(args, cfg)
    feature_bank = json.loads(Path(spec["feature_bank_path"]).read_text(encoding="utf-8"))
    layer = int(feature_bank["layer"])
    site = str(feature_bank["site"])
    pooling = str(feature_bank["pooling"])
    discovery_split = str(feature_bank["discovery_split"])
    evaluation_split = str(feature_bank["evaluation_split"])
    feature_ids = np.asarray(feature_bank["feature_ids"], dtype=np.int64)
    weights = np.asarray(feature_bank["feature_weights"], dtype=np.float64)

    manifest_root = Path(args.manifest_root)
    source_pos, source_neg = _load_pair_rows(manifest_root, spec["source_family"], spec["source_proxy_slug"], discovery_split)
    target_pos, target_neg = _load_pair_rows(manifest_root, spec["target_family"], spec["target_proxy_slug"], evaluation_split)

    source_proxy_name = source_pos[0].get("proxy_name")
    target_proxy_name = target_pos[0].get("proxy_name")
    keywords = sorted(set(proxy_keywords(source_proxy_name) + proxy_keywords(target_proxy_name)))

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    template = str(prompt_cfg["template_v1"])
    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    sae = load_sae_for_layer(cfg, layer=layer, site=site, device=device, npz_path=args.sae_npz)
    batch_size = int(cfg.get("batch_size", 4))
    max_length = int(cfg.get("max_length", 1024))
    use_chat_template = bool(cfg.get("use_chat_template", False))

    pos_prompts = build_prompts(target_pos, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    neg_prompts = build_prompts(target_neg, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    feats_pos = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=pos_prompts,
        layer=layer,
        site=site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    feats_neg = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=neg_prompts,
        layer=layer,
        site=site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )

    y_true = np.concatenate([np.ones(len(target_pos), dtype=np.int64), np.zeros(len(target_neg), dtype=np.int64)])
    transfer_scores = _weighted_score(np.concatenate([feats_pos, feats_neg], axis=0), feature_ids, weights)
    transfer_auc = quick_roc_auc(y_true, transfer_scores)

    source_train_texts = [row["text"] for row in source_pos] + [row["text"] for row in source_neg]
    source_labels = [1] * len(source_pos) + [0] * len(source_neg)
    lexical_vectorizer, lexical_model = fit_tfidf_logistic(source_train_texts, source_labels)
    lexical_scores = score_tfidf_logistic(
        lexical_vectorizer,
        lexical_model,
        [row["text"] for row in target_pos] + [row["text"] for row in target_neg],
    )
    lexical_auc = quick_roc_auc(y_true, lexical_scores)

    masked_source_texts = masked_texts(source_train_texts, keywords)
    masked_target_texts = masked_texts([row["text"] for row in target_pos] + [row["text"] for row in target_neg], keywords)
    masked_vectorizer, masked_model = fit_tfidf_logistic(masked_source_texts, source_labels)
    masked_lexical_scores = score_tfidf_logistic(masked_vectorizer, masked_model, masked_target_texts)
    masked_lexical_auc = quick_roc_auc(y_true, masked_lexical_scores)

    masked_target_pos = _copy_rows_with_text(target_pos, masked_target_texts[: len(target_pos)])
    masked_target_neg = _copy_rows_with_text(target_neg, masked_target_texts[len(target_pos) :])
    masked_pos_prompts = build_prompts(masked_target_pos, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    masked_neg_prompts = build_prompts(masked_target_neg, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    masked_feats_pos = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=masked_pos_prompts,
        layer=layer,
        site=site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    masked_feats_neg = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=masked_neg_prompts,
        layer=layer,
        site=site,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    masked_transfer_scores = _weighted_score(
        np.concatenate([masked_feats_pos, masked_feats_neg], axis=0),
        feature_ids,
        weights,
    )
    masked_transfer_auc = quick_roc_auc(y_true, masked_transfer_scores)

    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "proxy_transfer"
        / spec["source_family"]
    )
    out_path = out_dir / f"{spec['source_proxy_slug']}__to__{spec['target_proxy_slug']}.json"
    save_with_metadata(
        output_path=out_path,
        payload={
            "source_family": spec["source_family"],
            "source_proxy_slug": spec["source_proxy_slug"],
            "source_proxy_name": source_proxy_name,
            "target_family": spec["target_family"],
            "target_proxy_slug": spec["target_proxy_slug"],
            "target_proxy_name": target_proxy_name,
            "family_relation": spec["family_relation"],
            "feature_bank_path": spec["feature_bank_path"],
            "feature_transfer_auc": transfer_auc,
            "feature_transfer_auc_masked": masked_transfer_auc,
            "lexical_baseline_auc": lexical_auc,
            "lexical_baseline_auc_masked": masked_lexical_auc,
            "proxy_keywords": keywords,
            "evaluation_split": evaluation_split,
            "discovery_split": discovery_split,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
