from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
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
            feats = encode_features(sae, pooled.to(torch.float32))
        all_feats.append(feats.detach().cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_mech_interp.yaml")
    parser.add_argument("--manifest_root", default="data/processed/public_values")
    parser.add_argument("--family", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument("--split", default="test")
    parser.add_argument("--validated_only", action="store_true")
    parser.add_argument("--sae_npz", default=None)
    args = parser.parse_args()

    setup_logging("run_strategy_modulation")
    cfg = read_config(args.config)
    core_path = (
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "family_core"
        / args.family
        / f"layer{args.layer}_{args.site}"
        / "family_core_bank.json"
    )
    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "strategy_modulation"
        / args.family
        / f"layer{args.layer}_{args.site}"
    )
    core_obj = json.loads(core_path.read_text(encoding="utf-8"))
    core_feature_ids = [int(row["feature_id"]) for row in core_obj.get("core_features", [])]
    if not core_feature_ids:
        save_with_metadata(
            output_path=out_dir / f"{args.split}.json",
            payload={
                "family_name": args.family,
                "layer": args.layer,
                "site": args.site,
                "split": args.split,
                "validated_only": args.validated_only,
                "skipped": True,
                "skip_reason": "family_core_bank_empty",
            },
            config={"run": cfg, "args": vars(args)},
        )
        return 0

    manifest_dir = Path(args.manifest_root) / ("validated" if args.validated_only else "eligible")
    rows = read_jsonl(manifest_dir / f"{args.split}.jsonl")
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    template = str(prompt_cfg["template_v1"])
    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    sae = load_sae_for_layer(cfg, layer=args.layer, site=args.site, device=device, npz_path=args.sae_npz)
    pooling = str(cfg.get("pooling", "mean"))

    prompts = build_prompts(rows, template, tokenizer=tokenizer, use_chat_template=bool(cfg.get("use_chat_template", False)))
    feats = _extract_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sae=sae,
        prompts=prompts,
        layer=args.layer,
        site=args.site,
        batch_size=int(cfg.get("batch_size", 4)),
        max_length=int(cfg.get("max_length", 1024)),
        pooling=pooling,
    )
    scores = feats[:, core_feature_ids].mean(axis=1)

    frame_rows = []
    for row, score in zip(rows, scores.tolist()):
        meta = row.get("metadata", {})
        strategies = row.get("strategy_categories", [])
        frame_rows.append(
            {
                "core_score": score,
                "length_chars": len(row.get("text", "")),
                "year": meta.get("year"),
                "authority": meta.get("authority") or "missing",
                "jurisdiction": meta.get("jurisdiction") or "missing",
                "document_form": meta.get("document_form") or "missing",
                "co_occurring_proxy_count": len(row.get("main_proxy_hits", [])) + len(row.get("secondary_proxy_hits", [])),
                **{f"strategy__{strategy}": int(strategy in strategies) for strategy in row.get("strategy_categories", [])},
            }
        )
    df = pd.DataFrame(frame_rows).fillna({"year": 0})
    strategy_cols = [col for col in df.columns if col.startswith("strategy__")]
    base_df = pd.get_dummies(
        df[["length_chars", "year", "authority", "jurisdiction", "document_form", "co_occurring_proxy_count"]],
        columns=["authority", "jurisdiction", "document_form"],
        dtype=float,
    )
    full_df = pd.concat([base_df, df[strategy_cols]], axis=1).fillna(0.0)
    y = df["core_score"].to_numpy(dtype=np.float64)

    base_model = LinearRegression().fit(base_df.to_numpy(dtype=np.float64), y)
    full_model = LinearRegression().fit(full_df.to_numpy(dtype=np.float64), y)
    base_r2 = r2_score(y, base_model.predict(base_df.to_numpy(dtype=np.float64)))
    full_r2 = r2_score(y, full_model.predict(full_df.to_numpy(dtype=np.float64)))

    strategy_coefficients = {}
    for idx, col in enumerate(full_df.columns.tolist()):
        if col.startswith("strategy__"):
            strategy_coefficients[col] = float(full_model.coef_[idx])

    save_with_metadata(
        output_path=out_dir / f"{args.split}.json",
        payload={
            "family_name": args.family,
            "layer": args.layer,
            "site": args.site,
            "split": args.split,
            "validated_only": args.validated_only,
            "n_rows": int(len(df)),
            "n_core_features": len(core_feature_ids),
            "base_r2": float(base_r2),
            "full_r2": float(full_r2),
            "strategy_delta_r2": float(full_r2 - base_r2),
            "strategy_coefficients": strategy_coefficients,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
