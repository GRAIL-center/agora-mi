from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from data.io import read_jsonl
from data.matching import bounded_r2
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
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sae_npz", default=None)
    args = parser.parse_args()

    setup_logging("run_family_compositionality")
    cfg = read_config(args.config)
    manifest_root = Path(args.manifest_root)

    membership: dict[str, set[str]] = defaultdict(set)
    row_lookup: dict[str, dict] = {}
    core_feature_union: set[int] = set()
    for family in args.families:
        rows = read_jsonl(manifest_root / family / "family_pools" / "all" / f"{args.split}.jsonl")
        for row in rows:
            membership[row["segment_id"]].add(family)
            row_lookup[row["segment_id"]] = row
        core_path = (
            Path(cfg.get("results_dir", "results/policy_mech_interp"))
            / "family_core"
            / family
            / f"layer{args.layer}_{args.site}"
            / "family_core_bank.json"
        )
        core_obj = json.loads(core_path.read_text(encoding="utf-8"))
        for feature_row in core_obj.get("core_features", []):
            core_feature_union.add(int(feature_row["feature_id"]))

    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "family_compositionality"
        / f"layer{args.layer}_{args.site}"
    )
    union_feature_ids = np.asarray(sorted(core_feature_union), dtype=np.int64)
    if union_feature_ids.size == 0:
        save_with_metadata(
            output_path=out_dir / f"{args.split}.json",
            payload={
                "families": args.families,
                "layer": args.layer,
                "site": args.site,
                "split": args.split,
                "skipped": True,
                "skip_reason": "family_core_union_empty",
            },
            config={"run": cfg, "args": vars(args)},
        )
        return 0

    rows = [row_lookup[segment_id] for segment_id in sorted(row_lookup.keys())]
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    template = str(prompt_cfg["template_v1"])
    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    sae = load_sae_for_layer(cfg, layer=args.layer, site=args.site, device=device, npz_path=args.sae_npz)
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
        pooling=str(cfg.get("pooling", "mean")),
    )
    union_feats = feats[:, union_feature_ids]

    single_family_centroids: dict[str, np.ndarray] = {}
    for family in args.families:
        family_indices = [
            idx for idx, row in enumerate(rows)
            if membership[row["segment_id"]] == {family}
        ]
        if not family_indices:
            continue
        single_family_centroids[family] = union_feats[family_indices].mean(axis=0)

    summary_rows = []
    for idx, row in enumerate(rows):
        member_families = sorted(membership[row["segment_id"]])
        if len(member_families) < 2:
            continue
        centroid_stack = [single_family_centroids[family] for family in member_families if family in single_family_centroids]
        if not centroid_stack:
            continue
        predicted = np.mean(np.stack(centroid_stack, axis=0), axis=0)
        observed = union_feats[idx]
        residual = observed - predicted
        summary_rows.append(
            {
                "segment_id": row["segment_id"],
                "families": member_families,
                "residual_norm": float(np.linalg.norm(residual)),
                "predicted_norm": float(np.linalg.norm(predicted)),
                "observed_norm": float(np.linalg.norm(observed)),
                "r2": bounded_r2(observed, predicted),
            }
        )

    save_with_metadata(
        output_path=out_dir / f"{args.split}.json",
        payload={
            "families": args.families,
            "layer": args.layer,
            "site": args.site,
            "split": args.split,
            "n_union_features": int(union_feature_ids.size),
            "n_multi_family_segments": len(summary_rows),
            "mean_residual_norm": float(np.mean([row["residual_norm"] for row in summary_rows])) if summary_rows else float("nan"),
            "mean_r2": float(np.mean([row["r2"] for row in summary_rows])) if summary_rows else float("nan"),
            "rows": summary_rows,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
