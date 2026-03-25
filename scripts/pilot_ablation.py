from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.bootstrap import bootstrap_ci
from analysis.permutation import paired_permutation_sign_flip_test
from data.io import read_jsonl
from interventions.ablate import intervention_logprob, random_feature_ids
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt
from sae.load_sae import load_sae_for_layer


def _resolve_token_ids(tokenizer, token: str) -> list[int]:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Target token '{token}' maps to no tokens.")
    return [int(i) for i in ids]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--safe_jsonl", default="data/processed/dsafe_dev.jsonl")
    parser.add_argument("--topk_safe_json", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--sae_npz", default=None)
    parser.add_argument("--bootstrap_B", type=int, default=None)
    parser.add_argument("--perm_N", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary_out", default=None)
    parser.add_argument("--sample_out", default=None)
    args = parser.parse_args()

    setup_logging("pilot_ablation")
    cfg = read_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])
    target_token = str(prompt_cfg.get("target_tokens", {}).get("RESTRICTION", "RESTRICTION"))

    rows = read_jsonl(args.safe_jsonl)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    prompts = [render_prompt(template, str(r["text"])) for r in rows]
    ids = [str(r.get("id", i)) for i, r in enumerate(rows)]
    if not prompts:
        raise ValueError("No prompts for pilot ablation.")

    topk_file = args.topk_safe_json or str(
        Path(cfg.get("results_dir", "results")) / "polarization" / f"layer{args.layer}_topk_safe.json"
    )
    with Path(topk_file).open("r", encoding="utf-8") as f:
        topk_obj = json.load(f)
    safe_feature_ids = [int(x) for x in topk_obj["feature_ids"]]
    topk = int(args.k if args.k is not None else cfg.get("topk", 64))
    safe_feature_ids = safe_feature_ids[:topk]
    if not safe_feature_ids:
        raise ValueError("No safety features found in topk safe json.")

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    sae = load_sae_for_layer(cfg, layer=args.layer, device=device, npz_path=args.sae_npz)
    target_token_ids = _resolve_token_ids(tokenizer, target_token)

    random_ids = random_feature_ids(
        d_sae=sae.d_sae,
        k=len(safe_feature_ids),
        exclude=safe_feature_ids,
        seed=seed,
    )

    max_length = int(cfg.get("max_length", 1024))
    rows_out = []
    for i, prompt in enumerate(prompts):
        base, abl = intervention_logprob(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            prompt=prompt,
            target_token_ids=target_token_ids,
            layer=args.layer,
            feature_ids=safe_feature_ids,
            device=device,
            max_length=max_length,
        )
        # Random matched control
        base2, rand_abl = intervention_logprob(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            prompt=prompt,
            target_token_ids=target_token_ids,
            layer=args.layer,
            feature_ids=random_ids,
            device=device,
            max_length=max_length,
        )
        rows_out.append(
            {
                "id": ids[i],
                "baseline_logprob": base,
                "ablated_logprob": abl,
                "effect": base - abl,
                "random_ablated_logprob": rand_abl,
                "random_effect": base2 - rand_abl,
            }
        )

    df = pd.DataFrame(rows_out)
    effects = df["effect"].to_numpy(dtype=np.float64)
    random_effects = df["random_effect"].to_numpy(dtype=np.float64)
    paired_delta = effects - random_effects
    df["paired_delta"] = paired_delta
    B = int(args.bootstrap_B if args.bootstrap_B is not None else cfg.get("bootstrap_B", 200))
    perm_N = int(args.perm_N if args.perm_N is not None else cfg.get("perm_N", 10000))
    ci_low, ci_high = bootstrap_ci(effects, B=B, alpha=0.05, seed=seed)
    r_ci_low, r_ci_high = bootstrap_ci(random_effects, B=B, alpha=0.05, seed=seed + 1)
    d_ci_low, d_ci_high = bootstrap_ci(paired_delta, B=B, alpha=0.05, seed=seed + 2)
    d_perm_p = paired_permutation_sign_flip_test(paired_delta, N=perm_N, seed=seed + 3)

    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "pilot_ablation")
    sample_path = Path(args.sample_out) if args.sample_out else (out_dir / f"effects_layer{args.layer}.csv")
    summary_path = Path(args.summary_out) if args.summary_out else (out_dir / "effect_summary.json")
    ensure_dir(sample_path.parent)
    ensure_dir(summary_path.parent)
    df.to_csv(sample_path, index=False)

    summary = {
        "layer": args.layer,
        "n_samples": int(len(df)),
        "k_features": int(len(safe_feature_ids)),
        "seed": seed,
        "target_token": target_token,
        "target_token_ids": target_token_ids,
        "mean_effect": float(effects.mean()),
        "bootstrap_95_ci": [ci_low, ci_high],
        "random_control_mean_effect": float(random_effects.mean()),
        "random_control_bootstrap_95_ci": [r_ci_low, r_ci_high],
        "paired_delta_mean": float(paired_delta.mean()),
        "paired_delta_bootstrap_95_ci": [d_ci_low, d_ci_high],
        "paired_delta_permutation_p": d_perm_p,
        "perm_N": perm_N,
        "safe_feature_ids": safe_feature_ids,
        "random_feature_ids": random_ids,
        "sample_csv": str(sample_path),
    }
    save_with_metadata(
        output_path=summary_path,
        payload={"effect_summary": summary},
        config={"run": cfg, "args": vars(args)},
    )
    logging.info("Saved pilot sample effects: %s", sample_path)
    logging.info("Saved effect summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
