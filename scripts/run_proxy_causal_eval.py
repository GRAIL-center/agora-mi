from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.bootstrap import bootstrap_ci
from data.io import read_jsonl
from interventions.ablate import random_feature_ids, sequence_logprob
from model.hooks import layer_overwrite_hook
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config
from sae.load_sae import load_sae_for_layer


def _resolve_token_ids(tokenizer, token: str) -> list[int]:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Token label maps to no tokens: {token}")
    return [int(value) for value in ids]


def _render_proxy_prompt(template: str, text: str, left_label: str, right_label: str) -> str:
    return (
        template.replace("{TEXT}", text)
        .replace("{LEFT_LABEL}", left_label)
        .replace("{RIGHT_LABEL}", right_label)
    )


def _editor_factory(*, sae, feature_ids: list[int], layer: int, site: str, device) -> callable:
    idx = torch.tensor(feature_ids, dtype=torch.long, device=device)

    def editor(selected: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
        features = sae.encode(selected.to(torch.float32))
        edited = features.clone()
        edited[:, idx] = 0.0
        return sae.decode(edited).to(selected.dtype)

    return editor


def _intervened_label_logprob(
    *,
    model,
    tokenizer,
    prompt: str,
    target_token_ids: list[int],
    editor,
    layer: int,
    site: str,
    device,
    max_length: int,
) -> float:
    baseline, encoded, prompt_pos = sequence_logprob(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_token_ids=target_token_ids,
        device=device,
        max_length=max_length,
    )
    if editor is None:
        return baseline
    with layer_overwrite_hook(
        model,
        layer=layer,
        site=site,
        editor=editor,
        attention_mask=encoded["attention_mask"],
        token_index=prompt_pos,
    ):
        with torch.no_grad():
            out = model(**encoded, use_cache=False)
            logits = out.logits[0]
            positions = torch.arange(
                prompt_pos,
                prompt_pos + len(target_token_ids),
                device=device,
                dtype=torch.long,
            )
            step_logits = logits[positions]
            log_probs = torch.log_softmax(step_logits, dim=-1)
            target = torch.tensor(target_token_ids, dtype=torch.long, device=device)
            gathered = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            return float(gathered.sum().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_mech_interp.yaml")
    parser.add_argument("--manifest_root", default="data/processed/public_values")
    parser.add_argument("--family", required=True)
    parser.add_argument("--proxy_slug", required=True)
    parser.add_argument("--paired_proxy_slug", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--random_sets", type=int, default=100)
    parser.add_argument("--feature_ids", required=True)
    parser.add_argument("--sae_npz", default=None)
    args = parser.parse_args()

    setup_logging("run_proxy_causal_eval")
    cfg = read_config(args.config)
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    template = str(prompt_cfg["proxy_forced_choice_template"])
    target_label = str(prompt_cfg["proxy_target_tokens"][args.proxy_slug])
    contrast_label = str(prompt_cfg["proxy_target_tokens"][args.paired_proxy_slug])

    rows = read_jsonl(Path(args.manifest_root) / args.family / "proxies" / args.proxy_slug / f"{args.split}.jsonl")
    rows = rows[: int(args.max_samples)]
    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "proxy_causal"
        / args.family
        / args.proxy_slug
        / f"layer{args.layer}_{args.site}"
    )

    feature_ids = [int(value) for value in str(args.feature_ids).split(",") if str(value).strip()]
    if not rows or not feature_ids:
        save_with_metadata(
            output_path=out_dir / f"top{len(feature_ids)}_{args.split}.json",
            payload={
                "family_name": args.family,
                "proxy_slug": args.proxy_slug,
                "paired_proxy_slug": args.paired_proxy_slug,
                "split": args.split,
                "effective_k": len(feature_ids),
                "feature_ids": feature_ids,
                "skipped": True,
                "skip_reason": "missing_rows_or_features",
            },
            config={"run": cfg, "args": vars(args)},
        )
        return 0

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    target_token_ids = _resolve_token_ids(tokenizer, target_label)
    contrast_token_ids = _resolve_token_ids(tokenizer, contrast_label)
    sae = load_sae_for_layer(cfg, layer=args.layer, site=args.site, device=device, npz_path=args.sae_npz)
    target_editor = _editor_factory(sae=sae, feature_ids=feature_ids, layer=args.layer, site=args.site, device=device)

    prompts = [
        _render_proxy_prompt(
            template,
            str(row["text"]),
            left_label=target_label,
            right_label=contrast_label,
        )
        for row in rows
    ]
    baseline_margins: list[float] = []
    ablated_margins: list[float] = []
    sample_rows: list[dict[str, float | str]] = []

    for row, prompt in zip(rows, prompts):
        baseline_target = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=target_token_ids,
            editor=None,
            layer=args.layer,
            site=args.site,
            device=device,
            max_length=int(cfg.get("max_length", 1024)),
        )
        baseline_contrast = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=contrast_token_ids,
            editor=None,
            layer=args.layer,
            site=args.site,
            device=device,
            max_length=int(cfg.get("max_length", 1024)),
        )
        edited_target = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=target_token_ids,
            editor=target_editor,
            layer=args.layer,
            site=args.site,
            device=device,
            max_length=int(cfg.get("max_length", 1024)),
        )
        edited_contrast = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=contrast_token_ids,
            editor=target_editor,
            layer=args.layer,
            site=args.site,
            device=device,
            max_length=int(cfg.get("max_length", 1024)),
        )
        baseline_margin = float(baseline_target - baseline_contrast)
        edited_margin = float(edited_target - edited_contrast)
        baseline_margins.append(baseline_margin)
        ablated_margins.append(edited_margin)
        sample_rows.append(
            {
                "segment_id": str(row["segment_id"]),
                "baseline_margin": baseline_margin,
                "ablated_margin": edited_margin,
            }
        )

    target_margin_drop = np.asarray(baseline_margins, dtype=np.float64) - np.asarray(ablated_margins, dtype=np.float64)
    random_margin_drop_runs: list[np.ndarray] = []
    for random_index in range(int(args.random_sets)):
        random_ids = random_feature_ids(
            d_sae=sae.d_sae,
            k=len(feature_ids),
            exclude=feature_ids,
            seed=int(cfg.get("seed", 0)) + random_index,
        )
        random_editor = _editor_factory(sae=sae, feature_ids=random_ids, layer=args.layer, site=args.site, device=device)
        random_ablated_margins: list[float] = []
        for prompt in prompts:
            edited_target = _intervened_label_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_token_ids=target_token_ids,
                editor=random_editor,
                layer=args.layer,
                site=args.site,
                device=device,
                max_length=int(cfg.get("max_length", 1024)),
            )
            edited_contrast = _intervened_label_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_token_ids=contrast_token_ids,
                editor=random_editor,
                layer=args.layer,
                site=args.site,
                device=device,
                max_length=int(cfg.get("max_length", 1024)),
            )
            random_ablated_margins.append(float(edited_target - edited_contrast))
        random_margin_drop_runs.append(np.asarray(baseline_margins, dtype=np.float64) - np.asarray(random_ablated_margins, dtype=np.float64))

    random_margin_drop_matrix = np.stack(random_margin_drop_runs, axis=0)
    mean_random_margin_drop = random_margin_drop_matrix.mean(axis=0)
    paired_delta = target_margin_drop - mean_random_margin_drop
    ci_low, ci_high = bootstrap_ci(
        paired_delta,
        B=int(cfg.get("bootstrap_B", 500)),
        alpha=0.05,
        seed=int(cfg.get("seed", 0)) + 17,
    )
    causal_selectivity = float(paired_delta.mean()) if paired_delta.size else float("nan")
    passes = not np.isnan(causal_selectivity) and causal_selectivity > 0.0 and not np.isnan(ci_low) and ci_low > 0.0

    for index, row in enumerate(sample_rows):
        row["target_margin_drop"] = float(target_margin_drop[index])
        row["mean_random_margin_drop"] = float(mean_random_margin_drop[index])
        row["paired_delta"] = float(paired_delta[index])

    save_with_metadata(
        output_path=out_dir / f"top{len(feature_ids)}_{args.split}.json",
        payload={
            "family_name": args.family,
            "proxy_slug": args.proxy_slug,
            "paired_proxy_slug": args.paired_proxy_slug,
            "split": args.split,
            "evaluation_rows": "positive_test_only",
            "effective_k": len(feature_ids),
            "feature_ids": feature_ids,
            "n_samples": len(sample_rows),
            "random_sets": int(args.random_sets),
            "mean_target_margin_drop": float(target_margin_drop.mean()) if target_margin_drop.size else float("nan"),
            "mean_random_margin_drop": float(mean_random_margin_drop.mean()) if mean_random_margin_drop.size else float("nan"),
            "causal_selectivity": causal_selectivity,
            "causal_selectivity_ci_low": ci_low,
            "causal_selectivity_ci_high": ci_high,
            "passes_positive_causality": passes,
            "sample_rows": sample_rows,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
