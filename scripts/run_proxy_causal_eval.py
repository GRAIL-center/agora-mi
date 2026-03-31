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


def _parse_off_target_specs(raw_value: str) -> list[tuple[str, str, str]]:
    if not str(raw_value).strip():
        return []
    specs: list[tuple[str, str, str]] = []
    for chunk in str(raw_value).split(","):
        part = chunk.strip()
        if not part:
            continue
        family, proxy_slug, contrast_slug = [value.strip() for value in part.split("::", maxsplit=2)]
        specs.append((family, proxy_slug, contrast_slug))
    return specs


def _editor_factory(*, sae, feature_ids: list[int], layer: int, site: str, device) -> callable:
    idx = torch.tensor(feature_ids, dtype=torch.long, device=device)

    def editor(selected: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
        features = sae.encode(selected.to(torch.float32))
        edited = features.clone()
        edited[:, idx] = 0.0
        return sae.decode(edited).to(selected.dtype)

    return editor


def _margin_series(
    *,
    rows: list[dict[str, object]],
    prompts: list[str],
    target_token_ids: list[int],
    contrast_token_ids: list[int],
    model,
    tokenizer,
    editor,
    layer: int,
    site: str,
    device,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    baseline_margins: list[float] = []
    edited_margins: list[float] = []
    sample_rows: list[dict[str, float | str]] = []
    for row, prompt in zip(rows, prompts):
        baseline_target = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=target_token_ids,
            editor=None,
            layer=layer,
            site=site,
            device=device,
            max_length=max_length,
        )
        baseline_contrast = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=contrast_token_ids,
            editor=None,
            layer=layer,
            site=site,
            device=device,
            max_length=max_length,
        )
        edited_target = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=target_token_ids,
            editor=editor,
            layer=layer,
            site=site,
            device=device,
            max_length=max_length,
        )
        edited_contrast = _intervened_label_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=contrast_token_ids,
            editor=editor,
            layer=layer,
            site=site,
            device=device,
            max_length=max_length,
        )
        baseline_margin = float(baseline_target - baseline_contrast)
        edited_margin = float(edited_target - edited_contrast)
        baseline_margins.append(baseline_margin)
        edited_margins.append(edited_margin)
        sample_rows.append(
            {
                "segment_id": str(row["segment_id"]),
                "baseline_margin": baseline_margin,
                "ablated_margin": edited_margin,
            }
        )
    return (
        np.asarray(baseline_margins, dtype=np.float64),
        np.asarray(edited_margins, dtype=np.float64),
        sample_rows,
    )


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
    parser.add_argument("--off_target_proxy_specs", default="")
    args = parser.parse_args()

    setup_logging("run_proxy_causal_eval")
    cfg = read_config(args.config)
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    template = str(prompt_cfg["proxy_forced_choice_template"])
    target_label = str(prompt_cfg["proxy_target_tokens"][args.proxy_slug])
    contrast_label = str(prompt_cfg["proxy_target_tokens"][args.paired_proxy_slug])

    rows = read_jsonl(Path(args.manifest_root) / args.family / "proxies" / args.proxy_slug / f"{args.split}.jsonl")
    rows = rows[: int(args.max_samples)]
    off_target_rows = read_jsonl(Path(args.manifest_root) / args.family / "proxies" / args.paired_proxy_slug / f"{args.split}.jsonl")
    off_target_rows = off_target_rows[: int(args.max_samples)]
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
    off_target_specs = _parse_off_target_specs(args.off_target_proxy_specs)

    prompts = [
        _render_proxy_prompt(
            template,
            str(row["text"]),
            left_label=target_label,
            right_label=contrast_label,
        )
        for row in rows
    ]
    max_length = int(cfg.get("max_length", 1024))
    baseline_margins, ablated_margins, sample_rows = _margin_series(
        rows=rows,
        prompts=prompts,
        target_token_ids=target_token_ids,
        contrast_token_ids=contrast_token_ids,
        model=model,
        tokenizer=tokenizer,
        editor=target_editor,
        layer=args.layer,
        site=args.site,
        device=device,
        max_length=max_length,
    )
    target_margin_drop = baseline_margins - ablated_margins
    random_margin_drop_runs: list[np.ndarray] = []
    random_run_mean_margin_drops: list[float] = []
    random_control_draws: list[dict[str, object]] = []
    random_control_feature_sets: list[list[int]] = []
    for random_index in range(int(args.random_sets)):
        random_ids = random_feature_ids(
            d_sae=sae.d_sae,
            k=len(feature_ids),
            exclude=feature_ids,
            seed=int(cfg.get("seed", 0)) + random_index,
        )
        random_control_feature_sets.append([int(value) for value in random_ids])
        random_editor = _editor_factory(sae=sae, feature_ids=random_ids, layer=args.layer, site=args.site, device=device)
        _, random_ablated_margins, _ = _margin_series(
            rows=rows,
            prompts=prompts,
            target_token_ids=target_token_ids,
            contrast_token_ids=contrast_token_ids,
            model=model,
            tokenizer=tokenizer,
            editor=random_editor,
            layer=args.layer,
            site=args.site,
            device=device,
            max_length=max_length,
        )
        run_drop = baseline_margins - random_ablated_margins
        random_margin_drop_runs.append(run_drop)
        random_run_mean_margin_drops.append(float(run_drop.mean()) if run_drop.size else float("nan"))
        random_control_draws.append(
            {
                "draw_index": int(random_index),
                "feature_ids": [int(value) for value in random_ids],
                "mean_margin_drop": float(run_drop.mean()) if run_drop.size else float("nan"),
                "std_margin_drop": float(run_drop.std(ddof=1)) if run_drop.size > 1 else 0.0,
            }
        )

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

    def _evaluate_off_target_proxy(
        *,
        family_name: str,
        proxy_slug: str,
        contrast_slug: str,
        seed_offset: int,
    ) -> dict[str, object]:
        off_rows = read_jsonl(Path(args.manifest_root) / family_name / "proxies" / proxy_slug / f"{args.split}.jsonl")
        off_rows = off_rows[: int(args.max_samples)]
        if not off_rows:
            return {
                "family_name": family_name,
                "proxy_slug": proxy_slug,
                "contrast_proxy_slug": contrast_slug,
                "n_samples": 0,
                "mean_target_margin_drop": float("nan"),
                "mean_random_margin_drop": float("nan"),
                "off_target_selectivity": float("nan"),
                "off_target_selectivity_ci_low": float("nan"),
                "off_target_selectivity_ci_high": float("nan"),
                "sample_rows": [],
            }
        off_target_label = str(prompt_cfg["proxy_target_tokens"][proxy_slug])
        off_contrast_label = str(prompt_cfg["proxy_target_tokens"][contrast_slug])
        off_target_token_ids = _resolve_token_ids(tokenizer, off_target_label)
        off_contrast_token_ids = _resolve_token_ids(tokenizer, off_contrast_label)
        off_prompts = [
            _render_proxy_prompt(
                template,
                str(row["text"]),
                left_label=off_target_label,
                right_label=off_contrast_label,
            )
            for row in off_rows
        ]
        off_baseline_margins, off_ablated_margins, off_sample_rows = _margin_series(
            rows=off_rows,
            prompts=off_prompts,
            target_token_ids=off_target_token_ids,
            contrast_token_ids=off_contrast_token_ids,
            model=model,
            tokenizer=tokenizer,
            editor=target_editor,
            layer=args.layer,
            site=args.site,
            device=device,
            max_length=max_length,
        )
        off_target_margin_drop = off_baseline_margins - off_ablated_margins
        off_random_margin_drop_runs: list[np.ndarray] = []
        for draw_index, random_ids in enumerate(random_control_feature_sets):
            random_editor = _editor_factory(sae=sae, feature_ids=random_ids, layer=args.layer, site=args.site, device=device)
            _, off_random_ablated_margins, _ = _margin_series(
                rows=off_rows,
                prompts=off_prompts,
                target_token_ids=off_target_token_ids,
                contrast_token_ids=off_contrast_token_ids,
                model=model,
                tokenizer=tokenizer,
                editor=random_editor,
                layer=args.layer,
                site=args.site,
                device=device,
                max_length=max_length,
            )
            off_random_margin_drop_runs.append(off_baseline_margins - off_random_ablated_margins)
        off_random_matrix = np.stack(off_random_margin_drop_runs, axis=0)
        off_mean_random_drop = off_random_matrix.mean(axis=0)
        off_delta = off_target_margin_drop - off_mean_random_drop
        off_ci_low, off_ci_high = bootstrap_ci(
            off_delta,
            B=int(cfg.get("bootstrap_B", 500)),
            alpha=0.05,
            seed=int(cfg.get("seed", 0)) + seed_offset,
        )
        for row_index, row in enumerate(off_sample_rows):
            row["target_margin_drop"] = float(off_target_margin_drop[row_index])
            row["mean_random_margin_drop"] = float(off_mean_random_drop[row_index])
            row["paired_delta"] = float(off_delta[row_index])
        return {
            "family_name": family_name,
            "proxy_slug": proxy_slug,
            "contrast_proxy_slug": contrast_slug,
            "n_samples": len(off_sample_rows),
            "mean_target_margin_drop": float(off_target_margin_drop.mean()) if off_target_margin_drop.size else float("nan"),
            "mean_random_margin_drop": float(off_mean_random_drop.mean()) if off_mean_random_drop.size else float("nan"),
            "off_target_selectivity": float(off_delta.mean()) if off_delta.size else float("nan"),
            "off_target_selectivity_ci_low": off_ci_low,
            "off_target_selectivity_ci_high": off_ci_high,
            "sample_rows": off_sample_rows,
        }

    paired_off_target_effect = _evaluate_off_target_proxy(
        family_name=args.family,
        proxy_slug=args.paired_proxy_slug,
        contrast_slug=args.proxy_slug,
        seed_offset=23,
    )
    additional_off_target_effects = [
        _evaluate_off_target_proxy(
            family_name=family_name,
            proxy_slug=proxy_slug,
            contrast_slug=contrast_slug,
            seed_offset=100 + spec_index,
        )
        for spec_index, (family_name, proxy_slug, contrast_slug) in enumerate(off_target_specs, start=1)
    ]
    off_target_effects = [paired_off_target_effect, *additional_off_target_effects]
    off_target_sample_rows = list(paired_off_target_effect.get("sample_rows", []))
    off_target_margin_drop = np.asarray([], dtype=np.float64)
    off_target_mean_random_drop = np.asarray([], dtype=np.float64)
    off_target_selectivity = float(paired_off_target_effect.get("off_target_selectivity", float("nan")))
    off_target_ci_low = float(paired_off_target_effect.get("off_target_selectivity_ci_low", float("nan")))
    off_target_ci_high = float(paired_off_target_effect.get("off_target_selectivity_ci_high", float("nan")))

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
            "intervention": {
                "layer": int(args.layer),
                "site": str(args.site),
                "feature_ids": feature_ids,
                "target_token_ids": target_token_ids,
                "contrast_token_ids": contrast_token_ids,
                "prompt_template_name": "proxy_forced_choice_template",
                "model_id": str(cfg.get("model_id", "")),
            },
            "mean_target_margin_drop": float(target_margin_drop.mean()) if target_margin_drop.size else float("nan"),
            "mean_random_margin_drop": float(mean_random_margin_drop.mean()) if mean_random_margin_drop.size else float("nan"),
            "std_target_margin_drop": float(target_margin_drop.std(ddof=1)) if target_margin_drop.size > 1 else 0.0,
            "std_random_margin_drop": float(mean_random_margin_drop.std(ddof=1)) if mean_random_margin_drop.size > 1 else 0.0,
            "causal_selectivity": causal_selectivity,
            "causal_selectivity_ci_low": ci_low,
            "causal_selectivity_ci_high": ci_high,
            "passes_positive_causality": passes,
            "random_control_draws": random_control_draws,
            "random_run_mean_margin_drops": random_run_mean_margin_drops,
            "random_run_mean_margin_drop_quantiles": {
                "q05": float(np.quantile(np.asarray(random_run_mean_margin_drops, dtype=np.float64), 0.05)) if random_run_mean_margin_drops else float("nan"),
                "q50": float(np.quantile(np.asarray(random_run_mean_margin_drops, dtype=np.float64), 0.50)) if random_run_mean_margin_drops else float("nan"),
                "q95": float(np.quantile(np.asarray(random_run_mean_margin_drops, dtype=np.float64), 0.95)) if random_run_mean_margin_drops else float("nan"),
            },
            "off_target_proxy_slug": args.paired_proxy_slug,
            "off_target_n_samples": len(off_target_sample_rows),
            "mean_off_target_margin_drop": paired_off_target_effect.get("mean_target_margin_drop"),
            "mean_off_target_random_margin_drop": paired_off_target_effect.get("mean_random_margin_drop"),
            "off_target_selectivity": off_target_selectivity,
            "off_target_selectivity_ci_low": off_target_ci_low,
            "off_target_selectivity_ci_high": off_target_ci_high,
            "sample_rows": sample_rows,
            "off_target_sample_rows": off_target_sample_rows,
            "off_target_proxy_effects": off_target_effects,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
