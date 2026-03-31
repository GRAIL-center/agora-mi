from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.bootstrap import bootstrap_ci
from benchmark.methods import InternalFeatureExtractor, _logreg_model, _rows_to_texts
from data.io import read_jsonl
from interventions.ablate import sequence_logprob
from model.hooks import layer_overwrite_hook
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config


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


def _dense_editor_factory(*, dimension_ids: list[int], device) -> callable:
    idx = torch.tensor(dimension_ids, dtype=torch.long, device=device)

    def editor(selected: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
        edited = selected.clone()
        if edited.dim() == 2:
            edited[:, idx] = 0.0
            return edited
        if edited.dim() == 3:
            edited[:, :, idx] = 0.0
            return edited
        raise ValueError(f"Unexpected dense editor input rank: {edited.dim()}")
        return edited

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


def _random_dimension_ids(hidden_size: int, k: int, *, exclude: list[int], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    candidates = [index for index in range(hidden_size) if index not in set(exclude)]
    if len(candidates) < k:
        return candidates
    return [int(value) for value in rng.choice(np.asarray(candidates, dtype=np.int64), size=k, replace=False).tolist()]


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
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    setup_logging("run_dense_subspace_control_eval")
    cfg = read_config(args.config)
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    template = str(prompt_cfg["proxy_forced_choice_template"])
    target_label = str(prompt_cfg["proxy_target_tokens"][args.proxy_slug])
    contrast_label = str(prompt_cfg["proxy_target_tokens"][args.paired_proxy_slug])

    train_pos = read_jsonl(Path(args.manifest_root) / args.family / "proxies" / args.proxy_slug / "train.jsonl")
    train_neg = read_jsonl(Path(args.manifest_root) / args.family / "negatives" / args.proxy_slug / "train.jsonl")
    rows = read_jsonl(Path(args.manifest_root) / args.family / "proxies" / args.proxy_slug / f"{args.split}.jsonl")
    rows = rows[: int(args.max_samples)]
    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "dense_control"
        / args.family
        / args.proxy_slug
        / f"layer{args.layer}_{args.site}"
    )

    if not rows or not train_pos or not train_neg:
        save_with_metadata(
            output_path=out_dir / f"top{int(args.k)}_{args.split}.json",
            payload={
                "family_name": args.family,
                "proxy_slug": args.proxy_slug,
                "paired_proxy_slug": args.paired_proxy_slug,
                "split": args.split,
                "effective_k": int(args.k),
                "skipped": True,
                "skip_reason": "missing_rows",
            },
            config={"run": cfg, "args": vars(args)},
        )
        return 0

    extractor = InternalFeatureExtractor(cfg, site=args.site, use_sae=False)
    x_train = np.concatenate(
        [
            extractor.extract(train_pos, layer=int(args.layer), pooling="mean"),
            extractor.extract(train_neg, layer=int(args.layer), pooling="mean"),
        ],
        axis=0,
    )
    y_train = np.concatenate([np.ones(len(train_pos), dtype=np.int64), np.zeros(len(train_neg), dtype=np.int64)])
    model = _logreg_model()
    model.fit(x_train, y_train)
    coef = np.asarray(model.coef_[0], dtype=np.float64)
    ranked_dims = np.argsort(-np.abs(coef))
    dimension_ids = [int(value) for value in ranked_dims[: int(args.k)].tolist()]

    bundle = load_model_bundle(cfg)
    base_model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    target_token_ids = _resolve_token_ids(tokenizer, target_label)
    contrast_token_ids = _resolve_token_ids(tokenizer, contrast_label)
    editor = _dense_editor_factory(dimension_ids=dimension_ids, device=device)
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
        model=base_model,
        tokenizer=tokenizer,
        editor=editor,
        layer=int(args.layer),
        site=str(args.site),
        device=device,
        max_length=max_length,
    )
    target_margin_drop = baseline_margins - ablated_margins

    random_run_mean_margin_drops: list[float] = []
    random_control_draws: list[dict[str, object]] = []
    random_margin_drop_runs: list[np.ndarray] = []
    hidden_size = int(x_train.shape[1])
    for draw_index in range(int(args.random_sets)):
        random_ids = _random_dimension_ids(
            hidden_size,
            int(args.k),
            exclude=dimension_ids,
            seed=int(cfg.get("seed", 0)) + draw_index,
        )
        random_editor = _dense_editor_factory(dimension_ids=random_ids, device=device)
        _, random_ablated_margins, _ = _margin_series(
            rows=rows,
            prompts=prompts,
            target_token_ids=target_token_ids,
            contrast_token_ids=contrast_token_ids,
            model=base_model,
            tokenizer=tokenizer,
            editor=random_editor,
            layer=int(args.layer),
            site=str(args.site),
            device=device,
            max_length=max_length,
        )
        run_drop = baseline_margins - random_ablated_margins
        random_margin_drop_runs.append(run_drop)
        random_run_mean_margin_drops.append(float(run_drop.mean()) if run_drop.size else float("nan"))
        random_control_draws.append(
            {
                "draw_index": int(draw_index),
                "dimension_ids": random_ids,
                "mean_margin_drop": float(run_drop.mean()) if run_drop.size else float("nan"),
                "std_margin_drop": float(run_drop.std(ddof=1)) if run_drop.size > 1 else 0.0,
            }
        )

    random_margin_drop_matrix = np.stack(random_margin_drop_runs, axis=0)
    mean_random_margin_drop = random_margin_drop_matrix.mean(axis=0)
    dense_delta = target_margin_drop - mean_random_margin_drop
    ci_low, ci_high = bootstrap_ci(
        dense_delta,
        B=int(cfg.get("bootstrap_B", 500)),
        alpha=0.05,
        seed=int(cfg.get("seed", 0)) + 41,
    )
    dense_selectivity = float(dense_delta.mean()) if dense_delta.size else float("nan")

    for row_index, row in enumerate(sample_rows):
        row["target_margin_drop"] = float(target_margin_drop[row_index])
        row["mean_random_margin_drop"] = float(mean_random_margin_drop[row_index])
        row["paired_delta"] = float(dense_delta[row_index])

    save_with_metadata(
        output_path=out_dir / f"top{int(args.k)}_{args.split}.json",
        payload={
            "family_name": args.family,
            "proxy_slug": args.proxy_slug,
            "paired_proxy_slug": args.paired_proxy_slug,
            "split": args.split,
            "effective_k": int(args.k),
            "dimension_ids": dimension_ids,
            "intervention": {
                "layer": int(args.layer),
                "site": str(args.site),
                "dimension_ids": dimension_ids,
                "target_token_ids": target_token_ids,
                "contrast_token_ids": contrast_token_ids,
                "prompt_template_name": "proxy_forced_choice_template",
                "model_id": str(cfg.get("model_id", "")),
            },
            "mean_target_margin_drop": float(target_margin_drop.mean()) if target_margin_drop.size else float("nan"),
            "mean_random_margin_drop": float(mean_random_margin_drop.mean()) if mean_random_margin_drop.size else float("nan"),
            "dense_selectivity": dense_selectivity,
            "dense_selectivity_ci_low": ci_low,
            "dense_selectivity_ci_high": ci_high,
            "random_control_draws": random_control_draws,
            "sample_rows": sample_rows,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
