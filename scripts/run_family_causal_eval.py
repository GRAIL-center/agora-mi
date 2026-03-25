from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from data.io import read_jsonl
from interventions.ablate import random_feature_ids, sequence_logprob
from model.hooks import capture_layer_site_sequence, layer_overwrite_hook, pool_sequence_activations
from model.load_model import load_model_bundle
from model.prompt import load_prompt_config, render_prompt
from sae.encode import encode_features
from sae.load_sae import load_sae_for_layer


def _resolve_label_ids(tokenizer, prompt_cfg: dict, family_keys: list[str]) -> dict[str, list[int]]:
    label_map = prompt_cfg.get("family_target_tokens", {})
    out: dict[str, list[int]] = {}
    for family_key in family_keys:
        label = str(label_map[family_key])
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Label text maps to no tokens: {label}")
        out[family_key] = [int(i) for i in ids]
    return out


def _compute_q95(
    *,
    rows: list[dict],
    model,
    tokenizer,
    device,
    sae,
    template: str,
    layer: int,
    site: str,
    pooling: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    prompts = [render_prompt(template, row["text"]) for row in rows]
    all_feats = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h_seq = capture_layer_site_sequence(model, layer=layer, site=site, inputs=enc)
            pooled = pool_sequence_activations(h_seq, attention_mask=enc["attention_mask"], pooling=pooling)
            feats = encode_features(sae, pooled.to(torch.float32))
        all_feats.append(feats.detach().cpu().numpy())
    return np.quantile(np.concatenate(all_feats, axis=0), 0.95, axis=0).astype(np.float32)


def _editor_factory(*, sae, feature_ids: list[int], mode: str, q95: np.ndarray | None, alpha: float, device) -> callable:
    idx = torch.tensor(feature_ids, dtype=torch.long, device=device)
    q95_tensor = None if q95 is None else torch.tensor(q95, dtype=torch.float32, device=device)

    def editor(selected: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
        feats = sae.encode(selected.to(torch.float32))
        edited = feats.clone()
        if mode == "ablate":
            edited[:, idx] = 0.0
        elif mode == "clamp":
            if q95_tensor is None:
                raise ValueError("q95 is required for clamp mode.")
            edited[:, idx] = torch.minimum(edited[:, idx], alpha * q95_tensor[idx].unsqueeze(0))
        else:
            raise ValueError(f"Unsupported mode: {mode}")
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
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", default="ablate", choices=["ablate", "clamp"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--sae_npz", default=None)
    args = parser.parse_args()

    setup_logging("run_family_causal_eval")
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
        / "family_causal"
        / args.family
        / f"layer{args.layer}_{args.site}"
    )
    core_obj = json.loads(core_path.read_text(encoding="utf-8"))
    core_feature_ids = [int(row["feature_id"]) for row in core_obj.get("core_features", [])]
    if not core_feature_ids:
        save_with_metadata(
            output_path=out_dir / f"{args.mode}_{args.split}.json",
            payload={
                "family_name": args.family,
                "mode": args.mode,
                "alpha": args.alpha,
                "split": args.split,
                "skipped": True,
                "skip_reason": "family_core_bank_empty",
            },
            config={"run": cfg, "args": vars(args)},
        )
        return 0

    rows = read_jsonl(Path(args.manifest_root) / args.family / "family_pools" / "main" / f"{args.split}.jsonl")[: args.max_samples]
    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/policy_text_prompt.yaml"))
    forced_choice_template = str(prompt_cfg["family_forced_choice_template"])
    family_keys = list(prompt_cfg.get("family_target_tokens", {}).keys())
    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    label_ids = _resolve_label_ids(tokenizer, prompt_cfg, family_keys)
    sae = load_sae_for_layer(cfg, layer=args.layer, site=args.site, device=device, npz_path=args.sae_npz)

    q95 = None
    if args.mode == "clamp":
        q95 = _compute_q95(
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            device=device,
            sae=sae,
            template=str(prompt_cfg["template_v1"]),
            layer=args.layer,
            site=args.site,
            pooling=str(cfg.get("pooling", "mean")),
            batch_size=int(cfg.get("batch_size", 4)),
            max_length=int(cfg.get("max_length", 1024)),
        )

    family_editor = _editor_factory(
        sae=sae,
        feature_ids=core_feature_ids,
        mode=args.mode,
        q95=q95,
        alpha=args.alpha,
        device=device,
    )
    random_ids = random_feature_ids(d_sae=sae.d_sae, k=len(core_feature_ids), exclude=core_feature_ids, seed=int(cfg.get("seed", 0)))
    random_editor = _editor_factory(
        sae=sae,
        feature_ids=random_ids,
        mode=args.mode,
        q95=q95,
        alpha=args.alpha,
        device=device,
    )

    sample_rows = []
    for row in rows:
        prompt = render_prompt(forced_choice_template, row["text"])
        baseline_scores = {
            family_key: _intervened_label_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_token_ids=label_ids[family_key],
                editor=None,
                layer=args.layer,
                site=args.site,
                device=device,
                max_length=int(cfg.get("max_length", 1024)),
            )
            for family_key in family_keys
        }
        family_scores = {
            family_key: _intervened_label_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_token_ids=label_ids[family_key],
                editor=family_editor,
                layer=args.layer,
                site=args.site,
                device=device,
                max_length=int(cfg.get("max_length", 1024)),
            )
            for family_key in family_keys
        }
        random_scores = {
            family_key: _intervened_label_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_token_ids=label_ids[family_key],
                editor=random_editor,
                layer=args.layer,
                site=args.site,
                device=device,
                max_length=int(cfg.get("max_length", 1024)),
            )
            for family_key in family_keys
        }
        sorted_baseline = sorted((v, k) for k, v in baseline_scores.items())
        top_distractor = sorted_baseline[-2][1] if len(sorted_baseline) > 1 else args.family
        sample_rows.append(
            {
                "segment_id": row["segment_id"],
                "correct_family": args.family,
                "baseline_correct": baseline_scores[args.family],
                "family_edit_correct": family_scores[args.family],
                "random_edit_correct": random_scores[args.family],
                "baseline_margin": baseline_scores[args.family] - baseline_scores[top_distractor],
                "family_edit_margin": family_scores[args.family] - family_scores[top_distractor],
                "random_edit_margin": random_scores[args.family] - random_scores[top_distractor],
            }
        )

    correct_deltas = np.asarray([row["family_edit_correct"] - row["baseline_correct"] for row in sample_rows], dtype=np.float64)
    random_correct_deltas = np.asarray([row["random_edit_correct"] - row["baseline_correct"] for row in sample_rows], dtype=np.float64)
    margin_deltas = np.asarray([row["family_edit_margin"] - row["baseline_margin"] for row in sample_rows], dtype=np.float64)
    random_margin_deltas = np.asarray([row["random_edit_margin"] - row["baseline_margin"] for row in sample_rows], dtype=np.float64)

    save_with_metadata(
        output_path=out_dir / f"{args.mode}_{args.split}.json",
        payload={
            "family_name": args.family,
            "mode": args.mode,
            "alpha": args.alpha,
            "split": args.split,
            "n_samples": len(sample_rows),
            "mean_correct_delta": float(correct_deltas.mean()) if correct_deltas.size else float("nan"),
            "mean_random_correct_delta": float(random_correct_deltas.mean()) if random_correct_deltas.size else float("nan"),
            "mean_margin_delta": float(margin_deltas.mean()) if margin_deltas.size else float("nan"),
            "mean_random_margin_delta": float(random_margin_deltas.mean()) if random_margin_deltas.size else float("nan"),
            "sample_rows": sample_rows,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
