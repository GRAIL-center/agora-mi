from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging
from analysis.bootstrap import bootstrap_ci
from analysis.permutation import paired_permutation_sign_flip_test
from data.io import read_jsonl
from interventions.ablate import random_feature_ids, sequence_logprob
from model.hooks import layer_overwrite_hook
from model.load_model import load_model_bundle
from model.prompt import build_prompts, load_prompt_config, render_prompt
from sae.load_sae import load_sae_for_layer


def _resolve_token_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Label text maps to no tokens: {text!r}")
    return [int(i) for i in ids]


def _label_prob(logp_inc: float, logp_res: float) -> float:
    m = max(logp_inc, logp_res)
    a = np.exp(logp_inc - m)
    b = np.exp(logp_res - m)
    return float(a / (a + b))


def _sequence_logprob_from_logits(
    logits: torch.Tensor,
    *,
    prompt_pos: int,
    target_token_ids: list[int],
) -> torch.Tensor:
    # logits: [1, seq, vocab]
    device = logits.device
    pos = torch.arange(prompt_pos, prompt_pos + len(target_token_ids), device=device, dtype=torch.long)
    step_logits = logits[0, pos, :]
    log_probs = torch.log_softmax(step_logits, dim=-1)
    target = torch.tensor(target_token_ids, dtype=torch.long, device=device)
    gathered = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return gathered.sum()


def _editor_factory(
    *,
    sae,
    feature_ids: list[int],
    q95: np.ndarray,
    alpha: float,
    mode: str,
    device: torch.device,
):
    idx = torch.tensor(feature_ids, dtype=torch.long, device=device)
    q95_t = torch.tensor(q95, dtype=torch.float32, device=device)

    def editor(selected: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
        f = sae.encode(selected.to(torch.float32))
        out = f.clone()
        if mode == "cap":
            thresh = alpha * q95_t[idx]
            out[:, idx] = torch.minimum(out[:, idx], thresh.unsqueeze(0))
        elif mode in {"scale", "scale_down"}:
            out[:, idx] = alpha * out[:, idx]
        else:
            raise ValueError(f"Unknown clamp mode: {mode}")
        return sae.decode(out).to(selected.dtype)

    return editor


def _prompt_innov_mass(
    *,
    model,
    tokenizer,
    prompt: str,
    vocab_ids: list[int],
    device: torch.device,
    max_length: int,
    layer: int,
    editor=None,
) -> float:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_pos = int(enc["attention_mask"].sum(dim=1).item() - 1)

    if editor is None:
        with torch.no_grad():
            out = model(**enc, use_cache=False)
            logits = out.logits[0, prompt_pos, :]
    else:
        with layer_overwrite_hook(
            model,
            layer=layer,
            editor=editor,
            attention_mask=enc["attention_mask"],
            token_index=prompt_pos,
        ):
            with torch.no_grad():
                out = model(**enc, use_cache=False)
                logits = out.logits[0, prompt_pos, :]

    probs = torch.softmax(logits, dim=-1)
    vid = torch.tensor(vocab_ids, dtype=torch.long, device=device)
    return float(probs[vid].sum().item())


def _pick_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate paths exists: {[str(x) for x in paths]}")


def _mode_seed_offset(mode: str) -> int:
    # Stable offset to avoid identical bootstrap/permutation streams across modes.
    return sum((i + 1) * ord(ch) for i, ch in enumerate(mode))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--innov_jsonl", default="data/processed/dinnov_test.jsonl")
    parser.add_argument("--selection_tag", default="train")
    parser.add_argument("--topk_safe_json", default=None)
    parser.add_argument("--delta_csv", default=None)
    parser.add_argument("--features_npz", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--alphas", default="0.5,1.0,1.5")
    parser.add_argument("--modes", default="cap")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--sae_npz", default=None)
    parser.add_argument("--bootstrap_B", type=int, default=None)
    parser.add_argument("--perm_N", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    setup_logging("interference_clamp")
    cfg = read_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    np.random.seed(seed)

    results_dir = Path(cfg.get("results_dir", "results"))
    pol_dir = results_dir / "polarization"
    topk_path = Path(args.topk_safe_json) if args.topk_safe_json else _pick_existing(
        [
            pol_dir / f"layer{args.layer}_{args.selection_tag}_topk_safe.json",
            pol_dir / f"layer{args.layer}_topk_safe.json",
        ]
    )
    delta_path = Path(args.delta_csv) if args.delta_csv else _pick_existing(
        [
            pol_dir / f"layer{args.layer}_{args.selection_tag}_delta.csv",
            pol_dir / f"layer{args.layer}_delta.csv",
        ]
    )
    feat_path = Path(args.features_npz) if args.features_npz else _pick_existing(
        [
            pol_dir / f"layer{args.layer}_{args.selection_tag}_features.npz",
            pol_dir / f"layer{args.layer}_dev_features.npz",
            pol_dir / f"layer{args.layer}_features.npz",
        ]
    )

    topk_obj = json.loads(topk_path.read_text(encoding="utf-8"))
    safe_ids_all = [int(x) for x in topk_obj["feature_ids"]]
    k = int(args.k if args.k is not None else cfg.get("topk", 64))
    safe_ids = safe_ids_all[:k]
    if not safe_ids:
        raise ValueError("No safe feature IDs found for clamp.")

    delta_df = pd.read_csv(delta_path)
    neutral_ids = (
        delta_df.assign(abs_delta=lambda d: d["delta"].abs())
        .sort_values("abs_delta", ascending=True)["feature_id"]
        .astype(int)
        .head(k)
        .tolist()
    )

    arr = np.load(feat_path, allow_pickle=True)
    safe_f = np.asarray(arr["safe_features"], dtype=np.float32)
    innov_f = np.asarray(arr["innov_features"], dtype=np.float32)
    all_f = np.concatenate([safe_f, innov_f], axis=0)
    q95 = np.quantile(all_f, 0.95, axis=0).astype(np.float32)

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])
    target_tokens = prompt_cfg.get("target_tokens", {})
    inc_label = str(target_tokens.get("INCENTIVE", "INCENTIVE"))
    res_label = str(target_tokens.get("RESTRICTION", "RESTRICTION"))
    innovation_vocab = [str(x) for x in prompt_cfg.get("innovation_vocab", [])]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    modes = [str(x).strip() for x in args.modes.split(",") if str(x).strip()]
    if not modes:
        raise ValueError("No clamp modes supplied.")
    allowed_modes = {"cap", "scale", "scale_down"}
    bad_modes = sorted(set(modes) - allowed_modes)
    if bad_modes:
        raise ValueError(f"Unsupported mode(s): {bad_modes}. Allowed: {sorted(allowed_modes)}")

    rows = read_jsonl(args.innov_jsonl)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    bundle = load_model_bundle(cfg)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device

    use_chat_template = cfg.get("use_chat_template", False)
    prompts = build_prompts(rows, template, tokenizer=tokenizer, use_chat_template=use_chat_template)
    ids = [str(r.get("id", i)) for i, r in enumerate(rows)]
    if not prompts:
        raise ValueError("No innovation prompts to evaluate.")
    sae = load_sae_for_layer(cfg, layer=args.layer, device=device, npz_path=args.sae_npz)
    inc_ids = _resolve_token_ids(tokenizer, inc_label)
    res_ids = _resolve_token_ids(tokenizer, res_label)
    vocab_ids = []
    for w in innovation_vocab:
        ids_w = tokenizer.encode(w, add_special_tokens=False)
        if ids_w:
            vocab_ids.append(int(ids_w[0]))
    vocab_ids = sorted(set(vocab_ids))
    max_length = int(cfg.get("max_length", 1024))

    random_ids = random_feature_ids(
        d_sae=sae.d_sae,
        k=k,
        exclude=safe_ids,
        seed=seed,
    )

    out_rows = []
    for i, prompt in enumerate(prompts):
        lp_inc_base, encoded_inc, prompt_pos_inc = sequence_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=inc_ids,
            device=device,
            max_length=max_length,
        )
        lp_res_base, _, _ = sequence_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_ids=res_ids,
            device=device,
            max_length=max_length,
        )
        p_inc_base = _label_prob(lp_inc_base, lp_res_base)
        mass_base = _prompt_innov_mass(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            vocab_ids=vocab_ids or [inc_ids[0]],
            device=device,
            max_length=max_length,
            layer=args.layer,
            editor=None,
        )

        for mode in modes:
            for cond, feat_ids in (
                ("safe", safe_ids),
                ("random", random_ids),
                ("neutral", neutral_ids),
            ):
                for alpha in alphas:
                    editor = _editor_factory(
                        sae=sae,
                        feature_ids=feat_ids,
                        q95=q95,
                        alpha=alpha,
                        mode=mode,
                        device=device,
                    )

                    with layer_overwrite_hook(
                        model,
                        layer=args.layer,
                        editor=editor,
                        attention_mask=encoded_inc["attention_mask"],
                        token_index=prompt_pos_inc,
                    ):
                        with torch.no_grad():
                            out_inc = model(**encoded_inc, use_cache=False)
                            lp_inc_clamp = float(
                                _sequence_logprob_from_logits(
                                    out_inc.logits,
                                    prompt_pos=prompt_pos_inc,
                                    target_token_ids=inc_ids,
                                ).item()
                            )

                    # Need separate encoded for RESTRICTION sequence
                    _, encoded_res, prompt_pos_res = sequence_logprob(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        target_token_ids=res_ids,
                        device=device,
                        max_length=max_length,
                    )
                    with layer_overwrite_hook(
                        model,
                        layer=args.layer,
                        editor=editor,
                        attention_mask=encoded_res["attention_mask"],
                        token_index=prompt_pos_res,
                    ):
                        with torch.no_grad():
                            out_res = model(**encoded_res, use_cache=False)
                            lp_res_clamp = float(
                                _sequence_logprob_from_logits(
                                    out_res.logits,
                                    prompt_pos=prompt_pos_res,
                                    target_token_ids=res_ids,
                                ).item()
                            )

                    p_inc_clamp = _label_prob(lp_inc_clamp, lp_res_clamp)
                    mass_clamp = _prompt_innov_mass(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        vocab_ids=vocab_ids or [inc_ids[0]],
                        device=device,
                        max_length=max_length,
                        layer=args.layer,
                        editor=editor,
                    )

                    out_rows.append(
                        {
                            "id": ids[i],
                            "mode": mode,
                            "condition": cond,
                            "alpha": alpha,
                            "prob_inc_base": p_inc_base,
                            "prob_inc_clamp": p_inc_clamp,
                            "delta_prob_inc": p_inc_clamp - p_inc_base,
                            "logprob_inc_base": lp_inc_base,
                            "logprob_inc_clamp": lp_inc_clamp,
                            "delta_logprob_inc": lp_inc_clamp - lp_inc_base,
                            "innov_mass_base": mass_base,
                            "innov_mass_clamp": mass_clamp,
                            "delta_mass": mass_clamp - mass_base,
                        }
                    )

    df = pd.DataFrame(out_rows)
    B = int(args.bootstrap_B if args.bootstrap_B is not None else cfg.get("bootstrap_B", 2000))
    perm_N = int(args.perm_N if args.perm_N is not None else cfg.get("perm_N", 10000))

    summary_rows = []
    for (mode, alpha, cond), g in df.groupby(["mode", "alpha", "condition"], sort=True):
        dprob = g["delta_prob_inc"].to_numpy(dtype=np.float64)
        dlogp = g["delta_logprob_inc"].to_numpy(dtype=np.float64)
        dmass = g["delta_mass"].to_numpy(dtype=np.float64)
        so = _mode_seed_offset(mode)
        ci_prob = bootstrap_ci(dprob, B=B, alpha=0.05, seed=seed + so + int(alpha * 100) + 1)
        ci_logp = bootstrap_ci(dlogp, B=B, alpha=0.05, seed=seed + so + int(alpha * 100) + 2)
        ci_mass = bootstrap_ci(dmass, B=B, alpha=0.05, seed=seed + so + int(alpha * 100) + 3)
        summary_rows.append(
            {
                "mode": mode,
                "alpha": alpha,
                "condition": cond,
                "n": int(len(g)),
                "mean_delta_prob_inc": float(dprob.mean()),
                "ci95_delta_prob_inc": [ci_prob[0], ci_prob[1]],
                "mean_delta_logprob_inc": float(dlogp.mean()),
                "ci95_delta_logprob_inc": [ci_logp[0], ci_logp[1]],
                "mean_delta_mass": float(dmass.mean()),
                "ci95_delta_mass": [ci_mass[0], ci_mass[1]],
            }
        )

    # Paired tests: safe vs controls on primary metric (delta_prob_inc), per alpha.
    paired_tests = []
    for mode in sorted(df["mode"].unique()):
        for alpha in sorted(df[df["mode"] == mode]["alpha"].unique()):
            sub = df[(df["mode"] == mode) & (df["alpha"] == alpha)]
            piv = sub.pivot(index="id", columns="condition", values="delta_prob_inc")
            for ctrl in ("random", "neutral"):
                if "safe" in piv.columns and ctrl in piv.columns:
                    d = (piv["safe"] - piv[ctrl]).dropna().to_numpy(dtype=np.float64)
                    so = _mode_seed_offset(mode)
                    ci = bootstrap_ci(d, B=B, alpha=0.05, seed=seed + so + int(alpha * 100) + 11)
                    p = paired_permutation_sign_flip_test(d, N=perm_N, seed=seed + so + int(alpha * 100) + 12)
                    paired_tests.append(
                        {
                            "mode": mode,
                            "alpha": float(alpha),
                            "compare": f"safe_minus_{ctrl}",
                            "n": int(d.size),
                            "mean_delta": float(d.mean()) if d.size else float("nan"),
                            "ci95": [ci[0], ci[1]],
                            "perm_p": float(p),
                        }
                    )

    out_dir = ensure_dir(results_dir / "interference")
    sample_path = out_dir / f"clamp_samples_layer{args.layer}_{args.selection_tag}.csv"
    summary_path = out_dir / f"clamp_summary_layer{args.layer}_{args.selection_tag}.json"
    df.to_csv(sample_path, index=False)
    save_with_metadata(
        output_path=summary_path,
        payload={
            "layer": args.layer,
            "selection_tag": args.selection_tag,
            "k": k,
            "alphas": alphas,
            "modes": modes,
            "safe_feature_ids": safe_ids,
            "random_feature_ids": random_ids,
            "neutral_feature_ids": neutral_ids,
            "summary_rows": summary_rows,
            "paired_tests_primary_metric": paired_tests,
            "sample_csv": str(sample_path),
        },
        config={"run": cfg, "args": vars(args)},
    )
    logging.info("Saved clamp samples: %s", sample_path)
    logging.info("Saved clamp summary: %s", summary_path)
    print(sample_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
