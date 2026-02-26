from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.hooks import layer_overwrite_hook
from sae.load_sae import LoadedSAE


def ablate_features(features: torch.Tensor, feature_ids: Iterable[int]) -> torch.Tensor:
    out = features.clone()
    if out.ndim != 2:
        raise ValueError(f"Expected features shape [batch, d_sae], got {tuple(out.shape)}")
    idx = torch.tensor(list(feature_ids), dtype=torch.long, device=out.device)
    if idx.numel() == 0:
        return out
    out[:, idx] = 0.0
    return out


def sequence_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    target_token_ids: list[int],
    *,
    device: torch.device,
    max_length: int,
) -> tuple[float, dict[str, torch.Tensor], int]:
    if not target_token_ids:
        raise ValueError("target_token_ids is empty")

    max_prompt_len = max_length - len(target_token_ids)
    if max_prompt_len < 1:
        raise ValueError("max_length is too small for target token sequence.")
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_len,
    )
    if not prompt_ids:
        raise ValueError("Prompt tokenization produced empty input.")

    full_ids = prompt_ids + target_token_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
    prompt_pos = len(prompt_ids) - 1

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
        v = float(gathered.sum().item())
    return v, encoded, prompt_pos


def _next_token_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    target_token_ids: list[int],
    *,
    device: torch.device,
    max_length: int,
) -> tuple[float, dict[str, torch.Tensor], int]:
    # Backward-compatible alias used by older call sites.
    return sequence_logprob(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_token_ids=target_token_ids,
        device=device,
        max_length=max_length,
    )


def intervention_logprob(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: LoadedSAE,
    prompt: str,
    target_token_ids: list[int],
    layer: int,
    feature_ids: list[int],
    device: torch.device,
    max_length: int,
) -> tuple[float, float]:
    baseline, encoded, prompt_pos = sequence_logprob(
        model, tokenizer, prompt, target_token_ids, device=device, max_length=max_length
    )
    if not feature_ids:
        return baseline, baseline

    feat_idx = torch.tensor(feature_ids, dtype=torch.long, device=device)

    def editor(selected: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
        # selected shape: [batch, d_model]
        f = sae.encode(selected.to(torch.float32))
        f_ab = f.clone()
        f_ab[:, feat_idx] = 0.0
        return sae.decode(f_ab).to(selected.dtype)

    with layer_overwrite_hook(
        model,
        layer=layer,
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
            ablated = float(gathered.sum().item())
    return baseline, ablated


def random_feature_ids(
    *,
    d_sae: int,
    k: int,
    exclude: list[int] | None,
    seed: int,
) -> list[int]:
    ex = set(exclude or [])
    pool = [i for i in range(d_sae) if i not in ex]
    if not pool:
        return []
    rng = np.random.default_rng(seed)
    kk = min(k, len(pool))
    return rng.choice(pool, size=kk, replace=False).astype(int).tolist()
