from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from aiforge.modeling import get_layer_module
from aiforge.sae import SAEBackend


@dataclass
class SAEFeatureEditor:
    sae: SAEBackend
    feature_idx: np.ndarray

    def edit_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.detach().to(torch.float32).cpu().numpy()
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        edited = self.sae.ablate_features(x2d, self.feature_idx)
        edited = edited.reshape(shape)
        edited_t = torch.from_numpy(edited).to(hidden.device, dtype=hidden.dtype)
        return edited_t


def _resolve_target_token_id(tokenizer: AutoTokenizer, target_token: str) -> int:
    token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Target token '{target_token}' maps to {len(token_ids)} tokens ({token_ids}). "
            "Use a string that maps to exactly one token."
        )
    return int(token_ids[0])


def _target_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    target_token_id: int,
    device: torch.device,
    max_length: int,
) -> float:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded, use_cache=False)
        logits = out.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        return float(log_probs[0, target_token_id].item())


def run_pilot_ablation(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompts: list[str],
    target_token: str,
    layer: int,
    editor: SAEFeatureEditor,
    max_length: int,
) -> pd.DataFrame:
    target_token_id = _resolve_target_token_id(tokenizer, target_token)

    layer_module = get_layer_module(model, layer)

    rows: list[dict] = []
    for prompt in tqdm(prompts, desc="Pilot ablation"):
        baseline = _target_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_token_id=target_token_id,
            device=device,
            max_length=max_length,
        )

        def hook_fn(_module, _inputs, output):
            if isinstance(output, tuple):
                edited_hidden = editor.edit_hidden(output[0])
                return (edited_hidden, *output[1:])
            return editor.edit_hidden(output)

        handle = layer_module.register_forward_hook(hook_fn)
        try:
            ablated = _target_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_token_id=target_token_id,
                device=device,
                max_length=max_length,
            )
        finally:
            handle.remove()

        rows.append(
            {
                "prompt": prompt,
                "target_token": target_token,
                "baseline_logprob": baseline,
                "ablated_logprob": ablated,
                "delta_logprob": baseline - ablated,
            }
        )
    return pd.DataFrame(rows)
