from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_label_token_ids(
    tokenizer: AutoTokenizer,
    label_tokens: dict[str, str],
    *,
    strict_single_token: bool = True,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for label, tok in label_tokens.items():
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if strict_single_token and len(ids) != 1:
            raise ValueError(
                f"Label token '{tok}' for label '{label}' maps to {len(ids)} tokens ({ids})."
            )
        if not ids:
            raise ValueError(f"Label token '{tok}' for label '{label}' mapped to no token ids.")
        out[label] = int(ids[0])
    return out


def compute_label_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    label_tokens: dict[str, str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    label_to_id = _resolve_label_token_ids(tokenizer, label_tokens)
    label_names = list(label_to_id.keys())

    rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, use_cache=False)
            logits = out.logits
            lengths = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(logits.size(0), device=device)
            final_logits = logits[batch_idx, lengths]
            log_probs = torch.log_softmax(final_logits, dim=-1)

        for i in range(len(batch_prompts)):
            row: dict[str, Any] = {
                "prompt_index": start + i,
                "prompt": batch_prompts[i],
            }
            vals = []
            for label in label_names:
                tid = label_to_id[label]
                v = float(log_probs[i, tid].item())
                row[f"logprob_{label}"] = v
                vals.append(v)
            probs = np.exp(np.array(vals) - np.max(vals))
            probs = probs / probs.sum()
            for label, p in zip(label_names, probs.tolist()):
                row[f"prob_{label}"] = float(p)
            rows.append(row)
    return rows
