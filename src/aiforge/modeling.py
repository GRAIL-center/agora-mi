from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype | None:
    dtype = dtype.lower()
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_causal_lm(model_name: str, device: str = "auto", dtype: str = "auto") -> LoadedModel:
    resolved_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(resolved_device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, device=resolved_device)


def _lookup_layers(model: AutoModelForCausalLM):
    candidates: list[tuple[str, ...]] = [
        ("model", "layers"),  # Gemma/Llama-like
        ("model", "decoder", "layers"),  # BART/T5 style
        ("transformer", "h"),  # GPT2 style
        ("gpt_neox", "layers"),  # GPT-NeoX style
    ]

    for path in candidates:
        obj = model
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok:
            return obj
    raise ValueError(
        "Unable to locate transformer block list on model. "
        "Expected one of model.layers / model.decoder.layers / transformer.h / gpt_neox.layers."
    )


def get_layer_module(model: AutoModelForCausalLM, layer: int):
    layers = _lookup_layers(model)
    n_layers = len(layers)
    if layer < 0:
        layer = n_layers + layer
    if not (0 <= layer < n_layers):
        raise ValueError(f"Layer index {layer} out of range for model with {n_layers} layers")
    return layers[layer]


def extract_last_token_residuals(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    texts: Iterable[str],
    layer: int,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_residuals: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []

    text_list = list(texts)
    for start in tqdm(range(0, len(text_list), batch_size), desc="Extract residuals"):
        batch = text_list[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            hs_index = layer + 1
            if hs_index >= len(hidden_states):
                raise ValueError(
                    f"Hidden state index {hs_index} not available (len={len(hidden_states)}). "
                    "Pick a smaller layer."
                )
            layer_states = hidden_states[hs_index]
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(layer_states.size(0), device=device)
            pooled = layer_states[batch_idx, lengths]

        all_residuals.append(pooled.to(torch.float32).cpu().numpy())
        all_lengths.append(lengths.cpu().numpy())

    residuals = np.concatenate(all_residuals, axis=0) if all_residuals else np.zeros((0, 0))
    lengths = np.concatenate(all_lengths, axis=0) if all_lengths else np.zeros((0,), dtype=np.int64)
    return residuals, lengths
