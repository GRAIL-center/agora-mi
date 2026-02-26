from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
from transformers import AutoModelForCausalLM


def _lookup_layers(model: AutoModelForCausalLM):
    candidates: list[tuple[str, ...]] = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for key in path:
            if not hasattr(obj, key):
                ok = False
                break
            obj = getattr(obj, key)
        if ok:
            return obj
    raise ValueError("Cannot find transformer layer list on model.")


def get_layer_module(model: AutoModelForCausalLM, layer: int):
    layers = _lookup_layers(model)
    n = len(layers)
    if layer < 0:
        layer = n + layer
    if layer < 0 or layer >= n:
        raise ValueError(f"Layer {layer} is out of range for model with {n} layers.")
    return layers[layer]


def _resolve_token_positions(
    attention_mask: torch.Tensor,
    token_index: int | str,
) -> torch.Tensor:
    if token_index == "last":
        return attention_mask.sum(dim=1) - 1
    idx = int(token_index)
    if idx < 0:
        lengths = attention_mask.sum(dim=1)
        return lengths + idx
    return torch.full((attention_mask.size(0),), idx, device=attention_mask.device, dtype=torch.long)


def capture_residual_stream(
    model: AutoModelForCausalLM,
    *,
    layer: int,
    token_index: int | str,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
    hs_idx = layer + 1
    if hs_idx >= len(hidden_states):
        raise ValueError(f"Hidden state index {hs_idx} unavailable for hidden-state len {len(hidden_states)}")
    layer_states = hidden_states[hs_idx]
    pos = _resolve_token_positions(inputs["attention_mask"], token_index)
    batch_idx = torch.arange(layer_states.size(0), device=layer_states.device)
    return layer_states[batch_idx, pos]


@contextmanager
def layer_overwrite_hook(
    model: AutoModelForCausalLM,
    *,
    layer: int,
    editor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    attention_mask: torch.Tensor,
    token_index: int | str = "last",
):
    layer_module = get_layer_module(model, layer)
    positions = _resolve_token_positions(attention_mask, token_index)

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()

        edited = hidden.clone()
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        selected = hidden[batch_idx, positions]
        edited_selected = editor(selected, positions)
        edited[batch_idx, positions] = edited_selected

        if isinstance(output, tuple):
            return (edited, *rest)
        return edited

    handle = layer_module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
