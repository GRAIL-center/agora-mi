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


def _get_attr_path(obj, path: tuple[str, ...]):
    current = obj
    for key in path:
        if not hasattr(current, key):
            return None
        current = getattr(current, key)
    return current


def get_layer_site_module(model: AutoModelForCausalLM, layer: int, site: str):
    layer_module = get_layer_module(model, layer)
    if site in {"resid_pre", "resid_post"}:
        return layer_module
    if site == "attn_out":
        candidates = [
            ("self_attn", "o_proj"),
            ("attn", "c_proj"),
        ]
    elif site == "mlp_out":
        candidates = [
            ("mlp", "down_proj"),
            ("mlp", "c_proj"),
            ("feed_forward", "w2"),
        ]
    else:
        raise ValueError(f"Unsupported site: {site}")
    for path in candidates:
        module = _get_attr_path(layer_module, path)
        if module is not None:
            return module
    raise ValueError(f"Cannot find module for site={site} at layer={layer}")


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


def capture_layer_site(
    model: AutoModelForCausalLM,
    *,
    layer: int,
    site: str,
    token_index: int | str,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    site_states = capture_layer_site_sequence(model, layer=layer, site=site, inputs=inputs)
    pos = _resolve_token_positions(inputs["attention_mask"], token_index)
    batch_idx = torch.arange(site_states.size(0), device=site_states.device)
    return site_states[batch_idx, pos]


def capture_layer_site_sequence(
    model: AutoModelForCausalLM,
    *,
    layer: int,
    site: str,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    if site == "resid_post":
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states
        hs_idx = layer + 1
        if hs_idx >= len(hidden_states):
            raise ValueError(f"Hidden state index {hs_idx} unavailable for hidden-state len {len(hidden_states)}")
        return hidden_states[hs_idx]

    if site == "resid_pre":
        captured: dict[str, torch.Tensor] = {}
        target_module = get_layer_module(model, layer)

        def _pre_hook(_module, args):
            captured["value"] = args[0]

        handle = target_module.register_forward_pre_hook(_pre_hook)
        try:
            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            handle.remove()

        if "value" not in captured:
            raise RuntimeError(f"Hook did not capture site={site} at layer={layer}")
        return captured["value"]

    captured: dict[str, torch.Tensor] = {}
    target_module = get_layer_site_module(model, layer, site)

    def _hook(_module, _inputs, output):
        captured["value"] = output[0] if isinstance(output, tuple) else output

    handle = target_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if "value" not in captured:
        raise RuntimeError(f"Hook did not capture site={site} at layer={layer}")
    return captured["value"]


def pool_sequence_activations(
    sequence_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "last":
        pos = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(sequence_states.size(0), device=sequence_states.device)
        return sequence_states[batch_idx, pos]

    expanded_mask = attention_mask.unsqueeze(-1).to(sequence_states.dtype)
    if pooling == "mean":
        summed = (sequence_states * expanded_mask).sum(dim=1)
        denom = expanded_mask.sum(dim=1).clamp_min(1)
        return summed / denom
    if pooling == "max":
        neg_inf = torch.full_like(sequence_states, float("-inf"))
        masked = torch.where(expanded_mask.bool(), sequence_states, neg_inf)
        return masked.max(dim=1).values
    raise ValueError(f"Unsupported pooling: {pooling}")


@contextmanager
def layer_overwrite_hook(
    model: AutoModelForCausalLM,
    *,
    layer: int,
    site: str = "resid_post",
    editor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    attention_mask: torch.Tensor,
    token_index: int | str = "last",
):
    positions = _resolve_token_positions(attention_mask, token_index)

    if site == "resid_pre":
        layer_module = get_layer_module(model, layer)

        def _pre_hook(_module, args):
            if not args:
                return args
            hidden = args[0]
            edited = hidden.clone()
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            selected = hidden[batch_idx, positions]
            edited_selected = editor(selected, positions)
            edited[batch_idx, positions] = edited_selected
            return (edited, *args[1:])

        handle = layer_module.register_forward_pre_hook(_pre_hook)
    else:
        layer_module = get_layer_site_module(model, layer, site)

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
