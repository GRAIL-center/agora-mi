from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower()
    if d == "bfloat16":
        return torch.bfloat16
    if d == "float16":
        return torch.float16
    if d == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model_bundle(config: dict[str, Any]) -> ModelBundle:
    model_id = str(config["model_id"])
    device = resolve_device(str(config.get("device", "auto")))
    dtype = resolve_dtype(str(config.get("dtype", "bfloat16")))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)
