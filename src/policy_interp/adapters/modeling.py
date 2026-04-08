"""Model and SAE adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy_interp.schemas import BackboneConfig, SaeConfig


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


@dataclass(slots=True)
class BackboneBundle:
    tokenizer: object
    model: object
    device: str
    model_id: str
    model_depth: int


class HuggingFaceBackboneAdapter:
    def __init__(self, config: BackboneConfig):
        self.config = config

    def load(self) -> BackboneBundle:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=resolve_torch_dtype(self.config.dtype),
            trust_remote_code=self.config.trust_remote_code,
        )
        model.to(self.config.device)
        model.eval()
        return BackboneBundle(
            tokenizer=tokenizer,
            model=model,
            device=self.config.device,
            model_id=self.config.model_name,
            model_depth=resolve_model_depth(model),
        )


class SaeLensAdapter:
    def __init__(self, config: SaeConfig):
        self.config = config

    def load_for_layer(self, layer: int) -> SAE:
        sae_id = self._sae_id_for_layer(layer)
        sae = SAE.from_pretrained(
            release=self.config.release,
            sae_id=sae_id,
            device=self.config.device,
        )
        sae.eval()
        return sae

    def _sae_id_for_layer(self, layer: int) -> str:
        if "layer_" in self.config.sae_id:
            return self.config.sae_id.replace("layer_24", f"layer_{layer}")
        return self.config.sae_id


def resolve_transformer_layer(model: object, layer_index: int) -> object:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_index]
    if hasattr(model, "layers"):
        return model.layers[layer_index]
    raise AttributeError("Unable to resolve transformer layers for the configured model.")


def resolve_model_depth(model: object) -> int:
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return int(len(model.model.layers))
    if hasattr(model, "layers"):
        return int(len(model.layers))
    raise AttributeError("Unable to resolve model depth for the configured model.")
