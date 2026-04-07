"""Text encoders used for matching and sentence embedding baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(slots=True)
class SentenceEncoder:
    model_name: str
    device: str = "cuda"
    max_length: int = 512
    tokenizer: object = field(init=False, repr=False)
    model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            attention = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(min=1)
            all_embeddings.append(pooled.detach().cpu().numpy())
        if not all_embeddings:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(all_embeddings).astype(np.float32)
