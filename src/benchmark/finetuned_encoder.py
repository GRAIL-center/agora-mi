from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ProxyEncoderBundle:
    model: AutoModelForSequenceClassification
    tokenizer: Any
    device: torch.device
    task_ids: list[str]
    task_proxy_names: list[str]
    model_id: str
    max_length: int
    train_metrics: dict[str, float]


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rows_to_label_matrix(rows: list[dict[str, Any]], task_proxy_names: list[str]) -> np.ndarray:
    labels = np.zeros((len(rows), len(task_proxy_names)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        tags = {str(tag) for tag in row.get("all_tags", [])}
        for task_index, proxy_name in enumerate(task_proxy_names):
            labels[row_index, task_index] = 1.0 if proxy_name in tags else 0.0
    return labels


def _batch_slices(size: int, batch_size: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    order = np.arange(size, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


def _tokenize(tokenizer, texts: list[str], *, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _forward_embeddings(outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = outputs.hidden_states[-1]
    mask = attention_mask.unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled


def fit_finetuned_proxy_encoder(
    *,
    task_registry: dict[str, Any],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    method_cfg: dict[str, Any],
    device_name: str,
    seed: int,
) -> ProxyEncoderBundle:
    task_ids = [str(task["task_id"]) for task in task_registry["coverage_tasks"]]
    task_proxy_names = [str(task["proxy_name"]) for task in task_registry["coverage_tasks"]]
    model_id = str(method_cfg.get("model_id", "distilroberta-base"))
    max_length = int(method_cfg.get("max_length", 512))
    batch_size = int(method_cfg.get("batch_size", 8))
    epochs = int(method_cfg.get("epochs", 3))
    learning_rate = float(method_cfg.get("learning_rate", 2e-5))
    weight_decay = float(method_cfg.get("weight_decay", 0.01))
    grad_clip = float(method_cfg.get("grad_clip", 1.0))
    patience = int(method_cfg.get("early_stopping_patience", 2))

    _set_seed(seed)
    device = _resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(task_ids),
    )
    model.config.problem_type = "multi_label_classification"
    model.to(device)

    train_labels = _rows_to_label_matrix(train_rows, task_proxy_names)
    dev_labels = _rows_to_label_matrix(dev_rows, task_proxy_names) if dev_rows else np.zeros((0, len(task_ids)), dtype=np.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_state = None
    best_dev_loss = float("inf")
    stale_epochs = 0
    last_train_loss = float("nan")
    last_dev_loss = float("nan")

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch_indices in _batch_slices(len(train_rows), batch_size, shuffle=True, seed=seed + epoch):
            batch_rows = [train_rows[int(index)] for index in batch_indices.tolist()]
            batch_texts = [str(row["text"]) for row in batch_rows]
            encoded = _tokenize(tokenizer, batch_texts, max_length=max_length, device=device)
            labels = torch.tensor(train_labels[batch_indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**encoded)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        last_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        if not dev_rows:
            continue

        model.eval()
        dev_losses: list[float] = []
        with torch.no_grad():
            for batch_indices in _batch_slices(len(dev_rows), batch_size, shuffle=False, seed=seed):
                batch_rows = [dev_rows[int(index)] for index in batch_indices.tolist()]
                batch_texts = [str(row["text"]) for row in batch_rows]
                encoded = _tokenize(tokenizer, batch_texts, max_length=max_length, device=device)
                labels = torch.tensor(dev_labels[batch_indices], dtype=torch.float32, device=device)
                outputs = model(**encoded)
                loss = criterion(outputs.logits, labels)
                dev_losses.append(float(loss.item()))
        last_dev_loss = float(np.mean(dev_losses)) if dev_losses else float("nan")

        if math.isnan(last_dev_loss):
            continue
        if last_dev_loss + 1e-6 < best_dev_loss:
            best_dev_loss = last_dev_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return ProxyEncoderBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_ids=task_ids,
        task_proxy_names=task_proxy_names,
        model_id=model_id,
        max_length=max_length,
        train_metrics={
            "train_loss": last_train_loss,
            "dev_loss": best_dev_loss if best_state is not None else last_dev_loss,
            "epochs": float(epochs),
        },
    )


def predict_proxy_scores(
    bundle: ProxyEncoderBundle,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        n_labels = len(bundle.task_ids)
        return np.zeros((0, n_labels), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    model.eval()

    score_batches: list[np.ndarray] = []
    embedding_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_indices in _batch_slices(len(rows), batch_size, shuffle=False, seed=0):
            batch_rows = [rows[int(index)] for index in batch_indices.tolist()]
            batch_texts = [str(row["text"]) for row in batch_rows]
            encoded = _tokenize(tokenizer, batch_texts, max_length=bundle.max_length, device=device)
            outputs = model(**encoded, output_hidden_states=True)
            probs = torch.sigmoid(outputs.logits).detach().cpu().numpy().astype(np.float32)
            pooled = _forward_embeddings(outputs, encoded["attention_mask"]).detach().cpu().numpy().astype(np.float32)
            score_batches.append(probs)
            embedding_batches.append(pooled)

    return np.concatenate(score_batches, axis=0), np.concatenate(embedding_batches, axis=0)
