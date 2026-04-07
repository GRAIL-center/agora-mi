"""Activation extraction for residual pools and sparse SAE features."""

from __future__ import annotations

import gc
import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from policy_interp.adapters.modeling import HuggingFaceBackboneAdapter, SaeLensAdapter
from policy_interp.io import read_parquet, write_jsonl, write_parquet, write_safetensors
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir, set_seed


@dataclass(slots=True)
class ExtractionArtifacts:
    layer_feature_summary_paths: dict[int, Path]
    layer_top_feature_paths: dict[int, Path]
    layer_context_paths: dict[int, Path]
    layer_residual_manifest_paths: dict[int, Path]


def run_extraction(config: ExperimentConfig) -> ExtractionArtifacts:
    set_seed(config.splits.seed)
    prepared_segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    prepared_segments = prepared_segments.reset_index(drop=True)
    extraction_root = ensure_dir(config.run_root / "extraction")
    return run_extraction_for_segments(config, prepared_segments, extraction_root)


def run_extraction_for_segments(
    config: ExperimentConfig,
    prepared_segments: pd.DataFrame,
    extraction_root: Path,
    backbone_bundle: object | None = None,
    sae_loader: object | None = None,
    sae_cache: dict[int, object] | None = None,
) -> ExtractionArtifacts:
    set_seed(config.splits.seed)
    extraction_root = ensure_dir(extraction_root)

    owns_backbone = backbone_bundle is None
    backbone = backbone_bundle or HuggingFaceBackboneAdapter(config.backbone).load()
    sae_loader = sae_loader or SaeLensAdapter(config.sae)
    if sae_cache is None:
        sae_cache = {}

    summary_paths: dict[int, Path] = {}
    top_feature_paths: dict[int, Path] = {}
    context_paths: dict[int, Path] = {}
    residual_manifest_paths: dict[int, Path] = {}

    for layer in config.extract.layers:
        sae = sae_cache.get(layer)
        if sae is None:
            sae = sae_loader.load_for_layer(layer)
            sae_cache[layer] = sae
        artifacts = _extract_one_layer(
            segments=prepared_segments,
            bundle=backbone,
            sae=sae,
            layer=layer,
            output_root=extraction_root,
            config=config,
        )
        summary_paths[layer] = artifacts["summary"]
        top_feature_paths[layer] = artifacts["top_features"]
        context_paths[layer] = artifacts["contexts"]
        residual_manifest_paths[layer] = artifacts["residual_manifest"]

    if owns_backbone:
        _release_loaded_resources(backbone, sae_cache)

    return ExtractionArtifacts(
        layer_feature_summary_paths=summary_paths,
        layer_top_feature_paths=top_feature_paths,
        layer_context_paths=context_paths,
        layer_residual_manifest_paths=residual_manifest_paths,
    )


def _extract_one_layer(
    segments: pd.DataFrame,
    bundle: object,
    sae: object,
    layer: int,
    output_root: Path,
    config: ExperimentConfig,
) -> dict[str, Path]:
    device = config.backbone.device
    tokenizer = bundle.tokenizer
    model = bundle.model
    top_feature_rows: list[dict[str, object]] = []
    context_heaps: dict[int, list[tuple[float, dict[str, object]]]] = {}
    residual_manifest_rows: list[dict[str, object]] = []
    pooled_batches: list[np.ndarray] = []
    document_ids: list[int] = []
    model_depth = int(bundle.model_depth)
    layer_fraction = _layer_depth_fraction(layer, model_depth)
    layer_stage = _layer_stage(layer_fraction)

    for shard_index, batch_indices in enumerate(_build_token_budget_batches(segments, tokenizer, config)):
        batch = segments.iloc[batch_indices].copy()
        encoded = tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            max_length=config.backbone.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]
            residual_pooled = _pool_sequence(hidden, encoded["attention_mask"], config.extract.residual_pooling_method)
            latents = sae.encode(hidden)
            pooled_latents = _pool_sequence(latents, encoded["attention_mask"], config.extract.pooling_method)

        pooled_cpu = pooled_latents.detach().cpu()
        residual_cpu = residual_pooled.detach().cpu()
        token_ids = encoded["input_ids"].detach().cpu()
        attention = encoded["attention_mask"].detach().cpu()
        latent_cpu = latents.detach().cpu()
        pooled_batches.append(pooled_cpu.to(torch.float32).numpy())
        document_ids.extend(batch["document_id"].astype(int).tolist())

        shard_tensor_path = output_root / f"residual_pool_layer_{layer}_shard_{shard_index:04d}.safetensors"
        write_safetensors({"residual_pooled": residual_cpu}, shard_tensor_path)

        for row_idx, record in enumerate(batch.itertuples(index=False)):
            pooled_row = pooled_cpu[row_idx]

            top_values, top_indices = torch.topk(
                pooled_row,
                k=min(config.extract.segment_top_feature_count, pooled_row.shape[0]),
            )
            tokens_for_row = token_ids[row_idx, attention[row_idx].bool()]
            latent_row = latent_cpu[row_idx, attention[row_idx].bool(), :]

            for feature_rank, (feature_id, pooled_value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
                token_values = latent_row[:, feature_id]
                local_top_k = min(config.extract.segment_top_token_positions_per_feature, token_values.shape[0])
                token_top_values, token_top_indices = torch.topk(token_values, k=local_top_k)
                token_positions = token_top_indices.tolist()
                token_value_list = [float(value) for value in token_top_values.tolist()]
                token_texts = tokenizer.convert_ids_to_tokens(tokens_for_row[token_top_indices].tolist())
                span_start, span_end, span_text = _decode_token_span(
                    tokenizer=tokenizer,
                    tokens=tokens_for_row,
                    token_positions=token_positions,
                    enabled=config.extract.store_token_span_text,
                )
                top_feature_rows.append(
                    {
                        "model_id": bundle.model_id,
                        "sae_release": config.sae.release,
                        "model_depth": model_depth,
                        "layer_depth_fraction": layer_fraction,
                        "layer_stage": layer_stage,
                        "segment_id": record.segment_id,
                        "document_id": int(record.document_id),
                        "split": record.split,
                        "layer": layer,
                        "feature_rank": feature_rank,
                        "feature_id": int(feature_id),
                        "pooled_activation": float(pooled_value),
                        "token_positions": token_positions,
                        "token_values": token_value_list,
                        "token_texts": token_texts,
                        "top_token_span_start": span_start,
                        "top_token_span_end": span_end,
                        "top_token_span_text": span_text,
                    }
                )
                _update_context_heap(
                    context_heaps=context_heaps,
                    model_id=bundle.model_id,
                    sae_release=config.sae.release,
                    model_depth=model_depth,
                    layer_fraction=layer_fraction,
                    layer_stage=layer_stage,
                    feature_id=int(feature_id),
                    pooled_value=float(pooled_value),
                    record=record,
                    token_positions=token_positions,
                    token_texts=token_texts,
                    tokenizer=tokenizer,
                    tokens_for_row=tokens_for_row,
                    window=config.extract.context_window,
                    top_k=config.extract.top_contexts_per_feature,
                )

            residual_manifest_rows.append(
                    {
                        "model_id": bundle.model_id,
                        "sae_release": config.sae.release,
                        "model_depth": model_depth,
                        "layer_depth_fraction": layer_fraction,
                        "layer_stage": layer_stage,
                        "segment_id": record.segment_id,
                        "document_id": int(record.document_id),
                        "split": record.split,
                        "layer": layer,
                    "shard_index": shard_index,
                    "row_index": row_idx,
                    "tensor_path": str(shard_tensor_path.resolve()),
                }
            )

        del outputs, hidden, residual_pooled, latents, pooled_latents
        del pooled_cpu, residual_cpu, token_ids, attention, latent_cpu, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_matrix = np.concatenate(pooled_batches, axis=0) if pooled_batches else np.zeros((0, sae.cfg.d_sae), dtype=np.float32)
    feature_summary = _build_feature_summary_frame(
        pooled_matrix=pooled_matrix,
        document_ids=document_ids,
        top_feature_rows=top_feature_rows,
        model_id=bundle.model_id,
        sae_release=config.sae.release,
        model_depth=model_depth,
        layer=layer,
        layer_fraction=layer_fraction,
        layer_stage=layer_stage,
    )
    top_feature_frame = pd.DataFrame(top_feature_rows)
    context_records = _heaps_to_records(context_heaps)
    residual_manifest = pd.DataFrame(residual_manifest_rows)

    summary_path = output_root / f"feature_summary_layer_{layer}.parquet"
    top_features_path = output_root / f"segment_top_features_layer_{layer}.parquet"
    contexts_path = output_root / f"feature_top_contexts_layer_{layer}.jsonl"
    residual_manifest_path = output_root / f"residual_pool_manifest_layer_{layer}.parquet"

    write_parquet(feature_summary, summary_path)
    write_parquet(top_feature_frame, top_features_path)
    write_jsonl(context_records, contexts_path)
    write_parquet(residual_manifest, residual_manifest_path)

    return {
        "summary": summary_path,
        "top_features": top_features_path,
        "contexts": contexts_path,
        "residual_manifest": residual_manifest_path,
    }


def _build_token_budget_batches(
    segments: pd.DataFrame,
    tokenizer: object,
    config: ExperimentConfig,
) -> list[list[int]]:
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_token_count = 0
    for index, text in enumerate(segments["text"].tolist()):
        token_count = len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=config.backbone.max_length,
            )
        )
        would_exceed = (
            current_batch
            and (
                (len(current_batch) + 1 > config.extract.max_segments_per_shard)
                or (current_token_count + token_count > config.extract.max_tokens_per_shard)
            )
        )
        if would_exceed:
            batches.append(current_batch)
            current_batch = []
            current_token_count = 0
        current_batch.append(index)
        current_token_count += token_count
    if current_batch:
        batches.append(current_batch)
    return batches


def _pool_sequence(values: torch.Tensor, attention_mask: torch.Tensor, pooling_method: str) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(values.dtype)
    masked = values * mask
    if pooling_method == "mean":
        denom = mask.sum(dim=1).clamp(min=1)
        return masked.sum(dim=1) / denom
    large_negative = torch.finfo(values.dtype).min
    masked_for_max = values.masked_fill(~attention_mask.bool().unsqueeze(-1), large_negative)
    pooled = masked_for_max.max(dim=1).values
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled


def _update_context_heap(
    context_heaps: dict[int, list[tuple[float, dict[str, object]]]],
    model_id: str,
    sae_release: str,
    model_depth: int,
    layer_fraction: float,
    layer_stage: str,
    feature_id: int,
    pooled_value: float,
    record: object,
    token_positions: list[int],
    token_texts: list[str],
    tokenizer: object,
    tokens_for_row: torch.Tensor,
    window: int,
    top_k: int,
) -> None:
    center = token_positions[0] if token_positions else 0
    start = max(0, center - window)
    end = min(tokens_for_row.shape[0], center + window + 1)
    context_text = tokenizer.decode(tokens_for_row[start:end].tolist(), skip_special_tokens=True)
    span_start, span_end, span_text = _decode_token_span(tokenizer, tokens_for_row, token_positions, enabled=True)
    left_context = tokenizer.decode(tokens_for_row[start:center].tolist(), skip_special_tokens=True) if center > start else ""
    right_context = tokenizer.decode(tokens_for_row[center + 1 : end].tolist(), skip_special_tokens=True) if center + 1 < end else ""
    payload = {
        "model_id": model_id,
        "sae_release": sae_release,
        "model_depth": model_depth,
        "layer_depth_fraction": layer_fraction,
        "layer_stage": layer_stage,
        "feature_id": feature_id,
        "segment_id": record.segment_id,
        "document_id": int(record.document_id),
        "split": record.split,
        "activation": pooled_value,
        "context_text": context_text,
        "token_positions": token_positions,
        "token_texts": token_texts,
        "top_token_span_start": span_start,
        "top_token_span_end": span_end,
        "top_token_span_text": span_text,
        "left_context": left_context,
        "right_context": right_context,
    }
    heap = context_heaps.setdefault(feature_id, [])
    heapq.heappush(heap, (pooled_value, str(record.segment_id), payload))
    if len(heap) > top_k:
        heapq.heappop(heap)


def _heaps_to_records(context_heaps: dict[int, list[tuple[float, dict[str, object]]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for feature_id, heap in context_heaps.items():
        ordered = [payload for _, _, payload in sorted(heap, key=lambda item: item[0], reverse=True)]
        for rank, payload in enumerate(ordered, start=1):
            item = payload.copy()
            item["rank"] = rank
            item["feature_id"] = feature_id
            output.append(item)
    return output


def _build_feature_summary_frame(
    pooled_matrix: np.ndarray,
    document_ids: list[int],
    top_feature_rows: list[dict[str, object]],
    model_id: str,
    sae_release: str,
    model_depth: int,
    layer: int,
    layer_fraction: float,
    layer_stage: str,
) -> pd.DataFrame:
    if pooled_matrix.size == 0:
        return pd.DataFrame(
            columns=[
                "model_id",
                "sae_release",
                "model_depth",
                "layer_depth_fraction",
                "layer_stage",
                "layer",
                "feature_id",
                "activation_count",
                "activation_frequency",
                "mean_all_activation",
                "mean_magnitude",
                "mean_positive_activation",
                "max_activation",
                "top20_mean_activation",
                "top100_mean_activation",
                "activation_gini",
                "document_frequency",
                "mean_token_peak",
                "global_dominance_score",
                "policy_specific_score",
            ]
        )

    positive_mask = pooled_matrix > 0
    activation_count = positive_mask.sum(axis=0).astype(np.int64)
    activation_frequency = activation_count / max(pooled_matrix.shape[0], 1)
    mean_all_activation = pooled_matrix.mean(axis=0)
    positive_sums = np.where(positive_mask, pooled_matrix, 0.0).sum(axis=0)
    mean_positive_activation = np.divide(
        positive_sums,
        np.maximum(activation_count, 1),
        out=np.zeros_like(positive_sums, dtype=np.float64),
        where=activation_count > 0,
    )
    max_activation = pooled_matrix.max(axis=0)
    top20_mean_activation = _compute_tail_means(pooled_matrix, top_k=20)
    top100_mean_activation = _compute_tail_means(pooled_matrix, top_k=100)
    activation_gini = _compute_activation_gini(pooled_matrix)
    document_frequency = _compute_document_frequency(pooled_matrix, document_ids)

    top_feature_frame = pd.DataFrame(top_feature_rows)
    if top_feature_frame.empty:
        mean_token_peak = np.zeros(pooled_matrix.shape[1], dtype=np.float64)
    else:
        peak_stats = (
            top_feature_frame.assign(
                peak_token_value=top_feature_frame["token_values"].apply(
                    lambda values: float(values[0]) if values else 0.0
                )
            )
            .groupby("feature_id")["peak_token_value"]
            .mean()
        )
        mean_token_peak = np.zeros(pooled_matrix.shape[1], dtype=np.float64)
        mean_token_peak[peak_stats.index.to_numpy(dtype=int)] = peak_stats.to_numpy(dtype=np.float64)

    global_dominance_score = activation_frequency * mean_positive_activation
    policy_specific_score = top20_mean_activation * (1.0 - activation_frequency)

    return pd.DataFrame(
        {
            "model_id": model_id,
            "sae_release": sae_release,
            "model_depth": model_depth,
            "layer_depth_fraction": layer_fraction,
            "layer_stage": layer_stage,
            "layer": layer,
            "feature_id": np.arange(pooled_matrix.shape[1], dtype=np.int64),
            "activation_count": activation_count,
            "activation_frequency": activation_frequency,
            "mean_all_activation": mean_all_activation,
            "mean_magnitude": mean_all_activation,
            "mean_positive_activation": mean_positive_activation,
            "max_activation": max_activation,
            "top20_mean_activation": top20_mean_activation,
            "top100_mean_activation": top100_mean_activation,
            "activation_gini": activation_gini,
            "document_frequency": document_frequency,
            "mean_token_peak": mean_token_peak,
            "global_dominance_score": global_dominance_score,
            "policy_specific_score": policy_specific_score,
        }
    )


def _compute_tail_means(matrix: np.ndarray, top_k: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    row_count = matrix.shape[0]
    tail_k = min(top_k, row_count)
    output = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[1], 512):
        end = min(matrix.shape[1], start + 512)
        chunk = matrix[:, start:end]
        partitioned = np.partition(chunk, kth=row_count - tail_k, axis=0)
        output[start:end] = partitioned[-tail_k:, :].mean(axis=0)
    return output


def _compute_activation_gini(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    row_count = matrix.shape[0]
    if row_count <= 1:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    output = np.zeros(matrix.shape[1], dtype=np.float64)
    index = np.arange(1, row_count + 1, dtype=np.float64).reshape(-1, 1)
    for start in range(0, matrix.shape[1], 256):
        end = min(matrix.shape[1], start + 256)
        chunk = np.sort(matrix[:, start:end], axis=0)
        chunk_sum = chunk.sum(axis=0, dtype=np.float64)
        numerator = ((2.0 * index - row_count - 1.0) * chunk).sum(axis=0, dtype=np.float64)
        output[start:end] = np.divide(
            numerator,
            row_count * np.maximum(chunk_sum, 1e-12),
            out=np.zeros(end - start, dtype=np.float64),
            where=chunk_sum > 0,
        )
    return np.clip(output, 0.0, 1.0)


def _compute_document_frequency(matrix: np.ndarray, document_ids: list[int]) -> np.ndarray:
    if matrix.size == 0 or not document_ids:
        return np.zeros((matrix.shape[1] if matrix.ndim == 2 else 0,), dtype=np.int64)
    document_ids_array = np.asarray(document_ids, dtype=np.int64)
    unique_docs, inverse = np.unique(document_ids_array, return_inverse=True)
    output = np.zeros(matrix.shape[1], dtype=np.int64)
    for doc_index in range(unique_docs.shape[0]):
        doc_rows = matrix[inverse == doc_index]
        if doc_rows.size == 0:
            continue
        output += (doc_rows.max(axis=0) > 0).astype(np.int64)
    return output


def _decode_token_span(
    tokenizer: object,
    tokens: torch.Tensor,
    token_positions: list[int],
    enabled: bool,
) -> tuple[int | None, int | None, str]:
    if not enabled or not token_positions:
        return None, None, ""
    span_start = int(min(token_positions))
    span_end = int(max(token_positions))
    span_text = tokenizer.decode(tokens[span_start : span_end + 1].tolist(), skip_special_tokens=True)
    return span_start, span_end, span_text


def _layer_depth_fraction(layer: int, model_depth: int) -> float:
    if model_depth <= 1:
        return 0.0
    return float(layer / max(model_depth - 1, 1))


def _layer_stage(layer_fraction: float) -> str:
    if layer_fraction < 0.33:
        return "early"
    if layer_fraction < 0.66:
        return "mid"
    return "late"


def _release_loaded_resources(backbone: object | None, sae_cache: dict[int, object] | None) -> None:
    if backbone is not None and hasattr(backbone, "model"):
        del backbone.model
    if sae_cache:
        for layer in list(sae_cache.keys()):
            del sae_cache[layer]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
