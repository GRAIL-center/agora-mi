"""Helpers to build dense analysis matrices from sparse artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from policy_interp.io import read_parquet


def build_module_score_matrix(
    stable_modules: pd.DataFrame,
    top_features: pd.DataFrame,
    segments: pd.DataFrame,
    layer: int,
) -> pd.DataFrame:
    layer_modules = stable_modules.loc[(stable_modules["layer"] == layer) & (stable_modules["stable"])].copy()
    if layer_modules.empty:
        return segments[["segment_id", "split"]].drop_duplicates().copy()

    grouped = top_features.groupby(["segment_id", "feature_id"])["pooled_activation"].max().reset_index()
    base = segments[["segment_id", "split"]].drop_duplicates().copy()
    for module in layer_modules.itertuples(index=False):
        module_features = set(module.feature_ids)
        scores = (
            grouped.loc[grouped["feature_id"].isin(module_features)]
            .groupby("segment_id")["pooled_activation"]
            .mean()
            .rename(module.stable_module_id)
            .reset_index()
        )
        base = base.merge(scores, on="segment_id", how="left")
    score_columns = [column for column in base.columns if column not in {"segment_id", "split"}]
    base[score_columns] = base[score_columns].fillna(0.0)
    return base


def load_residual_matrix(manifest_path: str | Path) -> pd.DataFrame:
    manifest = read_parquet(manifest_path)
    rows: list[dict[str, object]] = []
    cache: dict[str, np.ndarray] = {}
    for item in manifest.itertuples(index=False):
        tensor_path = str(item.tensor_path)
        if tensor_path not in cache:
            cache[tensor_path] = load_file(tensor_path)["residual_pooled"].to(torch.float32).cpu().numpy()
        values = cache[tensor_path][int(item.row_index)]
        rows.append(
            {
                "segment_id": item.segment_id,
                "split": item.split,
                "layer": int(item.layer),
                "vector": values.astype(np.float32),
            }
        )
    return pd.DataFrame(rows)


def matrix_from_vector_frame(frame: pd.DataFrame, key_column: str = "segment_id") -> tuple[list[str], np.ndarray]:
    ordered = frame.sort_values(key_column).reset_index(drop=True)
    matrix = np.vstack(ordered["vector"].tolist()) if not ordered.empty else np.zeros((0, 1), dtype=np.float32)
    return ordered[key_column].tolist(), matrix
