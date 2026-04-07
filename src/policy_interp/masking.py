"""Lexical anchor masking and retention metrics."""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from policy_interp.adapters.modeling import HuggingFaceBackboneAdapter, SaeLensAdapter
from policy_interp.extract import run_extraction_for_segments
from policy_interp.feature_matrix import build_module_score_matrix
from policy_interp.io import read_parquet, write_parquet
from policy_interp.schemas import ExperimentConfig


def run_masking_retention(config: ExperimentConfig) -> Path:
    segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    stable_modules = read_parquet(config.run_root / "discovery" / "module_stability.parquet")
    alignment = read_parquet(config.run_root / "discovery" / "module_proxy_alignment.parquet")
    extraction_root = config.run_root / "extraction"
    masking_root = config.run_root / "masking"
    masking_root.mkdir(parents=True, exist_ok=True)
    retention_rows: list[dict[str, object]] = []
    backbone = HuggingFaceBackboneAdapter(config.backbone).load()
    sae_loader = SaeLensAdapter(config.sae)
    sae_cache: dict[int, object] = {}

    try:
        for module in stable_modules.loc[stable_modules["stable"]].itertuples(index=False):
            anchors = list(module.top_ngrams)[: config.masking.anchor_top_k]
            if not anchors:
                continue
            masked_segments = segments.copy()
            masked_segments["text"] = masked_segments["text"].map(lambda text: mask_text(text, anchors, config.masking.mask_token))
            module_root = masking_root / module.stable_module_id
            artifacts = run_extraction_for_segments(
                config=config.model_copy(deep=True),
                prepared_segments=masked_segments,
                extraction_root=module_root / "extraction",
                backbone_bundle=backbone,
                sae_loader=sae_loader,
                sae_cache=sae_cache,
            )

            original_top_features = read_parquet(extraction_root / f"segment_top_features_layer_{module.layer}.parquet")
            masked_top_features = read_parquet(artifacts.layer_top_feature_paths[module.layer])
            original_scores = build_module_score_matrix(stable_modules, original_top_features, segments, int(module.layer))
            masked_scores = build_module_score_matrix(stable_modules, masked_top_features, masked_segments, int(module.layer))

            original_col = original_scores[["segment_id", "split", module.stable_module_id]].rename(columns={module.stable_module_id: "original_score"})
            masked_col = masked_scores[["segment_id", "split", module.stable_module_id]].rename(columns={module.stable_module_id: "masked_score"})
            joined = original_col.merge(masked_col, on=["segment_id", "split"], how="inner")
            test_frame = joined.loc[joined["split"] == "test"].copy()
            mean_original = float(test_frame["original_score"].mean()) if not test_frame.empty else np.nan
            mean_masked = float(test_frame["masked_score"].mean()) if not test_frame.empty else np.nan
            if np.isfinite(mean_original) and mean_original != 0.0:
                retention = mean_masked / mean_original
            else:
                retention = np.nan
            retention_clipped = _clip_ratio(retention)
            retention_exceeds_one = bool(np.isfinite(retention) and retention > 1.0)

            layer_alignment = alignment.loc[alignment["stable_module_id"] == module.stable_module_id].copy()
            for item in layer_alignment.itertuples(index=False):
                proxy_values = segments.set_index("segment_id").loc[test_frame["segment_id"], item.proxy].astype(int).to_numpy()
                if np.unique(proxy_values).size < 2:
                    auc_retention = np.nan
                else:
                    masked_auc = roc_auc_score(proxy_values, test_frame["masked_score"])
                    if np.isfinite(item.test_auc) and item.test_auc != 0.0:
                        auc_retention = masked_auc / item.test_auc
                    else:
                        auc_retention = np.nan
                auc_retention_clipped = _clip_ratio(auc_retention)
                auc_retention_exceeds_one = bool(np.isfinite(auc_retention) and auc_retention > 1.0)
                retention_rows.append(
                    {
                        "stable_module_id": module.stable_module_id,
                        "layer": int(module.layer),
                        "proxy": item.proxy,
                        "anchors": anchors,
                        "module_score_retention_raw": retention,
                        "module_score_retention_clipped": retention_clipped,
                        "module_score_retention_exceeds_one": retention_exceeds_one,
                        "proxy_alignment_auc_retention_raw": auc_retention,
                        "proxy_alignment_auc_retention_clipped": auc_retention_clipped,
                        "proxy_alignment_auc_retention_exceeds_one": auc_retention_exceeds_one,
                        "mean_original_score": mean_original,
                        "mean_masked_score": mean_masked,
                    }
                )
    finally:
        if hasattr(backbone, "model"):
            del backbone.model
        sae_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = masking_root / "masking_retention.parquet"
    write_parquet(pd.DataFrame(retention_rows), output_path)
    return output_path


def mask_text(text: str, anchors: list[str], mask_token: str) -> str:
    masked = text
    for anchor in sorted({anchor for anchor in anchors if anchor}, key=len, reverse=True):
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(anchor).replace('\\ ', r'\\s+')}(?!\w)")
        masked = pattern.sub(mask_token, masked)
    return masked


def _clip_ratio(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0.0, 1.0))
