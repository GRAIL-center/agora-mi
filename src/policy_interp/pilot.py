"""Pilot subset construction for medium scale pipeline validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from policy_interp.io import read_parquet, write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir


def build_pilot_subset(config: ExperimentConfig) -> dict[str, Path]:
    if config.pilot is None or not config.pilot.enabled:
        raise ValueError("Pilot configuration is not enabled for this experiment.")

    source_root = config.dataset.artifacts_path / config.pilot.source_run_name
    source_segments = read_parquet(source_root / config.dataset.prepared_segments_name)
    source_documents = read_parquet(source_root / config.dataset.prepared_documents_name)
    source_matches = read_parquet(source_root / "matching" / "matched_negatives.parquet")

    eligible = source_segments.loc[source_segments["split"].isin(config.pilot.include_splits)].copy()
    split_targets = _allocate_split_targets(
        total=config.pilot.sample_size,
        split_names=config.pilot.include_splits,
        split_weights=_split_weights(config),
    )
    positive_targets = _allocate_split_targets(
        total=config.pilot.positive_target_size,
        split_names=config.pilot.include_splits,
        split_weights=_split_weights(config),
    )
    matched_targets = _allocate_split_targets(
        total=config.pilot.matched_negative_target_size,
        split_names=config.pilot.include_splits,
        split_weights=_split_weights(config),
    )

    positive_selection = _select_positive_segments(eligible, positive_targets, config)
    selected_ids = set(positive_selection["selection_frame"]["segment_id"].tolist())

    negative_selection = _select_matched_negatives(
        eligible=eligible,
        source_matches=source_matches,
        selected_positive_ids=selected_ids,
        matched_targets=matched_targets,
        config=config,
    )
    selected_ids.update(negative_selection["selection_frame"]["segment_id"].tolist())

    fill_selection = _fill_remaining_segments(
        eligible=eligible,
        selected_ids=selected_ids,
        split_targets=split_targets,
        config=config,
    )
    selected_ids.update(fill_selection["selection_frame"]["segment_id"].tolist())

    selected_segments = eligible.loc[eligible["segment_id"].isin(selected_ids)].copy()
    selected_segments = selected_segments.sort_values(["split", "document_id", "Segment position"]).reset_index(drop=True)
    selected_documents = source_documents.loc[
        source_documents["document_id"].isin(selected_segments["document_id"].unique())
    ].copy()
    split_manifest = (
        selected_segments[["document_id", "split"]]
        .drop_duplicates()
        .sort_values("document_id")
        .reset_index(drop=True)
    )

    matching_root = ensure_dir(config.run_root / "matching")
    selected_matches = source_matches.loc[
        source_matches["positive_segment_id"].isin(selected_ids)
        & source_matches["negative_segment_id"].isin(selected_ids)
    ].copy()

    selection_manifest = pd.concat(
        [
            positive_selection["selection_frame"],
            negative_selection["selection_frame"],
            fill_selection["selection_frame"],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["segment_id"], keep="first")
    selection_summary = _build_selection_summary(
        selected_segments=selected_segments,
        positive_frame=positive_selection["selection_frame"],
        negative_frame=negative_selection["selection_frame"],
        fill_frame=fill_selection["selection_frame"],
        split_targets=split_targets,
        config=config,
    )

    run_root = ensure_dir(config.run_root)
    prepared_segments_path = run_root / config.dataset.prepared_segments_name
    prepared_documents_path = run_root / config.dataset.prepared_documents_name
    split_manifest_path = run_root / "split_manifest.parquet"
    matches_path = matching_root / "matched_negatives.parquet"
    selection_manifest_path = run_root / "pilot_selection_manifest.parquet"
    selection_summary_path = run_root / "pilot_selection_summary.parquet"

    write_parquet(selected_segments, prepared_segments_path)
    write_parquet(selected_documents, prepared_documents_path)
    write_parquet(split_manifest, split_manifest_path)
    write_parquet(selected_matches, matches_path)
    write_parquet(selection_manifest, selection_manifest_path)
    write_parquet(selection_summary, selection_summary_path)

    return {
        "prepared_segments": prepared_segments_path,
        "prepared_documents": prepared_documents_path,
        "split_manifest": split_manifest_path,
        "matched_negatives": matches_path,
        "selection_manifest": selection_manifest_path,
        "selection_summary": selection_summary_path,
    }


def _split_weights(config: ExperimentConfig) -> dict[str, float]:
    weights = {
        "train": config.splits.train_ratio,
        "dev": config.splits.dev_ratio,
        "test": config.splits.test_ratio,
    }
    assert config.pilot is not None
    selected_weights = {name: weights.get(name, 0.0) for name in config.pilot.include_splits}
    total = sum(selected_weights.values())
    if total <= 0:
        uniform = 1.0 / max(len(selected_weights), 1)
        return {name: uniform for name in selected_weights}
    return {name: value / total for name, value in selected_weights.items()}


def _allocate_split_targets(
    total: int,
    split_names: list[str],
    split_weights: dict[str, float],
) -> dict[str, int]:
    if total <= 0:
        return {split_name: 0 for split_name in split_names}
    raw = {split_name: total * split_weights.get(split_name, 0.0) for split_name in split_names}
    floors = {split_name: int(np.floor(value)) for split_name, value in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(
        split_names,
        key=lambda split_name: (raw[split_name] - floors[split_name], -split_names.index(split_name)),
        reverse=True,
    )
    for split_name in order[:remainder]:
        floors[split_name] += 1
    return floors


def _select_positive_segments(
    eligible: pd.DataFrame,
    positive_targets: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    assert config.pilot is not None
    positive_mask = np.zeros(len(eligible), dtype=bool)
    for proxy in config.pilot.include_proxies:
        positive_mask |= eligible[proxy].astype(bool).to_numpy()

    positive_candidates = eligible.loc[positive_mask].copy()
    positive_candidates["proxy_count"] = positive_candidates[config.pilot.include_proxies].sum(axis=1)
    positive_candidates["source_type"] = "positive"

    selected_parts: list[pd.DataFrame] = []
    for split_name in config.pilot.include_splits:
        split_frame = positive_candidates.loc[positive_candidates["split"] == split_name].copy()
        if split_frame.empty:
            continue
        split_frame = split_frame.sort_values(
            ["proxy_count", "document_id", "Segment position"],
            ascending=[False, True, True],
        )
        take = min(len(split_frame), positive_targets.get(split_name, 0))
        if take <= 0:
            continue
        split_selected = _balanced_proxy_sample(
            split_frame=split_frame,
            take=take,
            proxy_keys=config.pilot.include_proxies,
        )
        split_selected["selection_stage"] = "positive"
        selected_parts.append(split_selected[["segment_id", "split", "selection_stage", "source_type"]])

    if not selected_parts:
        return {"selection_frame": pd.DataFrame(columns=["segment_id", "split", "selection_stage", "source_type"])}
    return {"selection_frame": pd.concat(selected_parts, ignore_index=True)}


def _balanced_proxy_sample(
    split_frame: pd.DataFrame,
    take: int,
    proxy_keys: list[str],
) -> pd.DataFrame:
    selected_ids: list[str] = []
    remaining = split_frame.copy()
    proxy_index = 0

    while len(selected_ids) < take and not remaining.empty:
        progress = False
        for _ in range(len(proxy_keys)):
            proxy = proxy_keys[proxy_index % len(proxy_keys)]
            proxy_index += 1
            proxy_candidates = remaining.loc[remaining[proxy].astype(bool)].copy()
            if proxy_candidates.empty:
                continue
            chosen = proxy_candidates.iloc[0]
            selected_ids.append(str(chosen["segment_id"]))
            remaining = remaining.loc[remaining["segment_id"] != chosen["segment_id"]].copy()
            progress = True
            if len(selected_ids) >= take:
                break
        if not progress:
            break

    if len(selected_ids) < take and not remaining.empty:
        selected_ids.extend(remaining.head(take - len(selected_ids))["segment_id"].astype(str).tolist())

    return split_frame.loc[split_frame["segment_id"].isin(selected_ids)].copy()


def _select_matched_negatives(
    eligible: pd.DataFrame,
    source_matches: pd.DataFrame,
    selected_positive_ids: set[str],
    matched_targets: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    assert config.pilot is not None
    if not config.pilot.include_matched_negatives or not selected_positive_ids:
        return {"selection_frame": pd.DataFrame(columns=["segment_id", "split", "selection_stage", "source_type"])}

    candidate_matches = source_matches.loc[
        source_matches["proxy"].isin(config.pilot.include_proxies)
        & source_matches["positive_segment_id"].isin(selected_positive_ids)
        & source_matches["split"].isin(config.pilot.include_splits)
    ].copy()
    if candidate_matches.empty:
        return {"selection_frame": pd.DataFrame(columns=["segment_id", "split", "selection_stage", "source_type"])}

    positive_meta = eligible[["segment_id", *config.pilot.include_proxies]].copy()
    positive_meta["positive_proxy_count"] = positive_meta[config.pilot.include_proxies].sum(axis=1)
    candidate_matches = candidate_matches.merge(
        positive_meta[["segment_id", "positive_proxy_count"]],
        how="left",
        left_on="positive_segment_id",
        right_on="segment_id",
    ).drop(columns=["segment_id"])

    negative_meta = eligible[["segment_id", "document_id", "split"]].drop_duplicates()
    candidate_matches = candidate_matches.merge(
        negative_meta,
        how="left",
        left_on="negative_segment_id",
        right_on="segment_id",
        suffixes=("", "_negative"),
    ).drop(columns=["segment_id"])
    candidate_matches = candidate_matches.rename(columns={"split_negative": "negative_split"})
    candidate_matches["negative_split"] = candidate_matches["negative_split"].fillna(candidate_matches["split"])

    selected_negative_ids: set[str] = set()
    selected_parts: list[pd.DataFrame] = []
    for split_name in config.pilot.include_splits:
        split_matches = candidate_matches.loc[candidate_matches["negative_split"] == split_name].copy()
        if split_matches.empty:
            continue
        split_matches = split_matches.sort_values(
            ["positive_proxy_count", "proxy", "negative_segment_id"],
            ascending=[False, True, True],
        )
        chosen_rows: list[dict[str, object]] = []
        for row in split_matches.itertuples(index=False):
            if row.negative_segment_id in selected_negative_ids or row.negative_segment_id in selected_positive_ids:
                continue
            chosen_rows.append(
                {
                    "segment_id": row.negative_segment_id,
                    "split": split_name,
                    "selection_stage": "matched_negative",
                    "source_type": "matched_negative",
                }
            )
            selected_negative_ids.add(str(row.negative_segment_id))
            if len(chosen_rows) >= matched_targets.get(split_name, 0):
                break
        if chosen_rows:
            selected_parts.append(pd.DataFrame(chosen_rows))

    if not selected_parts:
        return {"selection_frame": pd.DataFrame(columns=["segment_id", "split", "selection_stage", "source_type"])}
    return {"selection_frame": pd.concat(selected_parts, ignore_index=True)}


def _fill_remaining_segments(
    eligible: pd.DataFrame,
    selected_ids: set[str],
    split_targets: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    assert config.pilot is not None
    rng = np.random.default_rng(config.pilot.random_fill_seed)
    selected_frame = eligible.loc[eligible["segment_id"].isin(selected_ids), ["segment_id", "split"]].drop_duplicates()
    current_split_counts = selected_frame["split"].value_counts().to_dict()
    candidate_segments = eligible.loc[~eligible["segment_id"].isin(selected_ids)].copy()
    selected_parts: list[pd.DataFrame] = []

    for split_name in config.pilot.include_splits:
        split_frame = candidate_segments.loc[candidate_segments["split"] == split_name].copy()
        if split_frame.empty:
            continue
        need = split_targets.get(split_name, 0) - int(current_split_counts.get(split_name, 0))
        if need <= 0:
            continue
        sample_size = min(need, len(split_frame))
        selected_ids_for_split = rng.choice(
            split_frame["segment_id"].tolist(),
            size=sample_size,
            replace=False,
        ).tolist()
        part = pd.DataFrame(
            {
                "segment_id": selected_ids_for_split,
                "split": split_name,
                "selection_stage": "random_fill",
                "source_type": "random_fill",
            }
        )
        selected_parts.append(part)
        candidate_segments = candidate_segments.loc[~candidate_segments["segment_id"].isin(selected_ids_for_split)].copy()

    selected_so_far = len(selected_ids) + sum(len(part) for part in selected_parts)
    remaining_needed = config.pilot.sample_size - selected_so_far
    if remaining_needed > 0 and not candidate_segments.empty:
        extra_ids = rng.choice(
            candidate_segments["segment_id"].tolist(),
            size=min(remaining_needed, len(candidate_segments)),
            replace=False,
        ).tolist()
        extra_parts = eligible.loc[eligible["segment_id"].isin(extra_ids), ["segment_id", "split"]].copy()
        extra_parts["selection_stage"] = "random_fill"
        extra_parts["source_type"] = "random_fill"
        selected_parts.append(extra_parts)

    if not selected_parts:
        return {"selection_frame": pd.DataFrame(columns=["segment_id", "split", "selection_stage", "source_type"])}
    return {"selection_frame": pd.concat(selected_parts, ignore_index=True)}


def _build_selection_summary(
    selected_segments: pd.DataFrame,
    positive_frame: pd.DataFrame,
    negative_frame: pd.DataFrame,
    fill_frame: pd.DataFrame,
    split_targets: dict[str, int],
    config: ExperimentConfig,
) -> pd.DataFrame:
    assert config.pilot is not None
    rows: list[dict[str, object]] = []
    for split_name in config.pilot.include_splits:
        split_frame = selected_segments.loc[selected_segments["split"] == split_name].copy()
        rows.append(
            {
                "split": split_name,
                "target_count": split_targets.get(split_name, 0),
                "selected_count": len(split_frame),
                "selected_positive_count": int(split_frame[config.pilot.include_proxies].any(axis=1).sum()),
                "privacy_count": int(split_frame["privacy"].sum()) if "privacy" in split_frame.columns else 0,
                "bias_count": int(split_frame["bias"].sum()) if "bias" in split_frame.columns else 0,
                "matched_negative_count": int(negative_frame["split"].eq(split_name).sum()),
                "random_fill_count": int(fill_frame["split"].eq(split_name).sum()),
            }
        )
    return pd.DataFrame(rows)
