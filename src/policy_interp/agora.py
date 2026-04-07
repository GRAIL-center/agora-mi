"""AGORA dataset preparation and deterministic document grouped splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from policy_interp.constants import APPLICATION_PREFIX
from policy_interp.io import write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir, extract_year, normalize_text, parse_bool, set_seed


@dataclass(slots=True)
class PreparedPaths:
    segments: Path
    documents: Path
    splits: Path


def infer_document_form(row: pd.Series) -> str:
    candidates = [
        row.get("Official pdf source"),
        row.get("Official plaintext source"),
        row.get("Link to document"),
    ]
    joined = " ".join(str(item or "") for item in candidates).lower()
    if ".pdf" in joined:
        return "pdf"
    if ".txt" in joined:
        return "txt"
    if ".doc" in joined:
        return "doc"
    if "html" in joined or "htm" in joined:
        return "html"
    return "other"


def load_and_prepare_agora(config: ExperimentConfig) -> PreparedPaths:
    set_seed(config.splits.seed)
    ensure_dir(config.run_root)
    raw_segments = pd.read_csv(config.dataset.segments_path)
    raw_documents = pd.read_csv(config.dataset.documents_path)
    authorities = pd.read_csv(config.dataset.authorities_path)

    authority_to_parent = {
        str(row["Name"]): (
            "" if pd.isna(row["Parent authority"]) else str(row["Parent authority"])
        )
        for _, row in authorities.iterrows()
    }

    documents = raw_documents.copy()
    documents["document_id"] = documents["AGORA ID"].astype(int)
    documents["authority"] = documents["Authority"].fillna("").astype(str)
    documents["jurisdiction"] = documents["authority"].map(authority_to_parent).fillna("")
    documents["document_form"] = documents.apply(infer_document_form, axis=1)
    documents["year"] = documents["Most recent activity date"].apply(extract_year)
    documents["collection_list"] = documents["Collections"].fillna("").astype(str)
    documents["fulltext_path"] = documents["document_id"].apply(
        lambda value: str((config.dataset.fulltext_path / f"{value}.txt").resolve())
    )
    documents["validated_document"] = documents["Validated?"].apply(parse_bool)

    segments = raw_segments.copy()
    segments["document_id"] = segments["Document ID"].astype(int)
    segments["segment_id"] = (
        segments["document_id"].astype(str)
        + "_"
        + segments["Segment position"].astype(str)
    )
    segments["text"] = segments["Text"].fillna("").astype(str).map(normalize_text)
    segments["segment_annotated"] = segments["Segment annotated"].apply(parse_bool)
    segments["segment_validated"] = segments["Segment validated"].apply(parse_bool)
    segments["not_ai_related"] = segments["Not AI-related"].apply(parse_bool)
    segments["non_operative"] = segments["Non-operative"].apply(parse_bool)
    segments["segment_length"] = segments["text"].str.len()

    keep_mask = (
        segments["segment_annotated"]
        & ~segments["not_ai_related"]
        & ~segments["non_operative"]
    )
    segments = segments.loc[keep_mask].copy()

    application_columns = [
        column for column in segments.columns if str(column).startswith(APPLICATION_PREFIX)
    ]
    segments["application_tags"] = segments[application_columns].apply(
        lambda row: [name for name, value in row.items() if parse_bool(value)],
        axis=1,
    )

    for proxy_key, proxy_column in config.dataset.proxy_columns.items():
        segments[proxy_key] = segments[proxy_column].apply(parse_bool)

    merged = segments.merge(
        documents[
            [
                "document_id",
                "authority",
                "jurisdiction",
                "document_form",
                "year",
                "collection_list",
                "validated_document",
                "fulltext_path",
            ]
        ],
        how="left",
        on="document_id",
        validate="many_to_one",
    )
    merged["validated_document"] = merged["validated_document"].where(
        merged["validated_document"].notna(),
        False,
    ).astype(bool)

    split_manifest = assign_document_grouped_splits(merged, config)
    merged = merged.merge(split_manifest, on="document_id", how="left", validate="many_to_one")

    prepared_segments_path = config.run_root / config.dataset.prepared_segments_name
    prepared_documents_path = config.run_root / config.dataset.prepared_documents_name
    split_manifest_path = config.run_root / "split_manifest.parquet"

    write_parquet(merged, prepared_segments_path)
    write_parquet(documents, prepared_documents_path)
    write_parquet(split_manifest, split_manifest_path)

    return PreparedPaths(
        segments=prepared_segments_path,
        documents=prepared_documents_path,
        splits=split_manifest_path,
    )


def assign_document_grouped_splits(segments: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    proxy_keys = list(config.dataset.proxy_columns.keys())
    document_proxy = (
        segments.groupby("document_id")[proxy_keys]
        .max()
        .reset_index()
    )
    document_proxy["validated_document"] = (
        segments.groupby("document_id")["validated_document"].any().values.astype(bool)
    )
    document_proxy["positive_count"] = document_proxy[proxy_keys].sum(axis=1)
    document_proxy = document_proxy.sort_values(
        by=["positive_count", "document_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    validated_docs = document_proxy.loc[document_proxy["validated_document"], "document_id"].tolist()
    non_validated = document_proxy.loc[~document_proxy["validated_document"]].copy()

    targets = {
        "train": config.splits.train_ratio,
        "dev": config.splits.dev_ratio,
        "test": config.splits.test_ratio,
    }
    split_rows: list[dict[str, object]] = []
    counts = {name: 0 for name in targets}
    label_totals = non_validated[proxy_keys].sum(axis=0).to_numpy(dtype=float)
    current_labels = {name: np.zeros(len(proxy_keys), dtype=float) for name in targets}

    total_docs = len(non_validated)
    raw_target_counts = {
        "train": int(round(total_docs * config.splits.train_ratio)),
        "dev": int(round(total_docs * config.splits.dev_ratio)),
    }
    raw_target_counts["test"] = total_docs - raw_target_counts["train"] - raw_target_counts["dev"]
    target_counts = {
        split_name: max(0, count)
        for split_name, count in raw_target_counts.items()
    }

    for _, row in non_validated.iterrows():
        row_labels = row[proxy_keys].to_numpy(dtype=float)
        best_split = None
        best_score = None
        for split_name, target_ratio in targets.items():
            over_capacity = counts[split_name] >= target_counts[split_name]
            simulated_counts = counts.copy()
            simulated_counts[split_name] += 1
            simulated_labels = {
                name: current_labels[name].copy()
                for name in targets
            }
            simulated_labels[split_name] = simulated_labels[split_name] + row_labels

            size_penalty = sum(
                abs((simulated_counts[name] / max(total_docs, 1)) - targets[name])
                for name in targets
            )
            label_penalty = sum(
                np.abs(simulated_labels[name] - (label_totals * targets[name])).sum()
                for name in targets
            ) / max(label_totals.sum(), 1.0)
            capacity_penalty = 1_000.0 if over_capacity else 0.0
            score = label_penalty + (0.5 * size_penalty) + capacity_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assert best_split is not None
        counts[best_split] += 1
        current_labels[best_split] += row_labels
        split_rows.append({"document_id": int(row["document_id"]), "split": best_split})

    for document_id in validated_docs:
        split_rows.append({"document_id": int(document_id), "split": "validated"})

    return pd.DataFrame(split_rows).sort_values("document_id").reset_index(drop=True)
