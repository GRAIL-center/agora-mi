"""Metadata matched negative selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from policy_interp.schemas import ExperimentConfig
from policy_interp.text_models import SentenceEncoder
from policy_interp.utils import normalize_text


@dataclass(slots=True)
class MatchingArtifacts:
    embeddings: pd.DataFrame
    matches: pd.DataFrame


def build_text_embeddings(segments: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    encoder = SentenceEncoder(
        model_name=config.matching.embedding_model,
        device=config.backbone.device,
        max_length=config.matching.max_length,
    )
    texts = [normalize_text(text) for text in segments["text"].tolist()]
    embeddings = encoder.encode(texts, batch_size=config.matching.batch_size)
    frame = pd.DataFrame(
        {
            "segment_id": segments["segment_id"].tolist(),
            "embedding": list(embeddings),
        }
    )
    return frame


def build_matched_negatives(
    segments: pd.DataFrame,
    embeddings: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    proxy_keys = list(config.dataset.proxy_columns.keys())
    weights = config.matching.weights
    embedding_lookup = {
        row.segment_id: row.embedding
        for row in embeddings.itertuples(index=False)
    }
    output_rows: list[dict[str, object]] = []
    app_lookup = {
        row.segment_id: set(row.application_tags)
        for row in segments[["segment_id", "application_tags"]].itertuples(index=False)
    }

    split_frame = segments.loc[segments["split"].isin(["train", "dev", "test"])].copy()
    for proxy_key in proxy_keys:
        for split_name, one_split in split_frame.groupby("split"):
            positives = one_split.loc[one_split[proxy_key]].copy()
            negatives = one_split.loc[~one_split[proxy_key]].copy()
            if positives.empty or negatives.empty:
                continue

            used_negatives: set[str] = set()
            negative_vectors = np.vstack([embedding_lookup[sid] for sid in negatives["segment_id"]])

            for positive in positives.itertuples(index=False):
                positive_vector = embedding_lookup[positive.segment_id].reshape(1, -1)
                text_scores = cosine_similarity(positive_vector, negative_vectors).ravel()
                scored = negatives.copy()
                scored["text_score"] = text_scores
                scored = scored.loc[~scored["segment_id"].isin(used_negatives)].copy()
                if scored.empty:
                    continue
                scored["authority_score"] = (scored["authority"] == positive.authority).astype(float)
                scored["jurisdiction_score"] = (scored["jurisdiction"] == positive.jurisdiction).astype(float)
                scored["form_score"] = (scored["document_form"] == positive.document_form).astype(float)
                scored["domain_score"] = scored["segment_id"].map(
                    lambda sid: jaccard_similarity(app_lookup[positive.segment_id], app_lookup[sid])
                )
                scored["year_score"] = scored["year"].map(
                    lambda year: inverse_distance_score(positive.year, year, scale=25.0)
                )
                scored["length_score"] = scored["segment_length"].map(
                    lambda value: inverse_distance_score(positive.segment_length, value, scale=4000.0)
                )
                scored["match_score"] = (
                    (weights.text * scored["text_score"])
                    + (weights.authority * scored["authority_score"])
                    + (weights.jurisdiction * scored["jurisdiction_score"])
                    + (weights.form * scored["form_score"])
                    + (weights.domain * scored["domain_score"])
                    + (weights.year * scored["year_score"])
                    + (weights.length * scored["length_score"])
                )
                best = scored.sort_values(["match_score", "segment_id"], ascending=[False, True]).iloc[0]
                used_negatives.add(str(best["segment_id"]))
                output_rows.append(
                    {
                        "proxy": proxy_key,
                        "split": split_name,
                        "positive_segment_id": positive.segment_id,
                        "negative_segment_id": best["segment_id"],
                        "match_score": float(best["match_score"]),
                    }
                )
    return pd.DataFrame(output_rows)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def inverse_distance_score(left: float | int | None, right: float | int | None, scale: float) -> float:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return 0.0
    distance = abs(float(left) - float(right))
    return max(0.0, 1.0 - (distance / scale))
