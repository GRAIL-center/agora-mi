"""AutoInterp style feature explanation and held out simulation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy_interp.io import read_jsonl, read_parquet, write_jsonl, write_parquet
from policy_interp.schemas import ExperimentConfig
from policy_interp.utils import ensure_dir, normalize_text, set_seed


AUTOINTERP_ROOT = "autointerp"


@dataclass(slots=True)
class AutoInterpArtifacts:
    candidates_path: str
    simulation_path: str
    scores_path: str
    labels_path: str


def run_autointerp(config: ExperimentConfig) -> AutoInterpArtifacts:
    set_seed(config.splits.seed)
    feature_root = config.run_root / "features"
    output_root = ensure_dir(feature_root / AUTOINTERP_ROOT)

    catalog = read_parquet(feature_root / "feature_catalog.parquet")
    segment_scores = read_parquet(feature_root / "feature_catalog_segment_scores.parquet")
    exemplars = read_parquet(feature_root / "feature_catalog_exemplars.parquet")
    contexts = read_parquet(feature_root / "feature_top_contexts.parquet")
    overlay = read_parquet(feature_root / "feature_proxy_overlay.parquet")
    logit_table = read_parquet(feature_root / "feature_catalog_logit_attribution.parquet")
    segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)

    candidates = _build_autointerp_candidates(catalog, config)
    write_jsonl(candidates.to_dict(orient="records"), output_root / "autointerp_feature_candidates.jsonl")

    generator = _load_text_generator(config.autointerp.generation_model, config)
    simulator_model_name = config.autointerp.simulation_model or config.autointerp.generation_model
    if simulator_model_name == config.autointerp.generation_model:
        simulator = generator
        simulator_owned = False
    else:
        simulator = _load_text_generator(simulator_model_name, config)
        simulator_owned = True

    label_rows: list[dict[str, object]] = []
    simulation_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    try:
        for candidate in candidates.itertuples(index=False):
            candidate_contexts = contexts.loc[
                (contexts["layer"] == int(candidate.layer)) & (contexts["feature_id"] == int(candidate.feature_id))
            ].sort_values("rank")
            candidate_segment_scores = segment_scores.loc[
                (segment_scores["layer"] == int(candidate.layer)) & (segment_scores["feature_id"] == int(candidate.feature_id))
            ].copy()
            candidate_exemplars = exemplars.loc[
                (exemplars["layer"] == int(candidate.layer)) & (exemplars["feature_id"] == int(candidate.feature_id))
            ].copy()
            positive_pool = _build_positive_example_pool(
                contexts=candidate_contexts,
                segment_scores=candidate_segment_scores,
                segments=segments,
            )
            required_positive_budget = config.autointerp.num_train_positive + config.autointerp.num_holdout_positive
            if len(positive_pool) < required_positive_budget:
                continue
            negative_pool = _build_negative_example_pool(
                segment_scores=candidate_segment_scores,
                exemplar_rows=candidate_exemplars.loc[candidate_exemplars["example_kind"] == "negative"].copy(),
                segments=segments,
                config=config,
            )
            if len(negative_pool) < config.autointerp.num_train_negative + config.autointerp.num_holdout_negative:
                continue
            positive_train = positive_pool.head(config.autointerp.num_train_positive).copy()
            negative_train = negative_pool.head(config.autointerp.num_train_negative).copy()
            holdout = _build_holdout_examples(
                positive_pool=positive_pool,
                negative_pool=negative_pool,
                train_positive=positive_train,
                train_negative=negative_train,
                config=config,
            )
            if holdout.empty:
                continue

            candidate_overlay = overlay.loc[
                (overlay["layer"] == int(candidate.layer)) & (overlay["feature_id"] == int(candidate.feature_id))
            ].sort_values("test_auc", ascending=False)
            best_proxy = candidate_overlay.iloc[0]["proxy"] if not candidate_overlay.empty else "unknown"
            candidate_logit = logit_table.loc[
                (logit_table["layer"] == int(candidate.layer)) & (logit_table["feature_id"] == int(candidate.feature_id))
            ]
            positive_tokens = candidate_logit.iloc[0]["top_positive_tokens"] if not candidate_logit.empty else []

            interpretation = _generate_feature_interpretation(
                generator=generator,
                candidate=candidate,
                best_proxy=str(best_proxy),
                contexts=candidate_contexts,
                train_positive=positive_train,
                train_negative=negative_train,
                positive_tokens=positive_tokens,
                config=config,
            )

            candidate_simulation_rows = _simulate_feature_activation(
                simulator=simulator,
                candidate=candidate,
                interpretation=interpretation,
                holdout=holdout,
                config=config,
            )
            simulation_rows.extend(candidate_simulation_rows)
            score_row = _score_simulation_rows(candidate, interpretation, candidate_simulation_rows, best_proxy)
            score_rows.append(score_row)
            label_rows.append(
                {
                    "model_id": candidate.model_id,
                    "sae_release": candidate.sae_release,
                    "layer": int(candidate.layer),
                    "feature_id": int(candidate.feature_id),
                    "primary_ranking_family": candidate.primary_ranking_family,
                    "rank": int(candidate.candidate_rank),
                    "ranking_families": list(candidate.ranking_families),
                    "feature_name": interpretation["feature_name"],
                    "activation_hypothesis": interpretation["activation_hypothesis"],
                    "boundary_text": interpretation["boundary_text"],
                    "best_proxy": best_proxy,
                    "faithfulness_score": score_row["faithfulness_score"],
                    "simulation_accuracy": score_row["simulation_accuracy"],
                    "simulation_balanced_accuracy": score_row["simulation_balanced_accuracy"],
                    "positive_precision": score_row["positive_precision"],
                    "positive_recall": score_row["positive_recall"],
                    "mean_confidence": score_row["mean_confidence"],
                    "context_summary": interpretation["context_summary"],
                    "semantic_tag": _infer_autointerp_semantic_tag(
                        interpretation["activation_hypothesis"],
                        interpretation["boundary_text"],
                    ),
                }
            )
    finally:
        _release_generator(generator)
        if simulator_owned:
            _release_generator(simulator)

    candidates_path = output_root / "autointerp_feature_candidates.jsonl"
    simulation_path = output_root / "autointerp_feature_simulation.parquet"
    scores_path = output_root / "autointerp_feature_scores.parquet"
    labels_path = output_root / "autointerp_feature_labels.jsonl"

    write_parquet(pd.DataFrame(simulation_rows), simulation_path)
    write_parquet(pd.DataFrame(score_rows), scores_path)
    write_jsonl(label_rows, labels_path)

    return AutoInterpArtifacts(
        candidates_path=str(candidates_path),
        simulation_path=str(simulation_path),
        scores_path=str(scores_path),
        labels_path=str(labels_path),
    )


def _build_autointerp_candidates(catalog: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    for priority, family in enumerate(config.autointerp.ranking_families):
        family_rows = catalog.loc[catalog["ranking_family"] == family].copy()
        if family_rows.empty:
            continue
        family_rows = (
            family_rows.sort_values(["layer", "rank"])
            .groupby("layer", group_keys=False)
            .head(config.autointerp.top_n_per_family)
            .copy()
        )
        family_rows["family_priority"] = priority
        selected_frames.append(family_rows)

    if not selected_frames:
        return pd.DataFrame()

    selected = pd.concat(selected_frames, ignore_index=True)
    grouped_rows: list[dict[str, object]] = []
    for (_, _), group in selected.groupby(["layer", "feature_id"]):
        ordered = group.sort_values(["family_priority", "rank"]).copy()
        first = ordered.iloc[0]
        grouped_rows.append(
            {
                "model_id": first["model_id"],
                "sae_release": first["sae_release"],
                "layer": int(first["layer"]),
                "feature_id": int(first["feature_id"]),
                "model_depth": int(first["model_depth"]),
                "layer_depth_fraction": float(first["layer_depth_fraction"]),
                "layer_stage": str(first["layer_stage"]),
                "primary_ranking_family": str(first["ranking_family"]),
                "ranking_families": ordered["ranking_family"].astype(str).tolist(),
                "candidate_rank": int(first["rank"]),
                "best_proxy": first.get("best_proxy"),
                "best_proxy_test_auc": float(first.get("best_proxy_test_auc", np.nan)),
                "best_proxy_validated_auc": float(first.get("best_proxy_validated_auc", np.nan)),
                "catalog_key": first["catalog_key"],
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(["layer", "primary_ranking_family", "candidate_rank"]).reset_index(drop=True)


def _build_holdout_examples(
    positive_pool: pd.DataFrame,
    negative_pool: pd.DataFrame,
    train_positive: pd.DataFrame,
    train_negative: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    used_segment_ids = set(train_positive["segment_id"].astype(str).tolist()) | set(train_negative["segment_id"].astype(str).tolist())
    positives = positive_pool.loc[~positive_pool["segment_id"].astype(str).isin(used_segment_ids)].head(
        config.autointerp.num_holdout_positive
    ).copy()
    negatives = negative_pool.loc[~negative_pool["segment_id"].astype(str).isin(used_segment_ids)].head(
        config.autointerp.num_holdout_negative
    ).copy()
    if len(positives) < config.autointerp.num_holdout_positive or len(negatives) < config.autointerp.num_holdout_negative:
        return pd.DataFrame()

    positives["gold_label"] = 1
    negatives["gold_label"] = 0
    holdout = pd.concat([positives, negatives], ignore_index=True)
    holdout["text"] = holdout["text"].fillna("").astype(str).map(normalize_text)
    holdout["pooled_activation"] = holdout["pooled_activation"].astype(float)
    holdout["example_kind"] = holdout["gold_label"].map({1: "positive", 0: "negative"})
    return holdout


def _build_positive_example_pool(
    contexts: pd.DataFrame,
    segment_scores: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    if not contexts.empty:
        frame = contexts.copy()
        frame = frame.merge(
            segments[["segment_id", "split", "text"]],
            on=["segment_id", "split"],
            how="left",
            suffixes=("", "_segment"),
        )
        frame["pooled_activation"] = frame["activation"].astype(float)
        frame["text"] = frame["text"].fillna(frame["context_text"]).astype(str)
        frame = frame.sort_values(["activation", "rank"], ascending=[False, True]).drop_duplicates(["segment_id"])
        return frame[["segment_id", "split", "text", "pooled_activation"]].reset_index(drop=True)

    if segment_scores.empty:
        return pd.DataFrame(columns=["segment_id", "split", "text", "pooled_activation"])
    frame = segment_scores.loc[segment_scores["pooled_activation"] > 0].copy()
    if frame.empty:
        return pd.DataFrame(columns=["segment_id", "split", "text", "pooled_activation"])
    frame = frame.merge(
        segments[["segment_id", "split", "text"]],
        on=["segment_id", "split"],
        how="left",
    )
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame.sort_values("pooled_activation", ascending=False).drop_duplicates(["segment_id"])
    return frame[["segment_id", "split", "text", "pooled_activation"]].reset_index(drop=True)


def _build_negative_example_pool(
    segment_scores: pd.DataFrame,
    exemplar_rows: pd.DataFrame,
    segments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    pools: list[pd.DataFrame] = []
    if not exemplar_rows.empty:
        exemplar_pool = exemplar_rows.rename(columns={"pooled_activation": "pooled_activation"}).copy()
        if "split" not in exemplar_pool.columns:
            exemplar_pool = exemplar_pool.merge(segments[["segment_id", "split"]], on="segment_id", how="left")
        exemplar_pool["text"] = exemplar_pool["text"].fillna("").astype(str)
        pools.append(exemplar_pool[["segment_id", "split", "text", "pooled_activation"]])
    if not segment_scores.empty:
        frame = segment_scores.merge(
            segments[["segment_id", "split", "text"]],
            on=["segment_id", "split"],
            how="left",
        )
        frame["text"] = frame["text"].fillna("").astype(str)
        low_threshold = float(frame["pooled_activation"].quantile(config.autointerp.low_activation_quantile))
        low_frame = frame.loc[frame["pooled_activation"] <= low_threshold].copy()
        if low_frame.empty:
            low_frame = frame.sort_values("pooled_activation", ascending=True).copy()
        pools.append(low_frame[["segment_id", "split", "text", "pooled_activation"]])
    if not pools:
        return pd.DataFrame(columns=["segment_id", "split", "text", "pooled_activation"])
    combined = pd.concat(pools, ignore_index=True)
    combined = combined.sort_values("pooled_activation", ascending=True).drop_duplicates(["segment_id"])
    return combined.reset_index(drop=True)


def _generate_feature_interpretation(
    generator: dict[str, object] | None,
    candidate: object,
    best_proxy: str,
    contexts: pd.DataFrame,
    train_positive: pd.DataFrame,
    train_negative: pd.DataFrame,
    positive_tokens: object,
    config: ExperimentConfig,
) -> dict[str, str]:
    context_examples = [normalize_text(str(item)) for item in contexts["context_text"].head(4).tolist()]
    span_examples = [normalize_text(str(item)) for item in contexts["top_token_span_text"].head(4).tolist()]
    positive_examples = [normalize_text(str(item)) for item in train_positive["text"].head(config.autointerp.num_train_positive).tolist()]
    negative_examples = [normalize_text(str(item)) for item in train_negative["text"].head(config.autointerp.num_train_negative).tolist()]
    token_list = _coerce_text_list(positive_tokens)[:12]
    context_summary = " | ".join(context_examples[:2] + span_examples[:2])

    fallback_name = f"Layer {candidate.layer} feature {candidate.feature_id}"
    fallback_hypothesis = (
        f"This feature activates on recurring policy text patterns associated with {best_proxy} and the evidence snippets shown in the training examples."
    )
    fallback_boundary = "It should not activate strongly on unrelated or contrastive snippets that lack the same obligation or concept pattern."
    if generator is None:
        return {
            "feature_name": fallback_name,
            "activation_hypothesis": fallback_hypothesis,
            "boundary_text": fallback_boundary,
            "context_summary": context_summary,
        }

    prompt = (
        "You are interpreting one sparse feature from a language model reading policy documents.\n"
        "Infer the narrow concept, clause pattern, or semantic trigger that best explains why the feature activates.\n"
        "Be concrete and avoid repeating the instruction.\n"
        "Return only the XML fields below.\n"
        "<name>2 to 6 words</name>\n"
        "<hypothesis>One sentence that states when the feature activates.</hypothesis>\n"
        "<boundary>One sentence that states when the feature should stay low.</boundary>\n\n"
        f"Layer: {candidate.layer}\n"
        f"Ranking families: {list(candidate.ranking_families)}\n"
        f"Best proxy overlay: {best_proxy}\n"
        f"Top contexts: {context_examples}\n"
        f"Top token spans: {span_examples}\n"
        f"Positive examples: {positive_examples}\n"
        f"Negative examples: {negative_examples}\n"
        f"Top logit tokens: {token_list}\n"
    )
    generated = _generate_text(generator, prompt, config.autointerp.max_new_tokens)
    parsed = _parse_interpretation_response(generated)
    if parsed is None:
        return {
            "feature_name": fallback_name,
            "activation_hypothesis": fallback_hypothesis,
            "boundary_text": fallback_boundary,
            "context_summary": context_summary,
        }
    feature_name = parsed["feature_name"]
    activation_hypothesis = parsed["activation_hypothesis"]
    boundary_text = parsed["boundary_text"]
    feature_name = _strip_xml_fragments(feature_name)
    activation_hypothesis = _strip_xml_fragments(activation_hypothesis)
    boundary_text = _strip_xml_fragments(boundary_text)
    if not _looks_reasonable_name(feature_name):
        feature_name = _derive_name_from_hypothesis(activation_hypothesis, fallback_name)
    if not _looks_reasonable_sentence(activation_hypothesis):
        activation_hypothesis = fallback_hypothesis
    if not _looks_reasonable_sentence(boundary_text):
        boundary_text = fallback_boundary
    if normalize_text(boundary_text).lower() == normalize_text(activation_hypothesis).lower():
        boundary_text = fallback_boundary
    return {
        "feature_name": feature_name,
        "activation_hypothesis": activation_hypothesis,
        "boundary_text": boundary_text,
        "context_summary": context_summary,
    }


def _simulate_feature_activation(
    simulator: dict[str, object] | None,
    candidate: object,
    interpretation: dict[str, str],
    holdout: pd.DataFrame,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in holdout.itertuples(index=False):
        prediction = _predict_activation_label(
            simulator=simulator,
            feature_name=interpretation["feature_name"],
            activation_hypothesis=interpretation["activation_hypothesis"],
            boundary_text=interpretation["boundary_text"],
            candidate_text=str(item.text),
            config=config,
        )
        rows.append(
            {
                "model_id": candidate.model_id,
                "sae_release": candidate.sae_release,
                "layer": int(candidate.layer),
                "feature_id": int(candidate.feature_id),
                "primary_ranking_family": candidate.primary_ranking_family,
                "segment_id": str(item.segment_id),
                "split": str(item.split),
                "example_kind": str(item.example_kind),
                "gold_label": int(item.gold_label),
                "gold_activation": float(item.pooled_activation),
                "predicted_label": int(prediction["predicted_label"]),
                "confidence": float(prediction["confidence"]),
                "reason": prediction["reason"],
            }
        )
    return rows


def _score_simulation_rows(
    candidate: object,
    interpretation: dict[str, str],
    simulation_rows: list[dict[str, object]],
    best_proxy: str,
) -> dict[str, object]:
    frame = pd.DataFrame(simulation_rows)
    if frame.empty:
        return {
            "model_id": candidate.model_id,
            "sae_release": candidate.sae_release,
            "layer": int(candidate.layer),
            "feature_id": int(candidate.feature_id),
            "primary_ranking_family": candidate.primary_ranking_family,
            "rank": int(candidate.candidate_rank),
            "ranking_families": list(candidate.ranking_families),
            "best_proxy": best_proxy,
            "feature_name": interpretation["feature_name"],
            "activation_hypothesis": interpretation["activation_hypothesis"],
            "boundary_text": interpretation["boundary_text"],
            "simulation_accuracy": np.nan,
            "simulation_balanced_accuracy": np.nan,
            "positive_precision": np.nan,
            "positive_recall": np.nan,
            "mean_confidence": np.nan,
            "faithfulness_score": np.nan,
            "holdout_count": 0,
        }

    accuracy = float((frame["gold_label"] == frame["predicted_label"]).mean())
    positive_mask = frame["gold_label"] == 1
    negative_mask = frame["gold_label"] == 0
    positive_recall = float(frame.loc[positive_mask, "predicted_label"].mean()) if positive_mask.any() else np.nan
    negative_recall = float((1 - frame.loc[negative_mask, "predicted_label"]).mean()) if negative_mask.any() else np.nan
    if math.isnan(positive_recall) and math.isnan(negative_recall):
        balanced_accuracy = np.nan
    elif math.isnan(positive_recall):
        balanced_accuracy = negative_recall
    elif math.isnan(negative_recall):
        balanced_accuracy = positive_recall
    else:
        balanced_accuracy = 0.5 * (positive_recall + negative_recall)
    predicted_positive = frame["predicted_label"] == 1
    if predicted_positive.any():
        positive_precision = float(frame.loc[predicted_positive, "gold_label"].mean())
    else:
        positive_precision = 0.0
    mean_confidence = float(frame["confidence"].mean())
    faithfulness_score = float(np.nanmean([accuracy, balanced_accuracy]))

    return {
        "model_id": candidate.model_id,
        "sae_release": candidate.sae_release,
        "layer": int(candidate.layer),
        "feature_id": int(candidate.feature_id),
        "primary_ranking_family": candidate.primary_ranking_family,
        "rank": int(candidate.candidate_rank),
        "ranking_families": list(candidate.ranking_families),
        "best_proxy": best_proxy,
        "feature_name": interpretation["feature_name"],
        "activation_hypothesis": interpretation["activation_hypothesis"],
        "boundary_text": interpretation["boundary_text"],
        "simulation_accuracy": accuracy,
        "simulation_balanced_accuracy": balanced_accuracy,
        "positive_precision": positive_precision,
        "positive_recall": positive_recall,
        "mean_confidence": mean_confidence,
        "faithfulness_score": faithfulness_score,
        "holdout_count": int(len(frame)),
    }


def _load_text_generator(model_name: str, config: ExperimentConfig) -> dict[str, object] | None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if config.backbone.dtype == "float16" else torch.float32,
        )
        model.to(config.backbone.device)
        model.eval()
        return {
            "tokenizer": tokenizer,
            "model": model,
            "supports_chat_template": hasattr(tokenizer, "apply_chat_template"),
        }
    except Exception:
        return None


def _generate_text(generator: dict[str, object], prompt: str, max_new_tokens: int) -> str:
    tokenizer = generator["tokenizer"]
    model = generator["model"]
    if generator.get("supports_chat_template"):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(encoded, device=model.device)
        model_inputs = {"input_ids": encoded, "attention_mask": attention_mask}
        prompt_length = int(encoded.shape[-1])
    else:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        model_inputs = dict(encoded)
        prompt_length = int(encoded["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _predict_activation_label(
    simulator: dict[str, object] | None,
    feature_name: str,
    activation_hypothesis: str,
    boundary_text: str,
    candidate_text: str,
    config: ExperimentConfig,
) -> dict[str, object]:
    fallback = {
        "predicted_label": 1 if any(token in candidate_text.lower() for token in _keyword_seeds(activation_hypothesis)) else 0,
        "confidence": 55.0,
        "reason": "Fallback keyword heuristic used because simulator output was unavailable.",
    }
    if simulator is None:
        return fallback

    prompt = (
        "You are validating an interpretation of one sparse feature in policy text.\n"
        "Use the hypothesis and boundary to judge whether the feature should activate strongly.\n"
        "Return only the XML fields below.\n"
        "<label>activates or not_activates</label>\n"
        "<confidence>0 to 100</confidence>\n"
        "<reason>short justification</reason>\n\n"
        f"Feature name: {feature_name}\n"
        f"Activation hypothesis: {activation_hypothesis}\n"
        f"Boundary: {boundary_text}\n"
        f"Candidate text: {candidate_text}\n"
    )
    generated = _generate_text(simulator, prompt, min(96, config.autointerp.max_new_tokens))
    parsed = _parse_simulation_response(generated)
    if parsed is None:
        return fallback
    return {
        "predicted_label": int(parsed["predicted_label"]),
        "confidence": float(np.clip(parsed["confidence"], 0.0, 100.0)),
        "reason": parsed["reason"],
    }


def _normalize_text_lines(text: str) -> list[str]:
    cleaned = re.sub(r"```[A-Za-z0-9_+-]*", " ", text)
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"Line\s*\d+\s*:\s*", "\n", cleaned, flags=re.IGNORECASE)
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = normalize_text(raw_line)
        line = re.sub(r"^(name|hypothesis|boundary|confidence|label|reason)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^[#>*`\s]+", "", line)
        if line:
            lines.append(line)
    return lines


def _extract_first_number(text: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _extract_tagged(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _strip_xml_fragments(match.group(1))


def _parse_interpretation_response(text: str) -> dict[str, str] | None:
    feature_name = _extract_tagged(text, "name")
    activation_hypothesis = _extract_tagged(text, "hypothesis")
    boundary_text = _extract_tagged(text, "boundary")
    if feature_name and activation_hypothesis and boundary_text:
        return {
            "feature_name": feature_name,
            "activation_hypothesis": activation_hypothesis,
            "boundary_text": boundary_text,
        }
    lines = _normalize_text_lines(text)
    if len(lines) < 3:
        return None
    return {
        "feature_name": lines[0],
        "activation_hypothesis": lines[1],
        "boundary_text": lines[2],
    }


def _parse_simulation_response(text: str) -> dict[str, object] | None:
    label_text = _extract_tagged(text, "label")
    confidence_text = _extract_tagged(text, "confidence")
    reason = _extract_tagged(text, "reason")
    if label_text is not None:
        predicted_label = 1 if "activate" in label_text.lower() and "not" not in label_text.lower() else 0
        confidence = _extract_first_number(confidence_text or "")
        return {
            "predicted_label": predicted_label,
            "confidence": 55.0 if confidence is None else confidence,
            "reason": reason or "No reason returned.",
        }
    lines = _normalize_text_lines(text)
    if not lines:
        return None
    label_line = lines[0].lower()
    predicted_label = 1 if "activate" in label_line and "not" not in label_line else 0
    confidence = _extract_first_number(lines[1] if len(lines) > 1 else "")
    return {
        "predicted_label": predicted_label,
        "confidence": 55.0 if confidence is None else confidence,
        "reason": lines[2] if len(lines) > 2 else "No reason returned.",
    }


def _coerce_text_list(value: object) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalize_text(str(item)) for item in value if normalize_text(str(item))]
    item = normalize_text(str(value))
    return [item] if item else []


def _keyword_seeds(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z]{4,}", text.lower())[:8]]


def _looks_reasonable_name(text: str) -> bool:
    lowered = text.lower()
    banned_fragments = [
        "import ",
        "return ",
        "def ",
        "based on the provided",
        "what is the most likely",
        "what causes the feature",
        "feature interpretation",
        "short feature name",
        "activation hypothesis",
        "explanation",
        "2 to 6 words",
        "policy_specific",
        "layer_unique",
        "global_dominance",
    ]
    return 3 <= len(text) <= 96 and "<" not in text and ">" not in text and not any(fragment in lowered for fragment in banned_fragments)


def _looks_reasonable_sentence(text: str) -> bool:
    lowered = text.lower()
    return 24 <= len(text) <= 220 and "<" not in text and ">" not in text and not any(
        fragment in lowered for fragment in ["import ", "return ", "def ", "```"]
    )


def _strip_xml_fragments(text: str) -> str:
    cleaned = re.sub(r"</?[A-Za-z_][A-Za-z0-9_:-]*>", " ", str(text))
    return normalize_text(cleaned)


def _derive_name_from_hypothesis(hypothesis: str, fallback_name: str) -> str:
    words = re.findall(r"[A-Za-z]{4,}", hypothesis)
    stopwords = {
        "this",
        "that",
        "when",
        "then",
        "text",
        "feature",
        "activates",
        "activate",
        "policy",
        "document",
        "documents",
        "should",
        "stay",
        "reading",
        "about",
        "discusses",
        "discuss",
    }
    kept = [word.title() for word in words if word.lower() not in stopwords]
    if not kept:
        return fallback_name
    candidate = " ".join(kept[:4])
    return candidate if _looks_reasonable_name(candidate) else fallback_name


def _infer_autointerp_semantic_tag(activation_hypothesis: str, boundary_text: str) -> str:
    blob = f"{activation_hypothesis} {boundary_text}".lower()
    surface_markers = ["section", "article", "paragraph", "heading", "enumeration", "list", "citation"]
    semantic_markers = ["privacy", "data", "fairness", "bias", "security", "rights", "transparency", "obligation"]
    surface_score = sum(marker in blob for marker in surface_markers)
    semantic_score = sum(marker in blob for marker in semantic_markers)
    return "semantic" if semantic_score >= surface_score else "surface"


def _release_generator(generator: dict[str, object] | None) -> None:
    if generator is None:
        return
    if "model" in generator:
        del generator["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
