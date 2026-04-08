"""Module labeling and natural language summary generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy_interp.adapters.modeling import resolve_torch_dtype
from policy_interp.feature_matrix import build_module_score_matrix
from policy_interp.io import read_jsonl, read_parquet, write_jsonl
from policy_interp.schemas import ExperimentConfig


@dataclass(slots=True)
class LabelArtifacts:
    labels_path: str


def run_labeling(config: ExperimentConfig) -> LabelArtifacts:
    stable_modules = read_parquet(config.run_root / "discovery" / "module_stability.parquet")
    alignment = read_parquet(config.run_root / "discovery" / "module_proxy_alignment.parquet")
    segments = read_parquet(config.run_root / config.dataset.prepared_segments_name)
    labels: list[dict[str, object]] = []
    generator = _maybe_load_generator(config)

    for layer in config.extract.layers:
        top_features = read_parquet(config.run_root / "extraction" / f"segment_top_features_layer_{layer}.parquet")
        contexts = read_jsonl(config.run_root / "extraction" / f"feature_top_contexts_layer_{layer}.jsonl")
        score_frame = build_module_score_matrix(stable_modules, top_features, segments, layer)
        for module in stable_modules.loc[(stable_modules["layer"] == layer) & (stable_modules["stable"])].itertuples(index=False):
            module_alignment = alignment.loc[alignment["stable_module_id"] == module.stable_module_id].copy()
            top_proxy_row = module_alignment.sort_values("test_auc", ascending=False).head(1)
            top_proxy = top_proxy_row["proxy"].iloc[0] if not top_proxy_row.empty else "unknown"
            module_contexts = [item for item in contexts if int(item["feature_id"]) in set(module.feature_ids)]
            exemplars = _top_exemplars(score_frame, module.stable_module_id, segments, config.labeling.num_exemplars)
            negatives = _contrastive_negatives(score_frame, module.stable_module_id, segments, exemplars, config.labeling.num_negative_controls)
            template_summary = _template_summary(module, top_proxy, module_contexts, exemplars, negatives)
            generated = _generate_name_and_rationale(generator, config, module, top_proxy, module_contexts, exemplars, negatives)
            labels.append(
                {
                    "stable_module_id": module.stable_module_id,
                    "layer": int(module.layer),
                    "top_proxy": top_proxy,
                    "template_summary": template_summary,
                    "generated_name": generated["name"],
                    "generated_rationale": generated["rationale"],
                    "exemplars": exemplars,
                    "contrastive_negatives": negatives,
                }
            )

    labels_path = config.run_root / "paper_exports" / "module_labels.jsonl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(labels, labels_path)
    return LabelArtifacts(str(labels_path))


def _maybe_load_generator(config: ExperimentConfig) -> dict[str, object] | None:
    if not config.labeling.llm_hook_enabled:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.labeling.generator_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.labeling.generator_model,
            dtype=resolve_torch_dtype(config.backbone.dtype),
        )
        model.to(config.backbone.device)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}
    except Exception:
        return None


def _top_exemplars(
    score_frame: pd.DataFrame,
    module_id: str,
    segments: pd.DataFrame,
    count: int,
) -> list[dict[str, object]]:
    joined = score_frame[["segment_id", module_id]].merge(
        segments[["segment_id", "text", "authority", "jurisdiction", "document_form", "collection_list"]],
        on="segment_id",
        how="left",
    )
    top = joined.sort_values(module_id, ascending=False).head(count)
    return top.to_dict(orient="records")


def _contrastive_negatives(
    score_frame: pd.DataFrame,
    module_id: str,
    segments: pd.DataFrame,
    exemplars: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    if not exemplars:
        return []
    joined = score_frame[["segment_id", module_id]].merge(
        segments[["segment_id", "text", "authority", "jurisdiction", "document_form", "collection_list"]],
        on="segment_id",
        how="left",
    )
    rows: list[dict[str, object]] = []
    for exemplar in exemplars:
        candidate = joined.loc[
            (joined["authority"] == exemplar["authority"])
            & (joined["jurisdiction"] == exemplar["jurisdiction"])
            & (joined["document_form"] == exemplar["document_form"])
            & (joined["collection_list"] == exemplar["collection_list"])
        ].sort_values(module_id, ascending=True)
        if candidate.empty:
            candidate = joined.loc[joined["collection_list"] == exemplar["collection_list"]].sort_values(module_id, ascending=True)
        if candidate.empty:
            continue
        rows.append(candidate.iloc[0].to_dict())
        if len(rows) >= count:
            break
    return rows


def _template_summary(
    module: object,
    top_proxy: str,
    contexts: list[dict[str, object]],
    exemplars: list[dict[str, object]],
    negatives: list[dict[str, object]],
) -> str:
    context_snippets = [re.sub(r"\s+", " ", str(item["context_text"])).strip() for item in contexts[:3]]
    exemplar_text = [re.sub(r"\s+", " ", str(item["text"]))[:160] for item in exemplars[:2]]
    negative_text = [re.sub(r"\s+", " ", str(item["text"]))[:120] for item in negatives[:2]]
    return (
        f"Layer {module.layer} module with {module.module_size} features. "
        f"Top proxy alignment is {top_proxy}. "
        f"Representative contexts: {' | '.join(context_snippets)}. "
        f"High activation examples: {' | '.join(exemplar_text)}. "
        f"Contrastive low activation examples: {' | '.join(negative_text)}."
    )


def _generate_name_and_rationale(
    generator: dict[str, object] | None,
    config: ExperimentConfig,
    module: object,
    top_proxy: str,
    contexts: list[dict[str, object]],
    exemplars: list[dict[str, object]],
    negatives: list[dict[str, object]],
) -> dict[str, str]:
    fallback_name = f"Layer {module.layer} {top_proxy} related module"
    fallback_rationale = (
        f"This module aligns most strongly with {top_proxy} and is supported by top activating contexts and contrastive negatives."
    )
    if generator is None:
        return {"name": fallback_name, "rationale": fallback_rationale}

    prompt = (
        "You are naming an interpretable sparse module for policy text analysis.\n"
        "Return two lines only.\n"
        "Line 1: short name.\n"
        "Line 2: two sentence rationale.\n\n"
        f"Top proxy: {top_proxy}\n"
        f"Contexts: {[item['context_text'] for item in contexts[:config.labeling.num_top_contexts]]}\n"
        f"Positive exemplars: {[item['text'][:180] for item in exemplars[:config.labeling.num_exemplars]]}\n"
        f"Contrastive negatives: {[item['text'][:180] for item in negatives[:config.labeling.num_negative_controls]]}\n"
    )
    tokenizer = generator["tokenizer"]
    model = generator["model"]
    encoded = tokenizer(prompt, return_tensors="pt").to(config.backbone.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=config.labeling.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = decoded[len(prompt) :].strip()
    lines = _normalize_generation_lines(generated)
    if len(lines) < 2:
        return {"name": fallback_name, "rationale": fallback_rationale}
    name = lines[0]
    rationale = " ".join(lines[1:])
    rationale = _normalize_sentence_text(rationale)
    if not _looks_like_label_name(name) or not _looks_like_rationale(rationale):
        return {"name": fallback_name, "rationale": fallback_rationale}
    return {"name": name, "rationale": rationale}


def _normalize_generation_lines(text: str) -> list[str]:
    cleaned = re.sub(r"```[A-Za-z0-9_+-]*", " ", text)
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"Line\s*1\s*:\s*", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Line\s*2\s*:\s*", "\n", cleaned, flags=re.IGNORECASE)
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(line\s*\d+\s*:\s*)", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(name|rationale)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^[#>*`\-\s]+", "", line)
        line = re.sub(r"[*_`]+", "", line).strip()
        line = _normalize_sentence_text(line)
        if line:
            lines.append(line)
    return lines


def _normalize_sentence_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"' ")


def _looks_like_label_name(text: str) -> bool:
    if len(text) < 3 or len(text) > 80:
        return False
    lowered = text.lower()
    banned_fragments = [
        "python",
        "tool_code",
        "```",
        "return two lines",
        "positive exemplars",
        "import ",
        "def ",
        "print(",
        "return ",
        "shortname",
        "line 1",
        "line 2",
    ]
    if any(fragment in lowered for fragment in banned_fragments):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 ]{2,79}", text))


def _looks_like_rationale(text: str) -> bool:
    if len(text) < 20:
        return False
    lowered = text.lower()
    banned_fragments = ["import ", "def ", "print(", "return ", "example usage", "```"]
    return not any(fragment in lowered for fragment in banned_fragments)
