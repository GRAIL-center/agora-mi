"""Activation Oracle helpers for supplementary explanation and scaffold evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy_interp.adapters.modeling import resolve_torch_dtype
from policy_interp.schemas import ExperimentConfig

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


AO_REAL_CONDITION = "ao_real"
AO_TEXT_ONLY_CONDITION = "ao_text_only"
AO_SHUFFLED_CONDITION = "ao_shuffled_activation"
AO_READY_STATUS = "ready"
AO_INSUFFICIENT_BUNDLE_STATUS = "insufficient_bundle_signal"

AO_OBLIGATION_FAMILIES = (
    "privacy",
    "bias",
    "discrimination",
    "transparency",
    "rights_violation",
    "interpretability",
    "governance_other",
)
AO_SPECIFICITY_LABELS = ("policy_specific", "generic_legalese")
AO_SCAFFOLD_FRAMES = (
    "raw_excerpt",
    "analyst_question",
    "compliance_memo",
    "neutral_restatement",
    "adversarial_bland",
)

_COMMON_STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "agency",
    "article",
    "because",
    "before",
    "being",
    "between",
    "compliance",
    "content",
    "document",
    "documents",
    "excerpt",
    "feature",
    "features",
    "first",
    "governance",
    "legal",
    "obligation",
    "obligations",
    "policy",
    "regulation",
    "regulatory",
    "rights",
    "shall",
    "should",
    "signal",
    "text",
    "this",
    "those",
    "these",
    "under",
    "where",
    "which",
}

_OBLIGATION_KEYWORDS = {
    "privacy": {"privacy", "data", "personal", "biometric", "consent", "sensitive", "governance", "protection"},
    "bias": {"bias", "biased", "fairness", "profiling"},
    "discrimination": {"discrimination", "discriminatory", "equal", "equality"},
    "transparency": {"transparency", "notice", "disclose", "disclosure", "logging", "documentation", "traceability"},
    "rights_violation": {"rights", "fundamental", "civil", "human", "dignity"},
    "interpretability": {"interpretability", "explainability", "interpretable", "explainable"},
    "governance_other": {"governance", "oversight", "risk", "safety", "assurance", "framework"},
}


@dataclass(slots=True)
class OracleEvaluationArtifacts:
    case_manifest_path: str
    scaffold_manifest_path: str
    predictions_path: str
    condition_summary_path: str
    scaffold_summary_path: str
    gold_labels_path: str
    human_eval_sheet_path: str
    human_eval_summary_path: str


def activation_oracle_is_compatible(config: ExperimentConfig) -> bool:
    model_name = str(config.backbone.model_name).lower()
    return "gemma-2-9b" in model_name


def activation_oracle_skip_reason(config: ExperimentConfig) -> str | None:
    if not config.activation_oracle.enabled:
        return "Activation Oracle is disabled in the configuration."
    if not activation_oracle_is_compatible(config):
        return "Activation Oracle v1 is only enabled for Gemma 2 9B compatible backbones."
    return None


def build_scaffold_manifest(case_manifest: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if case_manifest.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for row in case_manifest.itertuples(index=False):
        base = row._asdict()
        source_case_id = str(base["case_id"])
        for frame in config.audit.scaffold_eval.frames:
            wrapped = dict(base)
            wrapped["source_case_id"] = source_case_id
            wrapped["scaffold_frame"] = frame
            wrapped["case_id"] = f"{source_case_id}__{frame}"
            wrapped["text"] = wrap_scaffold_text(str(base["text"]), frame)
            rows.append(wrapped)
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest["segment_id"] = manifest["case_id"].astype(str)
    return manifest


def wrap_scaffold_text(text: str, frame: str) -> str:
    normalized = str(text).strip()
    if frame == "raw_excerpt":
        return normalized
    if frame == "analyst_question":
        return (
            "Analyst question: What is the main policy obligation expressed in the excerpt below?\n\n"
            f"Excerpt:\n{normalized}"
        )
    if frame == "compliance_memo":
        return (
            "Compliance memo task: Summarize the operative obligations in the excerpt below in concise compliance language.\n\n"
            f"Excerpt:\n{normalized}"
        )
    if frame == "neutral_restatement":
        return (
            "Neutral restatement task: Restate the governance content of the excerpt below without adding new obligations.\n\n"
            f"Excerpt:\n{normalized}"
        )
    if frame == "adversarial_bland":
        return (
            "Generic legal summary task: Describe the excerpt below in intentionally bland legal language that avoids domain-specific framing.\n\n"
            f"Excerpt:\n{normalized}"
        )
    return normalized


def build_oracle_requests(
    scored_features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: ExperimentConfig,
    include_controls: bool,
) -> pd.DataFrame:
    if scored_features.empty or manifest.empty:
        return pd.DataFrame()
    requests: list[dict[str, object]] = []
    target_layers = [int(layer) for layer in config.activation_oracle.target_layers]
    primary_layer = int(config.activation_oracle.primary_layer)
    manifest_columns = [
        column
        for column in [
            "segment_id",
            "source_case_id",
            "family_id",
            "family_label",
            "scaffold_frame",
            "text",
        ]
        if column in manifest.columns
    ]
    metadata = manifest[manifest_columns].drop_duplicates("segment_id").set_index("segment_id").to_dict(orient="index")

    for segment_id, group in scored_features.groupby("segment_id"):
        segment_metadata = metadata.get(segment_id, {})
        source_case_id = str(segment_metadata.get("source_case_id", segment_id))
        scaffold_frame = str(segment_metadata.get("scaffold_frame", "raw_excerpt"))
        family_id = str(segment_metadata.get("family_id", ""))
        family_label = str(segment_metadata.get("family_label", ""))
        excerpt_text = str(segment_metadata.get("text", group["text"].iloc[0] if "text" in group.columns else ""))
        policy_group = group.loc[group["ranking_family"] == "policy_specific"].copy()
        policy_signal_total = float(policy_group["pooled_activation"].astype(float).sum()) if not policy_group.empty else 0.0
        dominant_proxy = _infer_dominant_proxy(policy_group if not policy_group.empty else group)
        for unit_type in config.activation_oracle.explanation_units:
            summary = _build_unit_summary(
                feature_group=group,
                excerpt_text=excerpt_text,
                unit_type=unit_type,
                target_layers=target_layers,
                primary_layer=primary_layer,
            )
            requests.append(
                {
                    "request_id": f"{segment_id}__{unit_type}__{AO_REAL_CONDITION}",
                    "segment_id": str(segment_id),
                    "source_case_id": source_case_id,
                    "family_id": family_id,
                    "family_label": family_label,
                    "scaffold_frame": scaffold_frame,
                    "condition": AO_REAL_CONDITION,
                    "unit_type": unit_type,
                    "text": excerpt_text,
                    "activation_evidence": summary["activation_evidence"],
                    "request_status": summary["request_status"],
                    "primary_layer": int(summary["primary_layer"]) if summary["primary_layer"] is not None else np.nan,
                    "dominant_proxy": dominant_proxy,
                    "policy_signal_total": policy_signal_total,
                    "top_feature_names": summary["top_feature_names"],
                    "top_span_texts": summary["top_span_texts"],
                }
            )

    frame = pd.DataFrame(requests)
    if frame.empty:
        return frame
    control_rows: list[pd.DataFrame] = []
    if include_controls and config.activation_oracle.include_text_only_control:
        ready = frame.loc[frame["request_status"] == AO_READY_STATUS].copy()
        if not ready.empty:
            text_only = ready.copy()
            text_only["condition"] = AO_TEXT_ONLY_CONDITION
            text_only["request_id"] = text_only["segment_id"] + "__" + text_only["unit_type"] + "__" + AO_TEXT_ONLY_CONDITION
            text_only["activation_evidence"] = ""
            text_only["activation_donor_source_case_id"] = ""
            control_rows.append(text_only)
    if include_controls and config.activation_oracle.include_shuffled_activation_control:
        shuffled = build_shuffled_activation_controls(frame, int(config.splits.seed))
        if not shuffled.empty:
            control_rows.append(shuffled)
    if control_rows:
        frame = pd.concat([frame, *control_rows], ignore_index=True)
    return frame.sort_values(["segment_id", "unit_type", "condition"]).reset_index(drop=True)


def build_shuffled_activation_controls(real_requests: pd.DataFrame, seed: int) -> pd.DataFrame:
    if real_requests.empty:
        return pd.DataFrame()
    ready = real_requests.loc[
        (real_requests["condition"] == AO_REAL_CONDITION) & (real_requests["request_status"] == AO_READY_STATUS)
    ].copy()
    if ready.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for row in ready.itertuples(index=False):
        candidates = ready.loc[
            (ready["unit_type"] == row.unit_type)
            & (ready["source_case_id"] != row.source_case_id)
            & (ready["family_id"] != row.family_id)
        ].copy()
        if "scaffold_frame" in ready.columns and row.scaffold_frame in AO_SCAFFOLD_FRAMES:
            same_frame = candidates.loc[candidates["scaffold_frame"] == row.scaffold_frame].copy()
            if not same_frame.empty:
                candidates = same_frame
        if candidates.empty:
            continue
        donor = candidates.iloc[int(rng.integers(0, len(candidates)))]
        shuffled = dict(row._asdict())
        shuffled["condition"] = AO_SHUFFLED_CONDITION
        shuffled["request_id"] = f"{row.segment_id}__{row.unit_type}__{AO_SHUFFLED_CONDITION}"
        shuffled["activation_evidence"] = str(donor["activation_evidence"])
        shuffled["activation_donor_source_case_id"] = str(donor["source_case_id"])
        shuffled["activation_donor_family_id"] = str(donor["family_id"])
        rows.append(shuffled)
    return pd.DataFrame(rows)


def parse_structured_oracle_response(raw_text: str, regulatory_families: list[str]) -> dict[str, object] | None:
    if not raw_text.strip():
        return None
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    concept_summary = str(payload.get("concept_summary", "")).strip()
    specificity_label = _canonical_specificity(str(payload.get("specificity_label", "")))
    obligation_family = _canonical_obligation_family(str(payload.get("obligation_family", "")))
    regulatory_family = _canonical_regulatory_family(str(payload.get("regulatory_family", "")), regulatory_families)
    rationale_short = str(payload.get("rationale_short", "")).strip()
    confidence_raw = payload.get("confidence", 0)
    try:
        confidence = int(float(confidence_raw))
    except (TypeError, ValueError):
        confidence = 0
    if not concept_summary:
        return None
    return {
        "concept_summary": concept_summary,
        "specificity_label": specificity_label,
        "obligation_family": obligation_family,
        "regulatory_family": regulatory_family,
        "confidence": max(0, min(100, confidence)),
        "rationale_short": rationale_short,
    }


class ActivationOracleBackend:
    def __init__(self, config: ExperimentConfig, regulatory_families: list[str]):
        self.config = config
        self.regulatory_families = regulatory_families
        self.model = None
        self.tokenizer = None
        self.backend_name = "heuristic_fallback"
        self.load_error = ""

    def explain(self, requests: pd.DataFrame) -> pd.DataFrame:
        if requests.empty:
            return pd.DataFrame()
        self._ensure_model_loaded()
        rows: list[dict[str, object]] = []
        for row in requests.itertuples(index=False):
            payload = row._asdict()
            if str(payload.get("request_status", "")) != AO_READY_STATUS and str(payload.get("condition", "")) != AO_TEXT_ONLY_CONDITION:
                rows.append(
                    {
                        **payload,
                        "concept_summary": "",
                        "specificity_label": "",
                        "obligation_family": "",
                        "regulatory_family": "",
                        "confidence": 0,
                        "rationale_short": str(payload.get("request_status", "")),
                        "generation_backend": "skipped",
                    }
                )
                continue
            response = self._generate_or_fallback(payload)
            rows.append(
                {
                    **payload,
                    **response,
                    "generation_backend": self.backend_name,
                }
            )
        return pd.DataFrame(rows)

    def close(self) -> None:
        if self.model is not None:
            del self.model
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_model_loaded(self) -> None:
        if self.model is not None or self.load_error:
            return
        if PeftModel is None:
            self.load_error = "peft is not installed"
            return
        if not activation_oracle_is_compatible(self.config):
            self.load_error = activation_oracle_skip_reason(self.config) or "incompatible backbone"
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.activation_oracle.target_backbone_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.activation_oracle.target_backbone_name,
                torch_dtype=resolve_torch_dtype(self.config.backbone.dtype),
                trust_remote_code=self.config.backbone.trust_remote_code,
            )
            self.model = PeftModel.from_pretrained(base_model, self.config.activation_oracle.backend_model_name)
            self.model.to(self.config.backbone.device)
            self.model.eval()
            self.backend_name = "peft_text_generator"
        except Exception as exc:  # pragma: no cover
            self.load_error = str(exc)
            self.model = None
            self.tokenizer = None
            self.backend_name = "heuristic_fallback"

    def _generate_or_fallback(self, payload: dict[str, object]) -> dict[str, object]:
        if self.model is None or self.tokenizer is None:
            return _heuristic_oracle_response(payload, self.regulatory_families)
        prompt = _build_oracle_prompt(payload, self.regulatory_families)
        try:  # pragma: no cover
            generated = self._generate_text(prompt)
            parsed = parse_structured_oracle_response(generated, self.regulatory_families)
            if parsed is not None:
                return parsed
        except Exception:
            self.backend_name = "heuristic_fallback"
        return _heuristic_oracle_response(payload, self.regulatory_families)

    def _generate_text(self, prompt: str) -> str:  # pragma: no cover
        assert self.model is not None
        assert self.tokenizer is not None
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            model_inputs = {"input_ids": encoded.to(self.config.backbone.device)}
            prompt_length = int(model_inputs["input_ids"].shape[-1])
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            model_inputs = {key: value.to(self.config.backbone.device) for key, value in encoded.items()}
            prompt_length = int(model_inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=int(self.config.activation_oracle.max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output[0][prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def build_batch_oracle_report_summary(
    bundle_predictions: pd.DataFrame,
    window_predictions: pd.DataFrame,
) -> pd.DataFrame:
    if bundle_predictions.empty and window_predictions.empty:
        return pd.DataFrame()
    if bundle_predictions.empty:
        bundle_predictions = pd.DataFrame(columns=["segment_id", "source_case_id", "policy_signal_total", "concept_summary", "specificity_label", "obligation_family", "regulatory_family", "confidence", "rationale_short"])
    if window_predictions.empty:
        window_predictions = pd.DataFrame(columns=["segment_id", "source_case_id", "concept_summary", "specificity_label", "obligation_family", "regulatory_family", "confidence", "rationale_short"])
    left = bundle_predictions.rename(
        columns={
            "concept_summary": "bundle_concept_summary",
            "specificity_label": "bundle_specificity_label",
            "obligation_family": "bundle_obligation_family",
            "regulatory_family": "bundle_regulatory_family",
            "confidence": "bundle_confidence",
            "rationale_short": "bundle_rationale_short",
        }
    )
    right = window_predictions.rename(
        columns={
            "concept_summary": "window_concept_summary",
            "specificity_label": "window_specificity_label",
            "obligation_family": "window_obligation_family",
            "regulatory_family": "window_regulatory_family",
            "confidence": "window_confidence",
            "rationale_short": "window_rationale_short",
        }
    )
    merged = left.merge(
        right[
            [
                "segment_id",
                "source_case_id",
                "window_concept_summary",
                "window_specificity_label",
                "window_obligation_family",
                "window_regulatory_family",
                "window_confidence",
                "window_rationale_short",
            ]
        ],
        on=["segment_id", "source_case_id"],
        how="outer",
    )
    merged["agreement_status"] = merged.apply(_agreement_label, axis=1)
    return merged.sort_values(["policy_signal_total", "segment_id"], ascending=[False, True]).reset_index(drop=True)


def build_gold_label_sheet(
    source_cases: pd.DataFrame,
    config: ExperimentConfig,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sampled = _family_stratified_sample(
        source_cases=source_cases,
        total_cases=int(config.audit.scaffold_eval.gold_label_cases),
        seed=int(config.splits.seed),
    )
    frame = sampled[
        ["source_case_id", "family_id", "family_label", "text"]
    ].drop_duplicates("source_case_id").copy()
    frame["regulatory_family"] = frame["family_id"].astype(str)
    frame["primary_obligation_family"] = ""
    frame["specificity_label"] = ""
    if existing is None or existing.empty:
        return frame
    merged = frame.merge(
        existing[
            [
                "source_case_id",
                "regulatory_family",
                "primary_obligation_family",
                "specificity_label",
            ]
        ],
        on="source_case_id",
        how="left",
        suffixes=("", "_existing"),
    )
    for column in ["regulatory_family", "primary_obligation_family", "specificity_label"]:
        existing_column = f"{column}_existing"
        merged[column] = merged[existing_column].replace("", pd.NA).fillna(merged[column])
        merged = merged.drop(columns=[existing_column])
    return merged


def build_human_eval_sheet(
    source_cases: pd.DataFrame,
    requests: pd.DataFrame,
    predictions: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    sampled = _family_stratified_sample(
        source_cases=source_cases,
        total_cases=int(config.audit.scaffold_eval.human_eval_cases),
        seed=int(config.splits.seed) + 11,
    )
    raw_requests = requests.loc[
        (requests["condition"] == AO_REAL_CONDITION) & (requests["scaffold_frame"] == "raw_excerpt")
    ].copy()
    raw_predictions = predictions.loc[
        (predictions["condition"] == AO_REAL_CONDITION) & (predictions["scaffold_frame"] == "raw_excerpt")
    ].copy()
    snapshots = _build_human_eval_snapshots(raw_requests, raw_predictions)

    rows: list[dict[str, object]] = []
    for row in sampled.itertuples(index=False):
        snapshot = snapshots.get(str(row.source_case_id), {})
        condition_payloads = {
            "base_audit_only": snapshot.get("base_audit_only", ""),
            "base_plus_ao": snapshot.get("base_plus_ao", ""),
            "ao_only": snapshot.get("ao_only", ""),
        }
        for annotator_index in range(1, int(config.audit.scaffold_eval.human_eval_annotators) + 1):
            for condition, report_snapshot in condition_payloads.items():
                rows.append(
                    {
                        "source_case_id": str(row.source_case_id),
                        "family_id": str(row.family_id),
                        "family_label": str(row.family_label),
                        "text": str(row.text),
                        "condition": condition,
                        "annotator_id": f"annotator_{annotator_index}",
                        "report_snapshot": report_snapshot,
                        "regulatory_family_answer": "",
                        "primary_obligation_family_answer": "",
                        "specificity_label_answer": "",
                        "usefulness": np.nan,
                        "confidence": np.nan,
                        "time_seconds": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_oracle_predictions(
    predictions: pd.DataFrame,
    gold_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    gold = gold_labels.copy() if gold_labels is not None else pd.DataFrame()
    if not gold.empty:
        gold = gold.drop_duplicates("source_case_id")

    summary_rows: list[dict[str, object]] = []
    grouped = predictions.groupby(["condition", "unit_type"], dropna=False)
    specificity_rates: dict[tuple[str, str], float] = {}
    for (condition, unit_type), group in grouped:
        merged = group.merge(gold, on="source_case_id", how="left", suffixes=("", "_gold")) if not gold.empty else group.copy()
        specificity_rate = float((group["specificity_label"] == "policy_specific").mean()) if not group.empty else np.nan
        specificity_rates[(str(condition), str(unit_type))] = specificity_rate
        summary_rows.append(
            {
                "condition": str(condition),
                "unit_type": str(unit_type),
                "prediction_count": int(len(group)),
                "regulatory_family_accuracy": _label_accuracy(merged, "regulatory_family", "regulatory_family_gold"),
                "primary_obligation_macro_f1": _macro_f1(merged, "obligation_family", "primary_obligation_family"),
                "specificity_accuracy": _label_accuracy(merged, "specificity_label", "specificity_label_gold"),
                "best_proxy_agreement_rate": _best_proxy_agreement_rate(group),
                "policy_specific_rate": specificity_rate,
                "policy_specificity_margin": np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        for row in summary.itertuples(index=False):
            if row.condition != AO_REAL_CONDITION:
                continue
            controls = [
                specificity_rates.get((AO_TEXT_ONLY_CONDITION, row.unit_type), np.nan),
                specificity_rates.get((AO_SHUFFLED_CONDITION, row.unit_type), np.nan),
            ]
            controls = [value for value in controls if pd.notna(value)]
            margin = float(row.policy_specific_rate - float(np.mean(controls))) if controls else np.nan
            summary.loc[
                (summary["condition"] == row.condition) & (summary["unit_type"] == row.unit_type),
                "policy_specificity_margin",
            ] = margin

    scaffold_rows: list[dict[str, object]] = []
    for (condition, unit_type), group in predictions.groupby(["condition", "unit_type"], dropna=False):
        raw = group.loc[group["scaffold_frame"] == "raw_excerpt"].copy()
        comparison = group.loc[group["scaffold_frame"] != "raw_excerpt"].copy()
        if raw.empty or comparison.empty:
            continue
        anchor = raw.drop_duplicates("source_case_id").set_index("source_case_id")
        for frame_name, frame_group in comparison.groupby("scaffold_frame"):
            merged = frame_group.merge(
                anchor[["regulatory_family", "obligation_family", "specificity_label"]],
                left_on="source_case_id",
                right_index=True,
                how="inner",
                suffixes=("", "_anchor"),
            )
            if merged.empty:
                continue
            scaffold_rows.append(
                {
                    "condition": str(condition),
                    "unit_type": str(unit_type),
                    "scaffold_frame": str(frame_name),
                    "case_count": int(len(merged)),
                    "regulatory_family_retention_rate": float(
                        (merged["regulatory_family"] == merged["regulatory_family_anchor"]).mean()
                    ),
                    "obligation_family_retention_rate": float(
                        (merged["obligation_family"] == merged["obligation_family_anchor"]).mean()
                    ),
                    "specificity_retention_rate": float(
                        (merged["specificity_label"] == merged["specificity_label_anchor"]).mean()
                    ),
                }
            )
    return summary, pd.DataFrame(scaffold_rows)


def summarize_human_eval_sheet(sheet: pd.DataFrame, gold_labels: pd.DataFrame) -> pd.DataFrame:
    if sheet.empty:
        return pd.DataFrame()
    gold = gold_labels.copy() if gold_labels is not None else pd.DataFrame()
    if not gold.empty:
        gold = gold.drop_duplicates("source_case_id")
    merged = sheet.merge(gold, on="source_case_id", how="left", suffixes=("", "_gold")) if not gold.empty else sheet.copy()
    rows: list[dict[str, object]] = []
    for condition, group in merged.groupby("condition"):
        completed = group.loc[group["regulatory_family_answer"].astype(str).str.len() > 0].copy()
        rows.append(
            {
                "condition": str(condition),
                "completed_rows": int(len(completed)),
                "regulatory_family_accuracy": _label_accuracy(completed, "regulatory_family_answer", "regulatory_family"),
                "primary_obligation_accuracy": _label_accuracy(
                    completed,
                    "primary_obligation_family_answer",
                    "primary_obligation_family",
                ),
                "specificity_accuracy": _label_accuracy(completed, "specificity_label_answer", "specificity_label"),
                "mean_usefulness": float(pd.to_numeric(group["usefulness"], errors="coerce").mean()),
                "mean_confidence": float(pd.to_numeric(group["confidence"], errors="coerce").mean()),
                "mean_time_seconds": float(pd.to_numeric(group["time_seconds"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_unit_summary(
    feature_group: pd.DataFrame,
    excerpt_text: str,
    unit_type: str,
    target_layers: list[int],
    primary_layer: int,
) -> dict[str, object]:
    group = feature_group.copy()
    if "best_proxy" in group.columns:
        group["best_proxy"] = group["best_proxy"].fillna("").astype(str)
    if unit_type == "feature_bundle":
        candidates = group.loc[
            (group["ranking_family"] == "policy_specific") & (group["layer"].astype(int).isin(target_layers))
        ].copy()
        candidates["primary_priority"] = (candidates["layer"].astype(int) != primary_layer).astype(int)
        candidates = candidates.sort_values(
            ["primary_priority", "pooled_activation", "layer", "feature_id"],
            ascending=[True, False, True, True],
        ).head(5)
        if len(candidates) < 3:
            return {
                "activation_evidence": "",
                "request_status": AO_INSUFFICIENT_BUNDLE_STATUS,
                "primary_layer": primary_layer if primary_layer in target_layers else (target_layers[0] if target_layers else None),
                "top_feature_names": [],
                "top_span_texts": [],
            }
        evidence_lines = []
        top_names: list[str] = []
        top_spans: list[str] = []
        for row in candidates.itertuples(index=False):
            display_name = _display_name_from_row(row)
            top_names.append(display_name)
            span_text = str(getattr(row, "top_token_span_text", "") or "")
            if span_text:
                top_spans.append(span_text)
            evidence_lines.append(
                f"Layer {int(row.layer)} feature {int(row.feature_id)} named '{display_name}' has pooled activation "
                f"{float(row.pooled_activation):.4f}, proxy {str(getattr(row, 'best_proxy', '') or 'unknown')}, span '{span_text}'."
            )
        return {
            "activation_evidence": "\n".join(evidence_lines),
            "request_status": AO_READY_STATUS,
            "primary_layer": primary_layer,
            "top_feature_names": top_names,
            "top_span_texts": top_spans,
        }

    primary_candidates = group.loc[group["layer"].astype(int) == primary_layer].copy()
    if primary_candidates.empty:
        backup_layers = [layer for layer in target_layers if int(layer) in set(group["layer"].astype(int).tolist())]
        if backup_layers:
            primary_layer = int(backup_layers[0])
            primary_candidates = group.loc[group["layer"].astype(int) == primary_layer].copy()
    if primary_candidates.empty:
        return {
            "activation_evidence": "",
            "request_status": AO_INSUFFICIENT_BUNDLE_STATUS,
            "primary_layer": None,
            "top_feature_names": [],
            "top_span_texts": [],
        }
    primary_candidates = primary_candidates.sort_values(
        ["pooled_activation", "feature_id"],
        ascending=[False, True],
    ).head(8)
    window_lines = [f"Excerpt: {excerpt_text}"]
    top_names = []
    top_spans = []
    for row in primary_candidates.itertuples(index=False):
        display_name = _display_name_from_row(row)
        span_text = str(getattr(row, "top_token_span_text", "") or "")
        top_names.append(display_name)
        if span_text:
            top_spans.append(span_text)
        window_lines.append(
            f"Activation at layer {int(row.layer)} on span '{span_text}' with pooled activation "
            f"{float(row.pooled_activation):.4f} and label '{display_name}'."
        )
    return {
        "activation_evidence": "\n".join(window_lines),
        "request_status": AO_READY_STATUS,
        "primary_layer": primary_layer,
        "top_feature_names": top_names,
        "top_span_texts": top_spans,
    }


def _build_oracle_prompt(payload: dict[str, object], regulatory_families: list[str]) -> str:
    activation_evidence = str(payload.get("activation_evidence", "")).strip()
    unit_type = str(payload.get("unit_type", "feature_bundle"))
    condition = str(payload.get("condition", AO_REAL_CONDITION))
    text = str(payload.get("text", "")).strip()
    evidence_block = activation_evidence if activation_evidence else "Activation evidence is unavailable. Use the excerpt text only."
    return (
        "You are an activation explainer for a policy audit system.\n"
        "Return only a valid JSON object with the following fields:\n"
        "concept_summary, specificity_label, obligation_family, regulatory_family, confidence, rationale_short.\n"
        f"Allowed specificity_label values: {', '.join(AO_SPECIFICITY_LABELS)}.\n"
        f"Allowed obligation_family values: {', '.join(AO_OBLIGATION_FAMILIES)}.\n"
        f"Allowed regulatory_family values: {', '.join(regulatory_families)}.\n"
        f"Explanation unit: {unit_type}.\n"
        f"Condition: {condition}.\n"
        f"Excerpt text:\n{text}\n\n"
        f"Activation evidence:\n{evidence_block}\n\n"
        "Questions to answer:\n"
        "1. What policy concept is most represented in these activations?\n"
        "2. Is the signal mostly policy-specific governance content or generic legal boilerplate?\n"
        "3. Which obligation family and regulatory family are most likely represented here?\n"
    )


def _heuristic_oracle_response(payload: dict[str, object], regulatory_families: list[str]) -> dict[str, object]:
    condition = str(payload.get("condition", AO_REAL_CONDITION))
    text = str(payload.get("text", ""))
    activation_evidence = str(payload.get("activation_evidence", ""))
    dominant_proxy = str(payload.get("dominant_proxy", ""))
    evidence_text = activation_evidence if activation_evidence else text
    obligation_family = _infer_obligation_family(evidence_text, dominant_proxy if condition == AO_REAL_CONDITION else "")
    specificity_label = _infer_specificity_label(
        evidence_text=evidence_text,
        text=text,
        condition=condition,
        dominant_proxy=dominant_proxy,
    )
    regulatory_family = _infer_regulatory_family(evidence_text, regulatory_families)
    concept_summary = _build_concept_summary(evidence_text, obligation_family, specificity_label)
    rationale_short = _build_rationale_short(payload, obligation_family, specificity_label)
    confidence = 78 if condition == AO_REAL_CONDITION else 54 if condition == AO_TEXT_ONLY_CONDITION else 42
    return {
        "concept_summary": concept_summary,
        "specificity_label": specificity_label,
        "obligation_family": obligation_family,
        "regulatory_family": regulatory_family,
        "confidence": confidence,
        "rationale_short": rationale_short,
    }


def _build_human_eval_snapshots(requests: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, dict[str, str]]:
    snapshots: dict[str, dict[str, str]] = {}
    grouped_requests = requests.groupby("source_case_id")
    grouped_predictions = predictions.groupby("source_case_id")
    for source_case_id in sorted(set(grouped_requests.groups.keys()) | set(grouped_predictions.groups.keys())):
        request_group = grouped_requests.get_group(source_case_id) if source_case_id in grouped_requests.groups else pd.DataFrame()
        prediction_group = grouped_predictions.get_group(source_case_id) if source_case_id in grouped_predictions.groups else pd.DataFrame()
        base_lines: list[str] = []
        for row in request_group.itertuples(index=False):
            unit_label = "Feature bundle" if row.unit_type == "feature_bundle" else "Activation window"
            evidence = str(row.activation_evidence or "").strip()
            if evidence:
                base_lines.append(f"{unit_label}: {evidence}")
        ao_lines: list[str] = []
        for row in prediction_group.itertuples(index=False):
            unit_label = "Feature bundle" if row.unit_type == "feature_bundle" else "Activation window"
            if not str(row.concept_summary).strip():
                continue
            ao_lines.append(
                f"{unit_label}: {row.concept_summary} | specificity {row.specificity_label} | "
                f"obligation {row.obligation_family} | regulatory family {row.regulatory_family}."
            )
        base_snapshot = "\n".join(base_lines)
        ao_snapshot = "\n".join(ao_lines)
        snapshots[str(source_case_id)] = {
            "base_audit_only": base_snapshot,
            "base_plus_ao": "\n\n".join([item for item in [base_snapshot, ao_snapshot] if item]),
            "ao_only": ao_snapshot,
        }
    return snapshots


def _family_stratified_sample(source_cases: pd.DataFrame, total_cases: int, seed: int) -> pd.DataFrame:
    if source_cases.empty:
        return pd.DataFrame()
    deduped = source_cases.drop_duplicates("source_case_id").copy()
    family_groups = list(deduped.groupby("family_id"))
    if not family_groups:
        return deduped.head(total_cases).copy()
    base_quota = max(1, total_cases // len(family_groups))
    sampled_frames: list[pd.DataFrame] = []
    remaining_frames: list[pd.DataFrame] = []
    for index, (_, group) in enumerate(family_groups):
        take = min(len(group), base_quota)
        sampled_group = group.sample(n=take, random_state=seed + index) if len(group) > take else group.copy()
        sampled_frames.append(sampled_group)
        remaining_frames.append(group.drop(sampled_group.index, errors="ignore"))
    sampled = pd.concat(sampled_frames, ignore_index=True)
    if len(sampled) < min(total_cases, len(deduped)):
        remainder_needed = min(total_cases, len(deduped)) - len(sampled)
        remainder = pd.concat(remaining_frames, ignore_index=True)
        if not remainder.empty:
            sampled = pd.concat(
                [sampled, remainder.sample(n=min(remainder_needed, len(remainder)), random_state=seed + 101)],
                ignore_index=True,
            )
    return sampled.head(total_cases).reset_index(drop=True)


def _infer_dominant_proxy(group: pd.DataFrame) -> str:
    if group.empty or "best_proxy" not in group.columns:
        return ""
    frame = group.copy()
    frame["best_proxy"] = frame["best_proxy"].fillna("").astype(str)
    frame = frame.loc[frame["best_proxy"] != ""].copy()
    if frame.empty:
        return ""
    ranked = frame.groupby("best_proxy")["pooled_activation"].sum().sort_values(ascending=False)
    return str(ranked.index[0]) if not ranked.empty else ""


def _infer_obligation_family(evidence_text: str, dominant_proxy: str) -> str:
    canonical_proxy = _canonical_obligation_family(dominant_proxy)
    if canonical_proxy in AO_OBLIGATION_FAMILIES and canonical_proxy != "governance_other":
        return canonical_proxy
    scores: dict[str, float] = {}
    tokens = _content_tokens(evidence_text)
    for family, keywords in _OBLIGATION_KEYWORDS.items():
        scores[family] = float(len(tokens & keywords))
    best_family = max(scores.items(), key=lambda item: item[1])[0] if scores else "governance_other"
    return best_family if scores.get(best_family, 0.0) > 0.0 else "governance_other"


def _infer_specificity_label(
    evidence_text: str,
    text: str,
    condition: str,
    dominant_proxy: str,
) -> str:
    if dominant_proxy and condition == AO_REAL_CONDITION:
        return "policy_specific"
    tokens = _content_tokens(f"{evidence_text} {text}")
    policy_hits = 0
    for keywords in _OBLIGATION_KEYWORDS.values():
        policy_hits += len(tokens & keywords)
    if condition == AO_SHUFFLED_CONDITION:
        return "generic_legalese" if policy_hits <= 1 else "policy_specific"
    if condition == AO_TEXT_ONLY_CONDITION:
        return "policy_specific" if policy_hits >= 2 else "generic_legalese"
    return "policy_specific" if policy_hits >= 1 else "generic_legalese"


def _infer_regulatory_family(evidence_text: str, regulatory_families: list[str]) -> str:
    if not regulatory_families:
        return ""
    text_tokens = _content_tokens(evidence_text)
    if not text_tokens:
        return regulatory_families[0]
    scores: dict[str, int] = {}
    for family in regulatory_families:
        family_tokens = _content_tokens(family.replace("_", " "))
        scores[family] = len(text_tokens & family_tokens)
    best_family = max(scores.items(), key=lambda item: item[1])[0]
    return best_family


def _build_concept_summary(evidence_text: str, obligation_family: str, specificity_label: str) -> str:
    excerpt = " ".join(str(evidence_text).split())
    excerpt = excerpt[:180].strip()
    if specificity_label == "generic_legalese":
        return f"This signal reads as generic legal boilerplate with weak policy specificity. Evidence: {excerpt}"
    if obligation_family == "privacy":
        return f"This signal emphasizes privacy and data governance obligations. Evidence: {excerpt}"
    if obligation_family in {"bias", "discrimination"}:
        return f"This signal emphasizes fairness or discrimination-related policy constraints. Evidence: {excerpt}"
    if obligation_family in {"transparency", "interpretability"}:
        return f"This signal emphasizes transparency or explainability obligations. Evidence: {excerpt}"
    if obligation_family == "rights_violation":
        return f"This signal emphasizes civil or human rights concerns. Evidence: {excerpt}"
    return f"This signal emphasizes policy governance and implementation requirements. Evidence: {excerpt}"


def _build_rationale_short(payload: dict[str, object], obligation_family: str, specificity_label: str) -> str:
    spans = [str(item) for item in payload.get("top_span_texts", []) if str(item).strip()]
    span_note = f"Top spans include {', '.join(spans[:2])}." if spans else "No localized span evidence was retained."
    proxy = str(payload.get("dominant_proxy", "") or "unknown")
    return f"Specificity is {specificity_label}; inferred obligation family is {obligation_family}; dominant proxy is {proxy}. {span_note}"


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z]{4,}", str(text).lower())
        if token not in _COMMON_STOPWORDS
    }


def _canonical_specificity(value: str) -> str:
    lowered = value.strip().lower()
    if "generic" in lowered or "boilerplate" in lowered:
        return "generic_legalese"
    return "policy_specific"


def _canonical_obligation_family(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in AO_OBLIGATION_FAMILIES:
        return lowered
    for family in AO_OBLIGATION_FAMILIES:
        if family in lowered:
            return family
    if "human rights" in lowered or "civil rights" in lowered:
        return "rights_violation"
    if "explain" in lowered:
        return "interpretability"
    if "fair" in lowered or "bias" in lowered:
        return "bias"
    if "privacy" in lowered or "data" in lowered:
        return "privacy"
    if "transparen" in lowered or "notice" in lowered:
        return "transparency"
    return "governance_other"


def _canonical_regulatory_family(value: str, regulatory_families: list[str]) -> str:
    lowered = value.strip().lower().replace(" ", "_")
    if lowered in regulatory_families:
        return lowered
    if not regulatory_families:
        return ""
    match_scores = {
        family: len(_content_tokens(lowered) & _content_tokens(family.replace("_", " ")))
        for family in regulatory_families
    }
    return max(match_scores.items(), key=lambda item: item[1])[0]


def _display_name_from_row(row: Any) -> str:
    generated_name = str(getattr(row, "generated_name", "") or "")
    if generated_name:
        return generated_name
    return f"Layer {int(row.layer)} feature {int(row.feature_id)}"


def _label_accuracy(frame: pd.DataFrame, pred_col: str, gold_col: str) -> float:
    if frame.empty or pred_col not in frame.columns or gold_col not in frame.columns:
        return np.nan
    valid = frame.loc[frame[pred_col].astype(str).str.len() > 0].copy()
    valid = valid.loc[valid[gold_col].astype(str).str.len() > 0].copy()
    if valid.empty:
        return np.nan
    return float((valid[pred_col].astype(str) == valid[gold_col].astype(str)).mean())


def _macro_f1(frame: pd.DataFrame, pred_col: str, gold_col: str) -> float:
    if frame.empty or pred_col not in frame.columns or gold_col not in frame.columns:
        return np.nan
    valid = frame.loc[frame[pred_col].astype(str).str.len() > 0].copy()
    valid = valid.loc[valid[gold_col].astype(str).str.len() > 0].copy()
    if valid.empty:
        return np.nan
    return float(f1_score(valid[gold_col].astype(str), valid[pred_col].astype(str), average="macro"))


def _best_proxy_agreement_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "dominant_proxy" not in frame.columns:
        return np.nan
    valid = frame.loc[frame["dominant_proxy"].astype(str).str.len() > 0].copy()
    if valid.empty:
        return np.nan
    return float(
        (
            valid["obligation_family"].astype(str).map(_canonical_obligation_family)
            == valid["dominant_proxy"].astype(str).map(_canonical_obligation_family)
        ).mean()
    )


def _agreement_label(row: pd.Series) -> str:
    bundle_obligation = str(row.get("bundle_obligation_family", "") or "")
    window_obligation = str(row.get("window_obligation_family", "") or "")
    bundle_specificity = str(row.get("bundle_specificity_label", "") or "")
    window_specificity = str(row.get("window_specificity_label", "") or "")
    obligation_match = bundle_obligation and bundle_obligation == window_obligation
    specificity_match = bundle_specificity and bundle_specificity == window_specificity
    if obligation_match and specificity_match:
        return "agree"
    if obligation_match or specificity_match:
        return "partial_agree"
    return "disagree"
