from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch


GENERIC_PROXY_WORDS = {
    "applications",
    "harms",
    "incentives",
    "risk",
    "risks",
    "factors",
    "strategies",
    "about",
    "based",
    "and",
    "or",
    "of",
    "the",
    "including",
    "for",
    "on",
}


PROXY_SYNONYM_MAP: dict[str, list[str]] = {
    "bias": [
        "biased",
        "unfair",
        "fairness",
        "inequity",
        "inequities",
        "disparate",
        "disparity",
        "disparities",
    ],
    "discrimination": [
        "discriminatory",
        "protected class",
        "protected classes",
        "disparate impact",
        "equal treatment",
    ],
    "privacy": [
        "private",
        "personal data",
        "data privacy",
        "data protection",
        "retention",
        "surveillance",
        "confidentiality",
        "consent",
    ],
    "rights": [
        "right",
        "civil rights",
        "human rights",
        "fundamental rights",
        "civil liberties",
        "liberties",
    ],
    "transparency": [
        "transparent",
        "disclosure",
        "disclosures",
        "reporting",
        "audit trail",
        "openness",
    ],
    "interpretability": [
        "interpretable",
        "interpretability",
        "explainability",
        "explanation",
        "explanations",
    ],
    "explainability": [
        "interpretable",
        "interpretability",
        "explainability",
        "explanation",
        "explanations",
    ],
}


@dataclass(frozen=True)
class MatchConfig:
    method: str = "sentence"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    allow_replacement: bool = False
    weight_text: float = 0.45
    weight_authority: float = 0.1
    weight_jurisdiction: float = 0.1
    weight_document_form: float = 0.1
    weight_domain: float = 0.1
    weight_year: float = 0.05
    weight_length: float = 0.1


def proxy_keywords(proxy_name: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+", proxy_name.lower())
    return sorted({token for token in tokens if token not in GENERIC_PROXY_WORDS and len(token) > 2})


def mask_proxy_keywords(text: str, keywords: list[str]) -> str:
    return apply_mask_strategy(text, keywords, strategy="keyword_mask")


def _sort_mask_terms(terms: list[str]) -> list[str]:
    unique = {term.strip().lower() for term in terms if str(term).strip()}
    return sorted(unique, key=lambda value: (-len(value), value))


def _morphological_expansions(term: str) -> list[str]:
    cleaned = str(term).strip().lower()
    if not cleaned:
        return []
    expansions = {cleaned}
    if " " not in cleaned:
        expansions.add(f"{cleaned}s")
        expansions.add(f"{cleaned}ed")
        expansions.add(f"{cleaned}ing")
        if cleaned.endswith("y") and len(cleaned) > 3:
            expansions.add(f"{cleaned[:-1]}ies")
        if cleaned.endswith("e") and len(cleaned) > 3:
            expansions.add(f"{cleaned[:-1]}ion")
            expansions.add(f"{cleaned[:-1]}ive")
    return sorted(expansions)


def expanded_proxy_keywords(
    proxy_name: str,
    *,
    display_name: str | None = None,
    extra_terms: list[str] | None = None,
) -> list[str]:
    base_terms = proxy_keywords(proxy_name)
    if display_name:
        base_terms.extend(proxy_keywords(display_name))
    expanded: list[str] = []
    for term in base_terms:
        expanded.extend(_morphological_expansions(term))
        expanded.extend(PROXY_SYNONYM_MAP.get(term, []))
    if extra_terms:
        for term in extra_terms:
            expanded.extend(_morphological_expansions(term))
    return _sort_mask_terms([*base_terms, *expanded, *(extra_terms or [])])


def proxy_mask_terms(
    proxy_name: str,
    *,
    display_name: str | None = None,
    strategy: str = "keyword_mask",
    extra_terms: list[str] | None = None,
) -> list[str]:
    if strategy in {"expanded_keyword_mask", "expanded_char_mask"}:
        return expanded_proxy_keywords(proxy_name, display_name=display_name, extra_terms=extra_terms)
    base_terms = proxy_keywords(proxy_name)
    if display_name:
        base_terms.extend(proxy_keywords(display_name))
    return _sort_mask_terms([*base_terms, *(extra_terms or [])])


def _term_pattern(term: str) -> str:
    pieces = [re.escape(piece) for piece in str(term).split()]
    return r"\b" + r"\s+".join(pieces) + r"\b"


def _char_mask(match: re.Match[str]) -> str:
    text = match.group(0)
    return "".join("#" if char.isalnum() else char for char in text)


def apply_mask_strategy(
    text: str,
    keywords: list[str],
    *,
    strategy: str = "keyword_mask",
    replacement: str = "[MASK]",
) -> str:
    masked = text
    terms = _sort_mask_terms(keywords)
    for keyword in terms:
        pattern = _term_pattern(keyword)
        if strategy in {"char_mask", "expanded_char_mask"}:
            masked = re.sub(pattern, _char_mask, masked, flags=re.IGNORECASE)
        else:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
    return masked


def _normalize_dense(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _encode_sentence_transformer(
    texts: list[str],
    *,
    model_id: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(torch_device)
    model.eval()

    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(torch_device) for k, v in enc.items()}
        with torch.no_grad():
            model_out = model(**enc)
            hidden = model_out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return _normalize_dense(np.concatenate(outputs, axis=0))


def encode_texts(
    texts: list[str],
    *,
    method: str,
    embedding_model: str,
    batch_size: int,
    device: str = "auto",
):
    if method == "sentence":
        return _encode_sentence_transformer(
            texts,
            model_id=embedding_model,
            batch_size=batch_size,
            device=device,
        )
    if method == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        return vectorizer.fit_transform(texts)
    raise ValueError(f"Unsupported encoding method: {method}")


def fit_tfidf_logistic(
    train_texts: list[str],
    labels: list[int],
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    x_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, labels)
    return vectorizer, model


def score_tfidf_logistic(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    texts: list[str],
) -> np.ndarray:
    x_eval = vectorizer.transform(texts)
    return model.predict_proba(x_eval)[:, 1]


def _safe_equal(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    return float(a == b)


def _year_score(year_a: Any, year_b: Any) -> float:
    if year_a is None or year_b is None:
        return 0.0
    try:
        return float(1.0 / (1.0 + abs(int(year_a) - int(year_b))))
    except Exception:
        return 0.0


def _length_score(text_a: str, text_b: str) -> float:
    len_a = max(len(text_a), 1)
    len_b = max(len(text_b), 1)
    return float(1.0 / (1.0 + abs(len_a - len_b) / max(len_a, len_b)))


def _jaccard(values_a: list[str], values_b: list[str]) -> float:
    set_a = {str(v) for v in values_a if str(v)}
    set_b = {str(v) for v in values_b if str(v)}
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b) / len(union))


def pairwise_text_similarity(pos_vectors, cand_vectors) -> np.ndarray:
    return cosine_similarity(pos_vectors, cand_vectors)


def match_score_components(
    positive_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, float]:
    pos_meta = positive_row.get("metadata", {})
    cand_meta = candidate_row.get("metadata", {})
    return {
        "authority": _safe_equal(pos_meta.get("authority"), cand_meta.get("authority")),
        "jurisdiction": _safe_equal(pos_meta.get("jurisdiction"), cand_meta.get("jurisdiction")),
        "document_form": _safe_equal(pos_meta.get("document_form"), cand_meta.get("document_form")),
        "domain": _jaccard(pos_meta.get("application_tags", []), cand_meta.get("application_tags", [])),
        "year": _year_score(pos_meta.get("year"), cand_meta.get("year")),
        "length": _length_score(positive_row.get("text", ""), candidate_row.get("text", "")),
    }


def combined_match_score(
    *,
    text_similarity: float,
    components: dict[str, float],
    config: MatchConfig,
) -> float:
    return float(
        config.weight_text * text_similarity
        + config.weight_authority * components["authority"]
        + config.weight_jurisdiction * components["jurisdiction"]
        + config.weight_document_form * components["document_form"]
        + config.weight_domain * components["domain"]
        + config.weight_year * components["year"]
        + config.weight_length * components["length"]
    )


def greedy_match_rows(
    positives: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    similarity_matrix: np.ndarray,
    config: MatchConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if similarity_matrix.shape != (len(positives), len(candidates)):
        raise ValueError("Similarity matrix shape does not match positives/candidates.")

    available = set(range(len(candidates)))
    matched_rows: list[dict[str, Any]] = []
    component_logs: list[dict[str, float]] = []
    reused = 0

    ordered_indices = sorted(
        range(len(positives)),
        key=lambda idx: len(positives[idx].get("text", "")),
        reverse=True,
    )

    for pos_idx in ordered_indices:
        pos_row = positives[pos_idx]
        candidate_scores: list[tuple[float, int, dict[str, float]]] = []
        candidate_indices = available if available else range(len(candidates))
        if not available and not config.allow_replacement:
            raise ValueError("Ran out of candidate negatives without replacement.")
        if not available:
            reused += 1
        for cand_idx in candidate_indices:
            cand_row = candidates[cand_idx]
            components = match_score_components(pos_row, cand_row)
            score = combined_match_score(
                text_similarity=float(similarity_matrix[pos_idx, cand_idx]),
                components=components,
                config=config,
            )
            candidate_scores.append((score, cand_idx, components))
        candidate_scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_idx, best_components = candidate_scores[0]
        match_row = dict(candidates[best_idx])
        match_row["matched_positive_id"] = pos_row["segment_id"]
        match_row["matching_score"] = best_score
        match_row["matching_components"] = {
            **best_components,
            "text_similarity": float(similarity_matrix[pos_idx, best_idx]),
        }
        matched_rows.append(match_row)
        component_logs.append(match_row["matching_components"])
        if best_idx in available:
            available.remove(best_idx)

    diag = {
        "n_positives": len(positives),
        "n_candidates": len(candidates),
        "n_matched": len(matched_rows),
        "candidate_reuse_events": reused,
        "mean_matching_score": float(np.mean([row["matching_score"] for row in matched_rows])) if matched_rows else 0.0,
        "mean_text_similarity": float(np.mean([log["text_similarity"] for log in component_logs])) if component_logs else 0.0,
        "authority_match_rate": float(np.mean([log["authority"] for log in component_logs])) if component_logs else 0.0,
        "jurisdiction_match_rate": float(np.mean([log["jurisdiction"] for log in component_logs])) if component_logs else 0.0,
        "document_form_match_rate": (
            float(np.mean([log["document_form"] for log in component_logs])) if component_logs else 0.0
        ),
        "domain_overlap_mean": float(np.mean([log["domain"] for log in component_logs])) if component_logs else 0.0,
        "year_score_mean": float(np.mean([log["year"] for log in component_logs])) if component_logs else 0.0,
        "length_score_mean": float(np.mean([log["length"] for log in component_logs])) if component_logs else 0.0,
    }
    return matched_rows, diag


def try_encode_texts_with_fallback(
    texts: list[str],
    *,
    config: MatchConfig,
    device: str = "auto",
):
    if config.method == "sentence":
        try:
            return encode_texts(
                texts,
                method="sentence",
                embedding_model=config.embedding_model,
                batch_size=config.batch_size,
                device=device,
            ), "sentence"
        except Exception as exc:
            logging.warning("Sentence embedding load failed, falling back to TF IDF: %s", exc)
            return encode_texts(
                texts,
                method="tfidf",
                embedding_model=config.embedding_model,
                batch_size=config.batch_size,
                device=device,
            ), "tfidf"
    return encode_texts(
        texts,
        method=config.method,
        embedding_model=config.embedding_model,
        batch_size=config.batch_size,
        device=device,
    ), config.method


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def masked_texts(
    texts: list[str],
    keywords: list[str],
    *,
    strategy: str = "keyword_mask",
    replacement: str = "[MASK]",
) -> list[str]:
    return [
        apply_mask_strategy(text, keywords, strategy=strategy, replacement=replacement)
        for text in texts
    ]


def logit_margin(correct_scores: np.ndarray, distractor_scores: np.ndarray) -> np.ndarray:
    correct = np.asarray(correct_scores, dtype=np.float64)
    distractor = np.asarray(distractor_scores, dtype=np.float64)
    return correct - distractor


def bounded_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if obs.size == 0 or obs.shape != pred.shape:
        return float("nan")
    denom = float(((obs - obs.mean()) ** 2).sum())
    if math.isclose(denom, 0.0):
        return float("nan")
    resid = float(((obs - pred) ** 2).sum())
    return float(1.0 - resid / denom)
