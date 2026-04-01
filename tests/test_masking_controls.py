from __future__ import annotations

from data.matching import apply_mask_strategy, proxy_mask_terms


def test_expanded_proxy_mask_terms_include_proxy_synonyms():
    terms = proxy_mask_terms(
        "Risk factors: Privacy",
        display_name="Privacy",
        strategy="expanded_keyword_mask",
    )
    assert "privacy" in terms
    assert "personal data" in terms
    assert "data protection" in terms


def test_keyword_mask_replaces_whole_terms():
    text = "The policy imposes privacy safeguards and data protection duties."
    masked = apply_mask_strategy(
        text,
        ["privacy", "data protection"],
        strategy="keyword_mask",
    )
    assert "[MASK]" in masked
    assert "privacy" not in masked.lower()
    assert "data protection" not in masked.lower()


def test_char_mask_preserves_span_shape():
    text = "The policy imposes privacy safeguards."
    masked = apply_mask_strategy(
        text,
        ["privacy"],
        strategy="char_mask",
    )
    assert "privacy" not in masked.lower()
    assert "#######" in masked
    assert masked.endswith("safeguards.")
