from policy_interp.masking import mask_text


def test_mask_text_supports_multi_word_anchor_patterns() -> None:
    text = "The training data must be documented, and training data quality must be monitored."
    masked = mask_text(text, ["training data"], "[MASK]")

    assert masked.count("[MASK]") == 2
    assert "training data" not in masked.lower()
