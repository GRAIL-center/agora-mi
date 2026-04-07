import pandas as pd

from policy_interp.pilot import _allocate_split_targets, _balanced_proxy_sample


def test_allocate_split_targets_preserves_total() -> None:
    targets = _allocate_split_targets(
        total=200,
        split_names=["train", "dev", "test"],
        split_weights={"train": 0.68, "dev": 0.16, "test": 0.16},
    )
    assert sum(targets.values()) == 200
    assert targets["train"] > targets["dev"]
    assert targets["train"] > targets["test"]


def test_balanced_proxy_sample_covers_multiple_proxies() -> None:
    frame = pd.DataFrame(
        [
            {"segment_id": "a", "split": "dev", "privacy": True, "bias": False, "proxy_count": 1, "document_id": 1, "Segment position": 1, "source_type": "positive"},
            {"segment_id": "b", "split": "dev", "privacy": False, "bias": True, "proxy_count": 1, "document_id": 2, "Segment position": 1, "source_type": "positive"},
            {"segment_id": "c", "split": "dev", "privacy": True, "bias": False, "proxy_count": 1, "document_id": 3, "Segment position": 1, "source_type": "positive"},
        ]
    )
    sampled = _balanced_proxy_sample(frame, take=2, proxy_keys=["privacy", "bias"])
    assert sampled["segment_id"].nunique() == 2
    assert sampled["privacy"].any()
    assert sampled["bias"].any()
