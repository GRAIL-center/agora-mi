from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DocSplit:
    train: list[str]
    dev: list[str]
    test: list[str]


def split_doc_ids(
    doc_ids: list[str],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> DocSplit:
    total = train_ratio + dev_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train/dev/test ratios must sum to 1.0, got {train_ratio}+{dev_ratio}+{test_ratio}={total}"
        )

    ids = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = ids[:n_train]
    dev = ids[n_train : n_train + n_dev]
    test = ids[n_train + n_dev :]
    return DocSplit(train=train, dev=dev, test=test)


def assert_no_overlap(split: DocSplit) -> None:
    train_set = set(split.train)
    dev_set = set(split.dev)
    test_set = set(split.test)
    if train_set & dev_set or train_set & test_set or dev_set & test_set:
        raise AssertionError("Document split overlap detected.")
