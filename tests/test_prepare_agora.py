from pathlib import Path

import pandas as pd

from policy_interp.agora import assign_document_grouped_splits
from policy_interp.schemas import DatasetConfig, ExperimentConfig, SplitsConfig


def test_document_grouped_split_keeps_docs_together(tmp_path: Path) -> None:
    segments = pd.DataFrame(
        [
            {"document_id": 1, "validated_document": False, "bias": True, "privacy": False},
            {"document_id": 1, "validated_document": False, "bias": False, "privacy": False},
            {"document_id": 2, "validated_document": False, "bias": False, "privacy": True},
            {"document_id": 3, "validated_document": True, "bias": False, "privacy": True},
        ]
    )
    for column in ["discrimination", "rights_violation", "transparency", "interpretability"]:
        segments[column] = False

    config = ExperimentConfig(
        name="unit_test",
        dataset=DatasetConfig(base_dir=tmp_path),
        splits=SplitsConfig(train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=17),
    )
    manifest = assign_document_grouped_splits(segments, config)
    assert manifest["document_id"].is_unique
    validated_split = manifest.loc[manifest["document_id"] == 3, "split"].item()
    assert validated_split == "validated"
