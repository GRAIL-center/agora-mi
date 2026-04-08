from pathlib import Path

import pandas as pd

from policy_interp.baselines import _read_stable_modules


def test_read_stable_modules_returns_empty_schema_when_discovery_is_missing(tmp_path: Path) -> None:
    stable_modules = _read_stable_modules(tmp_path / "module_stability.parquet")

    assert stable_modules.empty
    assert stable_modules.columns.tolist() == [
        "stable_module_id",
        "layer",
        "stable",
        "feature_ids",
        "module_size",
    ]


def test_read_stable_modules_reads_existing_parquet(tmp_path: Path) -> None:
    path = tmp_path / "module_stability.parquet"
    expected = pd.DataFrame(
        [
            {
                "stable_module_id": "layer_24_stable_001",
                "layer": 24,
                "stable": True,
                "feature_ids": [1, 2, 3],
                "module_size": 3,
            }
        ]
    )
    expected.to_parquet(path, index=False)

    stable_modules = _read_stable_modules(path)

    row = stable_modules.iloc[0]
    assert row["stable_module_id"] == "layer_24_stable_001"
    assert int(row["layer"]) == 24
    assert bool(row["stable"]) is True
    assert list(row["feature_ids"]) == [1, 2, 3]
    assert int(row["module_size"]) == 3
