from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file

from policy_interp.feature_matrix import load_residual_matrix


def test_load_residual_matrix_casts_bfloat16_to_float32(tmp_path: Path) -> None:
    tensor_path = tmp_path / "residual.safetensors"
    save_file({"residual_pooled": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)}, tensor_path)

    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "segment_id": "s0",
                "split": "test",
                "layer": 24,
                "row_index": 0,
                "tensor_path": str(tensor_path),
            },
            {
                "segment_id": "s1",
                "split": "test",
                "layer": 24,
                "row_index": 1,
                "tensor_path": str(tensor_path),
            },
        ]
    ).to_parquet(manifest_path, index=False)

    frame = load_residual_matrix(manifest_path)

    assert frame["vector"].iloc[0].dtype.name == "float32"
    assert frame["vector"].iloc[0].tolist() == [1.0, 2.0]
    assert frame["vector"].iloc[1].tolist() == [3.0, 4.0]
