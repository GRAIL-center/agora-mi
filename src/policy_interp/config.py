"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from policy_interp.schemas import ExperimentConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(payload)
