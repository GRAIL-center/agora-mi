"""Shared constants for the Policy Interp pipeline."""

from __future__ import annotations

PROXY_COLUMNS = {
    "bias": "Risk factors: Bias",
    "discrimination": "Harms: Discrimination",
    "privacy": "Risk factors: Privacy",
    "rights_violation": "Harms: Violation of civil or human rights, including privacy",
    "transparency": "Risk factors: Transparency",
    "interpretability": "Risk factors: Interpretability and explainability",
}

PAIR_MAP = {
    "bias": "discrimination",
    "discrimination": "bias",
    "privacy": "rights_violation",
    "rights_violation": "privacy",
    "transparency": "interpretability",
    "interpretability": "transparency",
}

APPLICATION_PREFIX = "Applications:"
BOOLEAN_TRUE_VALUES = {"true", "1", "yes", "y", "t"}
DEFAULT_SEED = 17
MODULE_STATUS_STABLE = "stable"
MODULE_STATUS_UNSTABLE = "unstable"
