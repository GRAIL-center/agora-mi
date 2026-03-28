from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assistant.experiments import run_policy_analysis_experiments
from runtime import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the policy analysis assistant experiment suite.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "policy_analysis_assistant.yaml"),
        help="Path to the policy analysis assistant config.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional override for the experiment output root.",
    )
    args = parser.parse_args()

    setup_logging("run_policy_analysis_experiments")
    run_policy_analysis_experiments(args.config, output_root=args.output_root)


if __name__ == "__main__":
    main()
