from __future__ import annotations

import argparse

from _common import setup_logging
from benchmark.policy_feature_benchmark import aggregate_existing_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_feature_benchmark.yaml")
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    setup_logging("aggregate_policy_feature_benchmark")
    aggregate_existing_results(args.config, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
