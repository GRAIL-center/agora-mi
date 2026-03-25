from __future__ import annotations

import argparse

from _common import setup_logging
from benchmark.policy_feature_benchmark import run_benchmark, run_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_feature_benchmark.yaml")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()

    setup_logging("run_policy_feature_benchmark")
    if args.preflight_only:
        run_preflight(args.config, output_root=args.output_root)
    else:
        run_benchmark(args.config, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
