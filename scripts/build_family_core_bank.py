from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import ensure_dir, read_config, save_with_metadata, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy_mech_interp.yaml")
    parser.add_argument("--family", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", default="resid_post")
    parser.add_argument("--min_proxy_support", type=int, default=2)
    parser.add_argument("--min_bootstrap_stability", type=float, default=0.25)
    args = parser.parse_args()

    setup_logging("build_family_core_bank")
    cfg = read_config(args.config)
    discovery_root = (
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "policy_discovery"
        / args.family
    )
    if not discovery_root.exists():
        raise FileNotFoundError(f"Missing discovery root: {discovery_root}")

    support_map: dict[int, list[dict]] = defaultdict(list)
    for proxy_dir in sorted(path for path in discovery_root.iterdir() if path.is_dir()):
        run_dir = proxy_dir / f"layer{args.layer}_{args.site}"
        bank_path = run_dir / "feature_bank.json"
        if not bank_path.exists():
            continue
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        for feature_id in bank.get("feature_ids", []):
            support_map[int(feature_id)].append(
                {
                    "proxy_slug": proxy_dir.name,
                    "proxy_name": bank.get("proxy_name"),
                    "bootstrap_stability": float(bank.get("bootstrap_stability", {}).get(str(feature_id), 0.0)),
                    "weight": float(
                        bank.get("feature_weights", bank.get("train_top_deltas", []))[bank["feature_ids"].index(feature_id)]
                    ),
                }
            )

    core_rows = []
    for feature_id, supports in sorted(support_map.items()):
        proxy_support = len(supports)
        mean_stability = sum(item["bootstrap_stability"] for item in supports) / max(proxy_support, 1)
        if proxy_support < args.min_proxy_support:
            continue
        if mean_stability < args.min_bootstrap_stability:
            continue
        core_rows.append(
            {
                "feature_id": feature_id,
                "proxy_support": proxy_support,
                "mean_bootstrap_stability": mean_stability,
                "supports": supports,
            }
        )

    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results/policy_mech_interp"))
        / "family_core"
        / args.family
        / f"layer{args.layer}_{args.site}"
    )
    save_with_metadata(
        output_path=out_dir / "family_core_bank.json",
        payload={
            "family_name": args.family,
            "layer": args.layer,
            "site": args.site,
            "min_proxy_support": args.min_proxy_support,
            "min_bootstrap_stability": args.min_bootstrap_stability,
            "core_features": core_rows,
        },
        config={"run": cfg, "args": vars(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
