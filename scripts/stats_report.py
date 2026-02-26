from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import ensure_dir, read_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tag", default=None, help="Optional tag suffix, e.g. train")
    args = parser.parse_args()

    setup_logging("stats_report")
    cfg = read_config(args.config)
    results_dir = Path(cfg.get("results_dir", "results"))

    suffix = f"_{args.tag}" if args.tag else ""
    pol_csv = results_dir / "polarization" / f"layer{args.layer}{suffix}_delta.csv"
    # Fallback: try without suffix if tagged version does not exist.
    if not pol_csv.exists() and suffix:
        pol_csv = results_dir / "polarization" / f"layer{args.layer}_delta.csv"
    sanity_json = results_dir / "sanity" / f"layer{args.layer}_sanity.json"
    ablation_json = results_dir / "pilot_ablation" / "effect_summary.json"

    rows = []
    if pol_csv.exists():
        pol = pd.read_csv(pol_csv)
        rows.append(
            {
                "metric": "polarization_top_delta",
                "value": float(pol["delta"].iloc[0]),
                "note": f"feature_id={int(pol['feature_id'].iloc[0])}",
            }
        )
        if "fdr_reject" in pol.columns:
            n_survivors = int(pol["fdr_reject"].sum())
            rows.append(
                {
                    "metric": "fdr_survivors",
                    "value": n_survivors,
                    "note": f"{n_survivors}/{len(pol)} features (q<0.05)",
                }
            )
    if sanity_json.exists():
        sj = json.loads(sanity_json.read_text(encoding="utf-8"))
        auc = sj.get("sanity", {}).get("quick_auc")
        rows.append({"metric": "quick_auc", "value": auc, "note": "sanity check"})
    if ablation_json.exists():
        aj = json.loads(ablation_json.read_text(encoding="utf-8"))
        es = aj.get("effect_summary", {})
        rows.append(
            {
                "metric": "pilot_mean_effect",
                "value": es.get("mean_effect"),
                "note": f"ci={es.get('bootstrap_95_ci')}",
            }
        )
        rows.append(
            {
                "metric": "pilot_random_control_mean_effect",
                "value": es.get("random_control_mean_effect"),
                "note": f"ci={es.get('random_control_bootstrap_95_ci')}",
            }
        )
        if "paired_delta_mean" in es:
            rows.append(
                {
                    "metric": "pilot_paired_delta_mean",
                    "value": es.get("paired_delta_mean"),
                    "note": (
                        f"ci={es.get('paired_delta_bootstrap_95_ci')}; "
                        f"perm_p={es.get('paired_delta_permutation_p')}"
                    ),
                }
            )

    out_dir = ensure_dir(results_dir / "stats")
    table_path = out_dir / "tables.csv"
    md_path = out_dir / "summary.md"
    table = pd.DataFrame(rows)
    table.to_csv(table_path, index=False)

    lines = [
        "# Stats Summary",
        "",
        f"- Layer: {args.layer}",
        "",
    ]
    for _, r in table.iterrows():
        lines.append(f"- {r['metric']}: {r['value']} ({r['note']})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
