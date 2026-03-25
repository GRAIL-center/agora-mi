"""Compute topology metrics on circuit attribution graphs.

Usage:
    python scripts/graph_metrics.py --config configs/run.yaml --layer 24
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from _common import ensure_dir, read_config, save_with_metadata, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute topology metrics for circuit graphs."
    )
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--in_safe_json", default=None,
                        help="Path to safe circuit graph JSON. Auto-detected if omitted.")
    parser.add_argument("--in_innov_json", default=None,
                        help="Path to innov circuit graph JSON. Auto-detected if omitted.")
    args = parser.parse_args()

    setup_logging("graph_metrics")
    cfg = read_config(args.config)
    results_dir = Path(cfg.get("results_dir", "results"))
    graphs_dir = results_dir / "graphs"

    safe_path = Path(args.in_safe_json) if args.in_safe_json else graphs_dir / f"circuit_safe_layer{args.layer}.json"
    innov_path = Path(args.in_innov_json) if args.in_innov_json else graphs_dir / f"circuit_innov_layer{args.layer}.json"

    if not safe_path.exists():
        logging.error("Safe circuit graph not found: %s", safe_path)
        return 1
    if not innov_path.exists():
        logging.error("Innov circuit graph not found: %s", innov_path)
        return 1

    from analysis.circuit_graph import CircuitGraph
    from analysis.topology import compute_topology_metrics, compare_topologies

    safe_graph = CircuitGraph.load_json(safe_path)
    innov_graph = CircuitGraph.load_json(innov_path)

    logging.info("Safe graph: %s", safe_graph.summary())
    logging.info("Innov graph: %s", innov_graph.summary())

    safe_metrics = compute_topology_metrics(safe_graph.to_dict())
    innov_metrics = compute_topology_metrics(innov_graph.to_dict())
    comparison = compare_topologies(safe_metrics, innov_metrics)

    out_dir = ensure_dir(graphs_dir)
    out_path = out_dir / f"topology_layer{args.layer}.json"

    payload = {
        "layer": args.layer,
        "safe": safe_metrics,
        "innov": innov_metrics,
        "comparison": comparison,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote topology metrics: %s", out_path)

    # Print comparison summary.
    print(f"\n=== Topology Comparison (Layer {args.layer}) ===")
    for key, vals in comparison.items():
        if vals.get("delta") is not None:
            if key in ["n_nodes", "n_edges"]:
                print(f"  {key}: safe={int(vals['safe'])}  innov={int(vals['innov'])}  delta={int(vals['delta'])}")
            else:
                print(f"  {key}: safe={vals['safe']:.4f}  innov={vals['innov']:.4f}  delta={vals['delta']:.4f}")
        else:
            print(f"  {key}: safe={vals.get('safe')}  innov={vals.get('innov')}  delta=N/A")

    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
