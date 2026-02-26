"""
Run circuit graph extraction in batches to prevent CUDA OOM fragmentation.

It repeatedly calls `graph_extract_circuit_tracer.py` with an --offset and
--max_prompts (which acts as batch size), allowing the OS to completely 
reclaim GPU memory between runs. Finally, it merges the partial graphs.
"""

import argparse
import subprocess
import logging
import sys
from pathlib import Path

from analysis.circuit_graph import CircuitGraph
from _common import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Batch runner for circuit extraction.")
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--total_prompts", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)
    parser.add_argument("--max_feature_nodes", type=int, default=128)
    args = parser.parse_args()

    setup_logging("run_extract_batches")
    out_dir = Path("results/graphs")

    offsets = list(range(0, args.total_prompts, args.batch_size))
    logging.info(f"Running {args.total_prompts} prompts in {len(offsets)} batches of {args.batch_size}.")

    for offset in offsets:
        logging.info(f"=== Starting batch offset {offset} ===")
        cmd = [
            sys.executable, "scripts/graph_extract_circuit_tracer.py",
            "--config", args.config,
            "--layer", str(args.layer),
            "--offset", str(offset),
            "--max_prompts", str(args.batch_size),
            "--node_threshold", str(args.node_threshold),
            "--edge_threshold", str(args.edge_threshold),
            "--max_feature_nodes", str(args.max_feature_nodes),
        ]
        
        logging.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logging.error(f"Batch {offset} failed with exit code {result.returncode}. Stopping.")
            return result.returncode

    logging.info("All batches completed successfully. Merging partial graphs...")

    for category in ["safe", "innov"]:
        partial_graphs = []
        for offset in offsets:
            path = out_dir / f"circuit_{category}_layer{args.layer}_off{offset}.json"
            if path.exists():
                logging.info(f"Loading partial graph {path}")
                partial_graphs.append(CircuitGraph.load_json(path))
            else:
                logging.warning(f"Expected partial graph missing: {path}")

        if not partial_graphs:
            logging.warning(f"No partial graphs found for {category}")
            continue

        logging.info(f"Aggregating {len(partial_graphs)} {category} graphs...")
        # Since these are already individual consensus graphs from each batch,
        # we can aggregate them further. We use edge_tau=0.0 because pruning 
        # is already done or will be done properly.
        final_graph = CircuitGraph.aggregate(partial_graphs, edge_tau=0.0)
        
        final_path = out_dir / f"circuit_{category}_layer{args.layer}.json"
        final_graph.save_json(final_path)
        logging.info(f"Saved final merged {category} graph to {final_path} ({final_graph.summary()})")

    logging.info("Batch extraction complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
