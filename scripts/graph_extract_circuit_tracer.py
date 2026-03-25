"""Extract circuit attribution graphs using the circuit-tracer library.

Usage:
    python scripts/graph_extract_circuit_tracer.py \
        --config configs/run.yaml --layer 24 \
        --safe_jsonl data/processed/dsafe_train.jsonl \
        --innov_jsonl data/processed/dinnov_train.jsonl \
        --max_prompts 20
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from _common import ensure_dir, read_config, save_with_metadata, setup_logging

# Lazy-import circuit_tracer so the script can still be --help'd without it.
_CT_AVAILABLE = True
try:
    from circuit_tracer import ReplacementModel, attribute, Graph
    from circuit_tracer.graph import prune_graph
except ImportError:
    _CT_AVAILABLE = False


def _graph_to_circuit_graph(
    ct_graph: "Graph",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> "CircuitGraph":
    """Convert a circuit-tracer Graph into our CircuitGraph format.

    Uses prune_graph() to get node/edge masks, then extracts the pruned
    sub-graph into our format.
    """
    from analysis.circuit_graph import CircuitNode, CircuitGraph

    # Prune the graph using circuit-tracer's built-in pruning.
    pr = prune_graph(ct_graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    node_mask = pr.node_mask  # bool tensor [n_nodes]
    edge_mask = pr.edge_mask  # bool tensor [n_nodes, n_nodes]

    adj = ct_graph.adjacency_matrix  # [n_nodes, n_nodes]
    feats = ct_graph.selected_features  # [n_nodes] (index into active_features)

    cg = CircuitGraph(metadata={
        "input_string": ct_graph.input_string if hasattr(ct_graph, "input_string") else "",
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
    })

    n_nodes = feats.shape[0]
    node_ids = []
    
    # active_features shape: [total_active_features, 3] -> (layer, pos, feature_idx)
    active_feats = ct_graph.active_features 

    for i in range(n_nodes):
        if not node_mask[i]:
            node_ids.append(None)
            continue
            
        # feats[i] is the index into active_feats
        active_idx = int(feats[i].item())
        feature_info = active_feats[active_idx]
        layer = int(feature_info[0].item())
        feat_idx = int(feature_info[2].item())
        
        node_id = f"L{layer}_F{feat_idx}"
        node_ids.append(node_id)
        cg.add_node(CircuitNode(
            node_id=node_id,
            layer=layer,
            feature_idx=feat_idx,
            node_type="feature",
        ))

    # Extract edges from the pruned adjacency matrix.
    pruned_adj = adj * edge_mask.float()
    src_indices, dst_indices = torch.nonzero(pruned_adj, as_tuple=True)
    for s, d in zip(src_indices.tolist(), dst_indices.tolist()):
        if s >= len(node_ids) or d >= len(node_ids):
            continue  # Skip edges to/from non-feature nodes (e.g., logit nodes)
        src_id = node_ids[s]
        dst_id = node_ids[d]
        if src_id is not None and dst_id is not None:
            weight = pruned_adj[s, d].item()
            cg.add_edge(src_id, dst_id, weight)

    return cg


def _extract_graphs_for_prompts(
    model: "ReplacementModel",
    prompts: list[str],
    max_prompts: int,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    max_feature_nodes: int | None = 512,
) -> list:
    """Run circuit-tracer attribution on each prompt."""
    from analysis.circuit_graph import CircuitGraph

    graphs = []
    n = min(len(prompts), max_prompts)
    logging.info("Extracting attribution graphs for %d prompts...", n)

    for i, prompt in enumerate(prompts[:n]):
        logging.info("  [%d/%d] prompt: %.60s...", i + 1, n, prompt)
        try:
            # circuit-tracer API: attribute(prompt, model, ...)
            # Use offload="cpu" and small batch_size to reduce VRAM usage.
            ct_graph = attribute(
                prompt, model,
                verbose=True,
                max_feature_nodes=max_feature_nodes,
                batch_size=4,
                offload="disk",
            )
            cg = _graph_to_circuit_graph(
                ct_graph,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            graphs.append(cg)
            logging.info("    -> %d nodes, %d edges", len(cg.nodes), len(cg.edges))

        except Exception as e:
            logging.warning("    -> attribution failed: %s", e, exc_info=True)
            continue
        finally:
            # Free VRAM between prompts.
            torch.cuda.empty_cache()

    return graphs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract circuit attribution graphs via circuit-tracer."
    )
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--safe_jsonl", default="data/processed/dsafe_train.jsonl")
    parser.add_argument("--innov_jsonl", default="data/processed/dinnov_train.jsonl")
    parser.add_argument("--max_prompts", type=int, default=20,
                        help="Max prompts per category for graph extraction.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Starting index offset for prompt batching.")
    parser.add_argument("--transcoder_set", default="mntss/gemma-scope-transcoders",
                        help="HuggingFace repo or config for transcoders.")
    parser.add_argument("--node_threshold", type=float, default=0.8,
                        help="Node pruning threshold (fraction of influence to keep).")
    parser.add_argument("--edge_threshold", type=float, default=0.98,
                        help="Edge pruning threshold (fraction of influence to keep).")
    parser.add_argument("--max_feature_nodes", type=int, default=None,
                        help="Max feature nodes in attribution graph.")
    args = parser.parse_args()

    setup_logging("graph_extract_circuit_tracer")
    cfg = read_config(args.config)

    if not _CT_AVAILABLE:
        logging.error(
            "circuit-tracer is not installed. "
            "Install via: git clone https://github.com/decoderesearch/circuit-tracer "
            "&& cd circuit-tracer && pip install ."
        )
        return 1

    # Load data.
    from data.io import read_jsonl
    from model.prompt import build_prompts, load_prompt_config, render_prompt

    prompt_cfg = load_prompt_config(cfg.get("prompt_config", "configs/prompt_templates.yaml"))
    template = str(prompt_cfg["template_v1"])

    safe_rows = read_jsonl(args.safe_jsonl)
    innov_rows = read_jsonl(args.innov_jsonl)

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    rng.shuffle(safe_rows)
    rng.shuffle(innov_rows)
    safe_rows = safe_rows[args.offset:]
    innov_rows = innov_rows[args.offset:]

    model_id = cfg.get("model_id", "google/gemma-2-2b")

    # Load circuit-tracer ReplacementModel with transcoders.
    logging.info("Loading ReplacementModel (%s) with transcoders (%s)...",
                 model_id, args.transcoder_set)
    model = ReplacementModel.from_pretrained(
        model_name=model_id,
        transcoder_set=args.transcoder_set,
        backend="transformerlens",
    )
    logging.info("Model loaded successfully.")

    use_chat_template = cfg.get("use_chat_template", False)
    safe_prompts = build_prompts(safe_rows, template, tokenizer=model.tokenizer, use_chat_template=use_chat_template)
    innov_prompts = build_prompts(innov_rows, template, tokenizer=model.tokenizer, use_chat_template=use_chat_template)

    # Extract graphs for each category.
    logging.info("--- Extracting SAFE circuit graphs ---")
    safe_graphs = _extract_graphs_for_prompts(
        model, safe_prompts, args.max_prompts,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        max_feature_nodes=args.max_feature_nodes,
    )

    logging.info("--- Extracting INNOV circuit graphs ---")
    innov_graphs = _extract_graphs_for_prompts(
        model, innov_prompts, args.max_prompts,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        max_feature_nodes=args.max_feature_nodes,
    )

    # Aggregate per-category.
    from analysis.circuit_graph import CircuitGraph

    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "graphs")

    if safe_graphs:
        safe_consensus = CircuitGraph.aggregate(safe_graphs, edge_tau=0.0)
        safe_path = out_dir / f"circuit_safe_layer{args.layer}_off{args.offset}.json"
        safe_consensus.save_json(safe_path)
        logging.info("Saved safe circuit graph: %s (%s)", safe_path, safe_consensus.summary())
    else:
        logging.warning("No safe graphs extracted.")

    if innov_graphs:
        innov_consensus = CircuitGraph.aggregate(innov_graphs, edge_tau=0.0)
        innov_path = out_dir / f"circuit_innov_layer{args.layer}_off{args.offset}.json"
        innov_consensus.save_json(innov_path)
        logging.info("Saved innov circuit graph: %s (%s)", innov_path, innov_consensus.summary())
    else:
        logging.warning("No innov graphs extracted.")

    # Save metadata.
    save_with_metadata(
        output_path=out_dir / f"extraction_meta_layer{args.layer}_off{args.offset}.json",
        payload={
            "layer": args.layer,
            "n_safe_prompts": len(safe_graphs),
            "n_innov_prompts": len(innov_graphs),
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "transcoder_set": args.transcoder_set,
            "safe_graph_summary": safe_graphs[0].summary() if safe_graphs else None,
            "innov_graph_summary": innov_graphs[0].summary() if innov_graphs else None,
        },
        config={"run": cfg, "args": vars(args)},
    )

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
