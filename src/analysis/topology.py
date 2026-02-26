"""Topology metrics for circuit attribution graphs."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np


def _build_adjacency(
    nodes: list[str],
    edges: list[dict],
) -> dict[str, list[str]]:
    """Build an adjacency list from edge dicts with 'src' and 'dst' keys."""
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e["src"]].append(e["dst"])
    return dict(adj)


def graph_depth(
    nodes: list[str],
    adj: dict[str, list[str]],
) -> int:
    """Longest directed path length in the graph (via iterative DFS)."""
    if not nodes:
        return 0

    # Find source nodes (nodes with no incoming edges).
    has_incoming = set()
    for dsts in adj.values():
        for d in dsts:
            has_incoming.add(d)
    sources = [n for n in nodes if n not in has_incoming]
    if not sources:
        sources = nodes[:1]  # Fallback: pick an arbitrary node.

    max_depth = 0
    for start in sources:
        # Iterative DFS with (node, depth) stack.
        stack = [(start, 0)]
        visited: set[str] = set()
        while stack:
            node, depth = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            max_depth = max(max_depth, depth)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    stack.append((neighbor, depth + 1))
    return max_depth


def mean_branching_factor(
    nodes: list[str],
    adj: dict[str, list[str]],
) -> float:
    """Average out-degree across all nodes."""
    if not nodes:
        return 0.0
    degrees = [len(adj.get(n, [])) for n in nodes]
    return float(np.mean(degrees))


def graph_density(
    n_nodes: int,
    n_edges: int,
) -> float:
    """Edge density: actual / possible directed edges."""
    if n_nodes < 2:
        return 0.0
    possible = n_nodes * (n_nodes - 1)
    return n_edges / possible





def compute_topology_metrics(graph_dict: dict) -> dict[str, Any]:
    """Compute all topology metrics from a CircuitGraph dict.

    Args:
        graph_dict: Output of CircuitGraph.to_dict().

    Returns:
        Dictionary of topology metrics.
    """
    node_ids = list(graph_dict.get("nodes", {}).keys())
    edges = graph_dict.get("edges", [])

    adj = _build_adjacency(node_ids, edges)

    depth = graph_depth(node_ids, adj)
    branching = mean_branching_factor(node_ids, adj)
    density = graph_density(len(node_ids), len(edges))

    return {
        "n_nodes": len(node_ids),
        "n_edges": len(edges),
        "depth": depth,
        "branching_factor": branching,
        "density": density,
    }


def compare_topologies(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    label_a: str = "safe",
    label_b: str = "innov",
) -> dict[str, Any]:
    """Compare two topology metric dicts, computing deltas."""
    comparison = {}
    numeric_keys = ["n_nodes", "n_edges", "depth", "branching_factor", "density"]
    for key in numeric_keys:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if va is not None and vb is not None:
            comparison[key] = {
                label_a: va,
                label_b: vb,
                "delta": va - vb,
            }
        else:
            comparison[key] = {label_a: va, label_b: vb, "delta": None}
    return comparison
