"""Circuit graph data structures for aggregation, pruning, and serialization."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CircuitNode:
    """A node in a circuit graph (a transcoder / SAE feature)."""

    node_id: str  # e.g. "L12_F4773" or "input_tok_5"
    layer: int | None = None
    feature_idx: int | None = None
    node_type: str = "feature"  # "feature", "input", "output", "error"
    label: str | None = None


@dataclass
class CircuitEdge:
    """A directed edge with an attribution score."""

    src: str
    dst: str
    weight: float = 0.0


@dataclass
class CircuitGraph:
    """Directed attribution graph."""

    nodes: dict[str, CircuitNode] = field(default_factory=dict)
    edges: list[CircuitEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- construction helpers --------------------------------------------------

    def add_node(self, node: CircuitNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, src: str, dst: str, weight: float) -> None:
        self.edges.append(CircuitEdge(src=src, dst=dst, weight=weight))

    # -- pruning ---------------------------------------------------------------

    def prune(self, edge_tau: float) -> "CircuitGraph":
        """Return a new graph with edges below *edge_tau* removed."""
        kept_edges = [e for e in self.edges if abs(e.weight) >= edge_tau]
        # Keep only nodes that participate in at least one surviving edge.
        active_ids = {e.src for e in kept_edges} | {e.dst for e in kept_edges}
        kept_nodes = {nid: n for nid, n in self.nodes.items() if nid in active_ids}
        g = CircuitGraph(nodes=kept_nodes, edges=kept_edges, metadata=dict(self.metadata))
        g.metadata["edge_tau"] = edge_tau
        g.metadata["n_edges_before_prune"] = len(self.edges)
        return g

    # -- aggregation -----------------------------------------------------------

    @classmethod
    def aggregate(cls, graphs: list["CircuitGraph"], edge_tau: float = 0.0) -> "CircuitGraph":
        """Merge multiple per-prompt graphs into a consensus graph.

        Edge weights are averaged across graphs where the edge appears.
        Edges that appear in fewer than 20 % of graphs are dropped.
        """
        edge_weights: dict[tuple[str, str], list[float]] = defaultdict(list)
        all_nodes: dict[str, CircuitNode] = {}
        n = len(graphs)

        for g in graphs:
            all_nodes.update(g.nodes)
            for e in g.edges:
                edge_weights[(e.src, e.dst)].append(e.weight)

        min_count = max(1, int(0.2 * n))
        merged = cls(nodes=all_nodes, metadata={"n_source_graphs": n})
        for (src, dst), weights in edge_weights.items():
            if len(weights) >= min_count:
                avg_w = sum(weights) / len(weights)
                if abs(avg_w) >= edge_tau:
                    merged.add_edge(src, dst, avg_w)

        return merged

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CircuitGraph":
        nodes = {
            nid: CircuitNode(**ndata) for nid, ndata in d.get("nodes", {}).items()
        }
        edges = [CircuitEdge(**edata) for edata in d.get("edges", [])]
        return cls(nodes=nodes, edges=edges, metadata=d.get("metadata", {}))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "CircuitGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # -- summary ---------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "node_types": dict(
                defaultdict(
                    int,
                    **{
                        t: sum(1 for n in self.nodes.values() if n.node_type == t)
                        for t in {n.node_type for n in self.nodes.values()}
                    },
                )
            ),
        }
