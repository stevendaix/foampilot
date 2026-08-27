"""Topology graph for medical_build centerlines and reconstruction.

The graph is deliberately separate from geometry: nodes represent cap/junction
locations and edges represent centerline branches.  It is used for topology
validation, branch ordering, and volume partitioning; it is not a replacement
for the geometric STL union.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import json
import math

import numpy as np

from .analysis_data import BranchRecord, GeometryAnalysisData

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - exercised in minimal installs
    raise ImportError(
        "vascular_graph requires NetworkX. Install it with `pip install networkx`."
    ) from exc


@dataclass(frozen=True)
class GraphValidation:
    connected: bool
    acyclic: bool
    branch_count: int
    node_count: int
    component_count: int
    roots: tuple[Any, ...]
    terminals: tuple[Any, ...]
    bifurcations: tuple[Any, ...]
    isolated_nodes: tuple[Any, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "acyclic": self.acyclic,
            "branch_count": self.branch_count,
            "node_count": self.node_count,
            "component_count": self.component_count,
            "roots": [str(x) for x in self.roots],
            "terminals": [str(x) for x in self.terminals],
            "bifurcations": [str(x) for x in self.bifurcations],
            "isolated_nodes": [str(x) for x in self.isolated_nodes],
        }


class VascularGraph:
    """NetworkX representation of the branch topology.

    A branch is an edge from ``source_cap_id`` to ``target_cap_id``.  Spatial
    endpoint clustering can optionally replace unstable cap identifiers when
    importing data whose branch IDs are local to each branch.
    """

    def __init__(self, graph: nx.Graph, branch_to_edge: Mapping[int, tuple[Any, Any]]):
        self.graph = graph
        self.branch_to_edge = dict(branch_to_edge)

    @classmethod
    def from_analysis(
        cls,
        data: GeometryAnalysisData,
        endpoint_tolerance: Optional[float] = None,
    ) -> "VascularGraph":
        data.validate()
        graph = nx.Graph()
        branch_to_edge: Dict[int, tuple[Any, Any]] = {}
        branches = list(data.branches)
        endpoint_records = []
        for branch in branches:
            source = ("cap", branch.source_cap_id)
            target = ("cap", branch.target_cap_id)
            endpoint_records.extend(
                [
                    (branch.branch_id, "source", source, np.asarray(branch.points[0])),
                    (branch.branch_id, "target", target, np.asarray(branch.points[-1])),
                ]
            )
            graph.add_node(source, kind="cap", cap_id=branch.source_cap_id, position=branch.points[0].tolist())
            graph.add_node(target, kind="cap", cap_id=branch.target_cap_id, position=branch.points[-1].tolist())
            graph.add_edge(
                source,
                target,
                kind="branch",
                branch_id=branch.branch_id,
                length=float(branch.length),
                parent_branch_id=branch.parent_branch_id,
                children_branch_ids=list(branch.children_branch_ids),
                section_count=len(branch.sections),
            )
            branch_to_edge[branch.branch_id] = (source, target)

        if endpoint_tolerance is not None:
            remap = cls._merge_spatially_close_nodes(graph, endpoint_tolerance)
            branch_to_edge = {
                branch_id: (remap.get(u, u), remap.get(v, v))
                for branch_id, (u, v) in branch_to_edge.items()
            }
        return cls(graph, branch_to_edge)

    @staticmethod
    def _merge_spatially_close_nodes(graph: nx.Graph, tolerance: float) -> Dict[Any, Any]:
        if tolerance <= 0:
            raise ValueError("endpoint_tolerance must be positive")
        nodes = list(graph.nodes)
        parent = {node: node for node in nodes}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, a in enumerate(nodes):
            pa = graph.nodes[a].get("position")
            if pa is None:
                continue
            for b in nodes[i + 1 :]:
                pb = graph.nodes[b].get("position")
                if pb is not None and np.linalg.norm(np.asarray(pa) - np.asarray(pb)) <= tolerance:
                    union(a, b)
        groups: Dict[Any, list[Any]] = {}
        for node in nodes:
            groups.setdefault(find(node), []).append(node)
        remap: Dict[Any, Any] = {}
        for representative, members in groups.items():
            for member in members:
                remap[member] = representative
            if len(members) <= 1:
                continue
            position = np.mean([graph.nodes[n]["position"] for n in members], axis=0).tolist()
            graph.add_node(representative, position=position, kind="junction", merged_nodes=[str(n) for n in members])
            for member in members:
                if member != representative:
                    for neighbor, attrs in list(graph[member].items()):
                        if neighbor not in members:
                            graph.add_edge(representative, neighbor, **attrs)
                    graph.remove_node(member)
        return remap

    def validate(self) -> GraphValidation:
        graph = self.graph
        components = tuple(nx.connected_components(graph))
        roots = tuple(n for n, degree in graph.degree() if degree == 1 and self._is_root(n))
        terminals = tuple(n for n, degree in graph.degree() if degree == 1 and n not in roots)
        bifurcations = tuple(n for n, degree in graph.degree() if degree >= 3)
        isolated = tuple(n for n, degree in graph.degree() if degree == 0)
        return GraphValidation(
            connected=nx.is_connected(graph) if graph.number_of_nodes() else False,
            acyclic=nx.is_forest(graph),
            branch_count=sum(1 for _, _, d in graph.edges(data=True) if d.get("kind") == "branch"),
            node_count=graph.number_of_nodes(),
            component_count=len(components),
            roots=roots,
            terminals=terminals,
            bifurcations=bifurcations,
            isolated_nodes=isolated,
        )

    def _is_root(self, node: Any) -> bool:
        incoming = [
            self.graph.edges[node, other].get("parent_branch_id")
            for other in self.graph.neighbors(node)
        ]
        return any(value is None for value in incoming)

    def branch_volume_weights(self) -> Dict[int, float]:
        """Return centerline-integrated branch volumes for partitioning policies."""
        weights: Dict[int, float] = {}
        for branch_id, (u, v) in self.branch_to_edge.items():
            edge = self.graph.get_edge_data(u, v) or {}
            weights[branch_id] = float(edge.get("section_volume", 0.0))
        return weights

    def as_dict(self) -> Dict[str, Any]:
        validation = self.validate()
        return {
            "validation": validation.as_dict(),
            "nodes": [
                {"id": str(node), **{key: value for key, value in attrs.items()}}
                for node, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": str(u), "target": str(v), **attrs}
                for u, v, attrs in self.graph.edges(data=True)
            ],
            "branch_to_edge": {str(k): [str(x) for x in v] for k, v in self.branch_to_edge.items()},
        }

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2, default=str))
        return destination


def build_vascular_graph(
    data: GeometryAnalysisData, endpoint_tolerance: Optional[float] = None
) -> VascularGraph:
    """Build and validate a graph from serializable analysis data."""
    return VascularGraph.from_analysis(data, endpoint_tolerance=endpoint_tolerance)
