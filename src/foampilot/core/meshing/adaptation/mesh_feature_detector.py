#!/usr/bin/env python3
"""
Mesh-based micro-feature detector.

Analyzes the MESH (not the geometry) to detect:
- Very small surface triangles
- Very small volume cells
- High aspect ratio elements
- Thin sliver elements
- Elements near building surfaces with poor quality

This complements geometry_healing.py by looking at the actual mesh elements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gmsh
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeshFeatureThresholds:
    """Thresholds for mesh feature detection."""

    min_triangle_area: float = 0.001
    min_tet_volume: float = 1e-6
    max_aspect_ratio: float = 15.0
    max_skewness: float = 0.8
    small_element_ratio_warning: float = 0.01
    small_element_ratio_fail: float = 0.05


@dataclass
class MeshFeatureReport:
    """Report of mesh micro-features."""

    num_nodes: int = 0
    num_tets: int = 0
    num_tris: int = 0
    small_triangles: int = 0
    small_tets: int = 0
    high_aspect_ratio_tets: int = 0
    high_aspect_ratio_tris: int = 0
    sliver_tets: int = 0
    problematic_elements: List[Dict[str, Any]] = field(default_factory=list)

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("MESH MICRO-FEATURE REPORT")
        lines.append("=" * 60)
        lines.append(f"  Nodes           : {self.num_nodes}")
        lines.append(f"  Tetrahedra      : {self.num_tets}")
        lines.append(f"  Triangles       : {self.num_tris}")
        lines.append("")
        lines.append("  Micro-features:")
        lines.append(f"    Small triangles     : {self.small_triangles}")
        lines.append(f"    Small tets         : {self.small_tets}")
        lines.append(f"    High aspect tets   : {self.high_aspect_ratio_tets}")
        lines.append(f"    High aspect tris   : {self.high_aspect_ratio_tris}")
        lines.append(f"    Sliver tets        : {self.sliver_tets}")
        lines.append("")

        if self.problematic_elements:
            lines.append("  Worst elements:")
            for elem in self.problematic_elements[:10]:
                lines.append(f"    {elem['type']} {elem['id']}: quality={elem['quality']:.6f}, reason={elem['reason']}")

        lines.append("=" * 60)
        return "\n".join(lines)


class MeshFeatureDetector:
    """Detect micro-features in a Gmsh mesh."""

    def __init__(self, thresholds: Optional[MeshFeatureThresholds] = None) -> None:
        self.thresholds = thresholds or MeshFeatureThresholds()
        self.report = MeshFeatureReport()

    def analyze(self) -> MeshFeatureReport:
        """Analyze mesh for micro-features."""
        self.report = MeshFeatureReport()

        node_ids, node_coords, _ = gmsh.model.mesh.getNodes()
        self.report.num_nodes = len(node_ids)
        coords = np.array(node_coords).reshape(-1, 3)
        node_map = {int(n): i for i, n in enumerate(node_ids)}

        # Analyze tets
        tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4)
        self.report.num_tets = len(tet_tags)
        for j, tag in enumerate(tet_tags):
            nids = [int(k) for k in tet_nodes[j*4:(j+1)*4]]
            pts = coords[[node_map[nid] for nid in nids]]
            self._check_tet(tag, pts)

        # Analyze triangles
        tri_tags, tri_nodes = gmsh.model.mesh.getElementsByType(2)
        self.report.num_tris = len(tri_tags)
        for j, tag in enumerate(tri_tags):
            nids = [int(k) for k in tri_nodes[j*3:(j+1)*3]]
            pts = coords[[node_map[nid] for nid in nids]]
            self._check_tri(tag, pts)

        return self.report

    def _check_tet(self, tag: int, pts: np.ndarray) -> None:
        """Check a tetrahedron for micro-features."""
        vol = abs(np.dot(pts[0] - pts[3], np.cross(pts[1] - pts[3], pts[2] - pts[3]))) / 6.0
        edge_lengths = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[0] - pts[2]),
            np.linalg.norm(pts[3] - pts[0]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[1]),
        ]
        max_edge = max(edge_lengths)
        min_edge = min(edge_lengths)
        aspect_ratio = max_edge / min_edge if min_edge > 1e-12 else float("inf")

        # Sliver detection: small volume but reasonable edge lengths
        mean_edge = float(np.mean(edge_lengths))
        slenderness = vol / (mean_edge ** 3) if mean_edge > 1e-12 else 0.0

        problematic = False
        reasons = []

        if vol < self.thresholds.min_tet_volume:
            self.report.small_tets += 1
            problematic = True
            reasons.append("small_volume")

        if aspect_ratio > self.thresholds.max_aspect_ratio:
            self.report.high_aspect_ratio_tets += 1
            problematic = True
            reasons.append("high_aspect_ratio")

        if slenderness < 0.01 and vol > 1e-12:
            self.report.sliver_tets += 1
            problematic = True
            reasons.append("sliver")

        if problematic:
            self.report.problematic_elements.append({
                "type": "tet",
                "id": tag,
                "volume": vol,
                "aspect_ratio": aspect_ratio,
                "quality": slenderness,
                "reason": ", ".join(reasons),
            })

    def _check_tri(self, tag: int, pts: np.ndarray) -> None:
        """Check a triangle for micro-features."""
        a = np.linalg.norm(pts[1] - pts[0])
        b = np.linalg.norm(pts[2] - pts[1])
        c = np.linalg.norm(pts[0] - pts[2])
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        max_edge = max(a, b, c)
        min_edge = min(a, b, c)
        aspect_ratio = max_edge / min_edge if min_edge > 1e-12 else float("inf")

        problematic = False
        reasons = []

        if area < self.thresholds.min_triangle_area:
            self.report.small_triangles += 1
            problematic = True
            reasons.append("small_area")

        if aspect_ratio > self.thresholds.max_aspect_ratio:
            self.report.high_aspect_ratio_tris += 1
            problematic = True
            reasons.append("high_aspect_ratio")

        if problematic:
            self.report.problematic_elements.append({
                "type": "tri",
                "id": tag,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "quality": area / (max_edge ** 2) if max_edge > 1e-12 else 0.0,
                "reason": ", ".join(reasons),
            })
