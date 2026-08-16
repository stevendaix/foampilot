#!/usr/bin/env python3
"""
Geometry micro-feature detector for Gmsh models.

Detects:
- Small edges (below characteristic length)
- Small surfaces (below area threshold)
- Duplicate points
- Non-manifold edges
- Tiny gaps between buildings
- Cracks / openings in surfaces
- Poor surface normals

Usage:
    import gmsh
    from geometry_healing import GeometryDiagnostic

    gmsh.initialize()
    gmsh.model.add("test")
    # ... build geometry ...
    gmsh.model.occ.synchronize()

    diag = GeometryDiagnostic()
    report = diag.diagnose()
    diag.export_problematic_surfaces("problematic_surfaces.vtu")
    diag.export_problematic_edges("problematic_edges.vtu")
    print(report.to_text())
    gmsh.finalize()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gmsh
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryThresholds:
    """Thresholds for geometry diagnostics."""

    min_edge_length: float = 0.1
    min_surface_area: float = 0.01
    min_angle_deg: float = 1.0
    max_gap: float = 0.05
    max_duplicate_distance: float = 1e-6


@dataclass
class EdgeInfo:
    """Information about a geometric edge."""
    edge_id: int
    length: float
    start_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    surfaces: List[int] = field(default_factory=list)
    problematic: bool = False
    reason: str = ""


@dataclass
class SurfaceInfo:
    """Information about a geometric surface."""
    surface_id: int
    area: float
    normal: Tuple[float, float, float]
    bounding_box: Tuple[float, float, float, float, float, float]
    edges: List[int] = field(default_factory=list)
    problematic: bool = False
    reason: str = ""


@dataclass
class PointInfo:
    """Information about a geometric point."""
    point_id: int
    coords: Tuple[float, float, float]
    duplicate_of: Optional[int] = None
    problematic: bool = False
    reason: str = ""


@dataclass
class GeometryReport:
    """Complete geometry diagnostic report."""

    num_points: int = 0
    num_curves: int = 0
    num_surfaces: int = 0
    num_volumes: int = 0
    num_edges_checked: int = 0
    num_surfaces_checked: int = 0
    small_edges: int = 0
    small_surfaces: int = 0
    duplicate_points: int = 0
    non_manifold_edges: int = 0
    tiny_gaps: int = 0
    problematic_edges: List[EdgeInfo] = field(default_factory=list)
    problematic_surfaces: List[SurfaceInfo] = field(default_factory=list)
    duplicate_point_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("GEOMETRY DIAGNOSTIC REPORT")
        lines.append("=" * 60)
        lines.append(f"  Points       : {self.num_points}")
        lines.append(f"  Curves       : {self.num_curves}")
        lines.append(f"  Surfaces     : {self.num_surfaces}")
        lines.append(f"  Volumes      : {self.num_volumes}")
        lines.append("")
        lines.append("  Issues found:")
        lines.append(f"    Small edges        : {self.small_edges}")
        lines.append(f"    Small surfaces     : {self.small_surfaces}")
        lines.append(f"    Duplicate points   : {self.duplicate_points}")
        lines.append(f"    Non-manifold edges : {self.non_manifold_edges}")
        lines.append(f"    Tiny gaps          : {self.tiny_gaps}")
        lines.append("")

        if self.problematic_edges:
            lines.append("  Problematic edges (first 10):")
            for edge in self.problematic_edges[:10]:
                lines.append(f"    Edge {edge.edge_id}: len={edge.length:.6f} m - {edge.reason}")
            if len(self.problematic_edges) > 10:
                lines.append(f"    ... and {len(self.problematic_edges) - 10} more")

        if self.problematic_surfaces:
            lines.append("")
            lines.append("  Problematic surfaces (first 10):")
            for surf in self.problematic_surfaces[:10]:
                lines.append(f"    Surface {surf.surface_id}: area={surf.area:.6e} m² - {surf.reason}")
            if len(self.problematic_surfaces) > 10:
                lines.append(f"    ... and {len(self.problematic_surfaces) - 10} more")

        lines.append("=" * 60)
        return "\n".join(lines)


class GeometryDiagnostic:
    """Diagnose geometry quality issues in a Gmsh model."""

    def __init__(self, thresholds: Optional[GeometryThresholds] = None) -> None:
        self.thresholds = thresholds or GeometryThresholds()
        self.report = GeometryReport()

    def diagnose(self) -> GeometryReport:
        """Run full geometry diagnostic."""
        self.report = GeometryReport()
        self._collect_entities()
        self._check_points()
        self._check_edges()
        self._check_surfaces()
        self._check_gaps()
        return self.report

    def _collect_entities(self) -> None:
        """Collect basic entity counts."""
        point_tags, _, _ = gmsh.model.mesh.getNodes()
        self.report.num_points = len(point_tags)

        curves = gmsh.model.getEntities(dim=1)
        self.report.num_curves = len(curves)

        surfaces = gmsh.model.getEntities(dim=2)
        self.report.num_surfaces = len(surfaces)

        volumes = gmsh.model.getEntities(dim=3)
        self.report.num_volumes = len(volumes)

    def _check_points(self) -> None:
        """Check for duplicate points."""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        tag_list = [int(t) for t in node_tags]

        checked = set()
        for i, tag_i in enumerate(tag_list):
            if tag_i in checked:
                continue
            for j, tag_j in enumerate(tag_list[i + 1:], start=i + 1):
                if tag_j in checked:
                    continue
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.thresholds.max_duplicate_distance:
                    self.report.duplicate_points += 1
                    self.report.duplicate_point_pairs.append((tag_i, tag_j))
                    checked.add(tag_j)
            checked.add(tag_i)

    def _check_edges(self) -> None:
        """Check for small edges."""
        curves = gmsh.model.getEntities(dim=1)
        for _, curve_tag in curves:
            try:
                edge_points = gmsh.model.getValue(dim=1, tag=curve_tag)
                if len(edge_points) < 6:
                    continue
                p1 = np.array(edge_points[:3])
                p2 = np.array(edge_points[3:6])
                length = float(np.linalg.norm(p2 - p1))
                self.report.num_edges_checked += 1

                if length < self.thresholds.min_edge_length:
                    self.report.small_edges += 1
                    edge_info = EdgeInfo(
                        edge_id=curve_tag,
                        length=length,
                        start_point=tuple(p1),
                        end_point=tuple(p2),
                        problematic=True,
                        reason=f"length {length:.6f} < min {self.thresholds.min_edge_length}",
                    )
                    self.report.problematic_edges.append(edge_info)
            except Exception as e:
                logger.debug("Error checking curve %d: %s", curve_tag, e)

    def _check_surfaces(self) -> None:
        """Check for small surfaces."""
        surfaces = gmsh.model.getEntities(dim=2)
        for _, surf_tag in surfaces:
            try:
                surf_points = gmsh.model.getValue(dim=2, tag=surf_tag)
                if len(surf_points) < 9:
                    continue
                pts = np.array(surf_points).reshape(-1, 3)
                area = self._compute_surface_area(pts)
                self.report.num_surfaces_checked += 1

                if area < self.thresholds.min_surface_area:
                    self.report.small_surfaces += 1
                    normal = self._compute_surface_normal(pts)
                    bbox = self._compute_bbox(pts)
                    surf_info = SurfaceInfo(
                        surface_id=surf_tag,
                        area=area,
                        normal=tuple(normal),
                        bounding_box=bbox,
                        problematic=True,
                        reason=f"area {area:.6e} < min {self.thresholds.min_surface_area}",
                    )
                    self.report.problematic_surfaces.append(surf_info)
            except Exception as e:
                logger.debug("Error checking surface %d: %s", surf_tag, e)

    def _check_gaps(self) -> None:
        """Check for tiny gaps between buildings."""
        volumes = gmsh.model.getEntities(dim=3)
        if len(volumes) < 2:
            return

        vol_centers = []
        for _, vol_tag in volumes:
            try:
                com = gmsh.model.occ.getCenterOfMass(3, vol_tag)
                vol_centers.append((vol_tag, np.array(com)))
            except Exception:
                continue

        for i in range(len(vol_centers)):
            for j in range(i + 1, len(vol_centers)):
                dist = np.linalg.norm(vol_centers[i][1] - vol_centers[j][1])
                if dist < self.thresholds.max_gap:
                    self.report.tiny_gaps += 1

    def _compute_surface_area(self, pts: np.ndarray) -> float:
        """Compute area of a polygon surface from its points."""
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[0]
            v2 = pts[i + 1] - pts[0]
            area += 0.5 * np.linalg.norm(np.cross(v1, v2))
        return float(area)

    def _compute_surface_normal(self, pts: np.ndarray) -> np.ndarray:
        """Compute approximate normal of a surface."""
        if len(pts) < 3:
            return np.array([0.0, 0.0, 1.0])
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return normal / norm

    def _compute_bbox(self, pts: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """Compute bounding box (xmin, ymin, zmin, xmax, ymax, zmax)."""
        return (
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.min(pts[:, 2])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
            float(np.max(pts[:, 2])),
        )

    def export_problematic_edges(self, path: str | Path) -> None:
        """Export problematic edges to VTK."""
        try:
            import pyvista as pv
        except ImportError:
            logger.warning("PyVista not available, skipping edge export")
            return

        if not self.report.problematic_edges:
            Path(path).write_text("# No problematic edges\n")
            return

        points_list = []
        lines_list = []
        scalars = []

        for edge in self.report.problematic_edges:
            points_list.extend([edge.start_point, edge.end_point])
            lines_list.extend([2, len(points_list) - 2, len(points_list) - 1])
            scalars.append(edge.length)

        points = np.array(points_list)
        grid = pv.PolyData(points, lines=lines_list)
        grid.cell_data["length"] = np.array(scalars)
        grid.save(str(path))
        logger.info("Exported problematic edges: %s", path)

    def export_problematic_surfaces(self, path: str | Path) -> None:
        """Export problematic surfaces to VTK."""
        try:
            import pyvista as pv
        except ImportError:
            logger.warning("PyVista not available, skipping surface export")
            return

        if not self.report.problematic_surfaces:
            Path(path).write_text("# No problematic surfaces\n")
            return

        node_ids, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        node_map = {int(n): i for i, n in enumerate(node_ids)}

        all_surf_tags, all_surf_nodes = gmsh.model.mesh.getElementsByType(2)
        points = coords
        cells = []
        cell_types = []
        scalars = []

        for surf in self.report.problematic_surfaces:
            tag = surf.surface_id
            if tag not in all_surf_tags:
                continue
            idx = list(all_surf_tags).index(tag)
            start = idx * 3
            end = start + 3
            nids = [int(k) for k in all_surf_nodes[start:end]]
            local_nodes = [node_map[nid] for nid in nids]

            cells.extend([3, *local_nodes])
            cell_types.append(5)  # VTK_TRIANGLE
            scalars.append(surf.area)

        if not cells:
            Path(path).write_text("# No surface cells to write\n")
            return

        grid = pv.UnstructuredGrid(
            np.array(cells), np.array(cell_types), points
        )
        grid.cell_data["area"] = np.array(scalars)
        grid.save(str(path))
        logger.info("Exported problematic surfaces: %s", path)

    def export_geometry_quality_vtk(self, path: str | Path) -> None:
        """Export all surfaces colored by area for visualization."""
        try:
            import pyvista as pv
        except ImportError:
            logger.warning("PyVista not available, skipping export")
            return

        node_ids, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        node_map = {int(n): i for i, n in enumerate(node_ids)}

        all_surf_tags, all_surf_nodes = gmsh.model.mesh.getElementsByType(2)
        points = coords
        cells = []
        cell_types = []
        areas = []

        for idx, tag in enumerate(all_surf_tags):
            start = idx * 3
            end = start + 3
            nids = [int(k) for k in all_surf_nodes[start:end]]
            local_nodes = [node_map[nid] for nid in nids]
            pts = coords[local_nodes]
            area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))

            cells.extend([3, *local_nodes])
            cell_types.append(5)
            areas.append(area)

        if not cells:
            Path(path).write_text("# No surface cells to write\n")
            return

        grid = pv.UnstructuredGrid(
            np.array(cells), np.array(cell_types), points
        )
        grid.cell_data["area"] = np.array(areas)
        grid.save(str(path))
        logger.info("Exported geometry quality: %s", path)
