#!/usr/bin/env python3
"""
Wall-adjacent mesh quality analyzer for OpenFOAM cases.

Reads OpenFOAM polyMesh files directly and analyzes wall-adjacent quality:
- Surface triangle size on walls
- First cell height and quality
- Cell quality near walls
- Transition quality from wall to far-field

Usage:
    from wall_mesh_analyzer import WallMeshAnalyzer
    from pathlib import Path

    analyzer = WallMeshAnalyzer(Path("case"))
    report = analyzer.analyze()
    print(report.to_text())
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WallMeshThresholds:
    """Thresholds for wall-adjacent mesh quality."""

    max_surface_triangle_area: float = 1.0
    min_surface_coverage: float = 0.8
    max_first_cell_ratio: float = 2.0
    min_wall_quality: float = 0.3
    max_transition_ratio: float = 1.5
    wall_tolerance: float = 0.5


@dataclass
class WallElementInfo:
    """Information about a wall-adjacent element."""

    element_id: int
    element_type: int
    distance_to_wall: float
    volume: float = 0.0
    area: float = 0.0
    quality: float = 0.0
    aspect_ratio: float = 0.0
    is_first_cell: bool = False
    problematic: bool = False
    reason: str = ""


@dataclass
class WallMeshReport:
    """Wall-adjacent mesh quality report."""

    num_wall_faces: int = 0
    num_wall_cells: int = 0
    num_first_cells: int = 0
    avg_surface_triangle_area: float = 0.0
    max_surface_triangle_area: float = 0.0
    min_surface_triangle_area: float = 0.0
    surface_coverage_ratio: float = 0.0
    avg_first_cell_volume: float = 0.0
    min_first_cell_volume: float = 0.0
    max_first_cell_volume: float = 0.0
    avg_wall_quality: float = 0.0
    min_wall_quality: float = 0.0
    problematic_wall_elements: List[WallElementInfo] = field(default_factory=list)
    wall_quality_by_distance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("WALL-ADJACENT MESH QUALITY REPORT")
        lines.append("=" * 60)
        lines.append(f"  Wall faces           : {self.num_wall_faces}")
        lines.append(f"  Wall cells           : {self.num_wall_cells}")
        lines.append(f"  First cells (wall)   : {self.num_first_cells}")
        lines.append("")
        lines.append("  Surface triangles:")
        lines.append(f"    min area           : {self.min_surface_triangle_area:.6e} m²")
        lines.append(f"    avg area           : {self.avg_surface_triangle_area:.6e} m²")
        lines.append(f"    max area           : {self.max_surface_triangle_area:.6e} m²")
        lines.append(f"    coverage ratio     : {self.surface_coverage_ratio:.2%}")
        lines.append("")
        lines.append("  First cells:")
        lines.append(f"    min volume         : {self.min_first_cell_volume:.6e} m³")
        lines.append(f"    avg volume         : {self.avg_first_cell_volume:.6e} m³")
        lines.append(f"    max volume         : {self.max_first_cell_volume:.6e} m³")
        lines.append("")
        lines.append("  Wall quality:")
        lines.append(f"    min quality        : {self.min_wall_quality:.6f}")
        lines.append(f"    avg quality        : {self.avg_wall_quality:.6f}")
        lines.append("")

        if self.problematic_wall_elements:
            lines.append("  Problematic wall elements (first 10):")
            for elem in self.problematic_wall_elements[:10]:
                lines.append(f"    {elem.element_type} {elem.element_id}: dist={elem.distance_to_wall:.4f}m, quality={elem.quality:.4f} - {elem.reason}")
            if len(self.problematic_wall_elements) > 10:
                lines.append(f"    ... and {len(self.problematic_wall_elements) - 10} more")

        lines.append("=" * 60)
        return "\n".join(lines)


class WallMeshAnalyzer:
    """Analyze wall-adjacent mesh quality in an OpenFOAM case."""

    def __init__(self, case_dir: Path, thresholds: Optional[WallMeshThresholds] = None) -> None:
        self.case_dir = Path(case_dir)
        self.thresholds = thresholds or WallMeshThresholds()
        self.report = WallMeshReport()

    def analyze(self) -> WallMeshReport:
        """Run full wall-adjacent mesh analysis."""
        self.report = WallMeshReport()
        self._load_openfoam_mesh()
        self._identify_wall_faces()
        self._analyze_surface_quality()
        self._analyze_wall_cells()
        self._compute_distance_bins()
        self._check_thresholds()
        return self.report

    def _read_openfoam_file(self, filename: str) -> str:
        """Read an OpenFOAM mesh file."""
        filepath = self.case_dir / "constant" / "polyMesh" / filename
        if not filepath.exists():
            raise FileNotFoundError(f"OpenFOAM file not found: {filepath}")
        return filepath.read_text()

    def _parse_points(self) -> np.ndarray:
        """Parse OpenFOAM points file."""
        content = self._read_openfoam_file("points")

        # Find the number of points
        lines = content.split('\n')
        n_points = None
        for line in lines:
            line = line.strip()
            if line.isdigit():
                n_points = int(line)
                break

        if n_points is None:
            raise ValueError("Could not find number of points")

        # Find the start of the coordinate list
        coord_start = None
        for i, line in enumerate(lines):
            if line.strip() == '(':
                coord_start = i + 1
                break

        if coord_start is None:
            raise ValueError("Could not find coordinate list start")

        coords = []
        for i in range(coord_start, len(lines)):
            line = lines[i].strip()
            if line == ')':
                break
            if line:
                values = line.replace('(', '').replace(')', '').split()
                if len(values) >= 3:
                    coords.extend([float(values[0]), float(values[1]), float(values[2])])

        coords = np.array(coords[:n_points * 3]).reshape(n_points, 3)
        return coords

    def _parse_faces(self) -> List[List[int]]:
        """Parse OpenFOAM faces file."""
        content = self._read_openfoam_file("faces")

        # Find number of faces
        n_match = re.search(r'^(\d+)\s*$', content, re.MULTILINE)
        if not n_match:
            return []
        n_faces = int(n_match.group(1))

        # Extract all face definitions: N(node1 node2 ...)
        face_pattern = re.compile(r'(\d+)\(([^)]+)\)')
        faces = []
        for match in face_pattern.finditer(content):
            size = int(match.group(1))
            nodes_str = match.group(2).split()
            if len(nodes_str) >= size:
                face_nodes = [int(nodes_str[i]) for i in range(size)]
                faces.append(face_nodes)

        return faces[:n_faces]

    def _parse_owner(self) -> List[int]:
        """Parse OpenFOAM owner file."""
        content = self._read_openfoam_file("owner")
        match = re.search(r'(\d+)\s*\(([\d\s]+)\)', content, re.DOTALL)
        if not match:
            return []

        n_cells = int(match.group(1))
        owner_str = match.group(2).split()
        return [int(owner_str[i]) for i in range(min(len(owner_str), n_cells))]

    def _parse_neighbour(self) -> List[int]:
        """Parse OpenFOAM neighbour file."""
        content = self._read_openfoam_file("neighbour")
        match = re.search(r'(\d+)\s*\(([\d\s]+)\)', content, re.DOTALL)
        if not match:
            return []

        n_internal_faces = int(match.group(1))
        neighbour_str = match.group(2).split()
        return [int(neighbour_str[i]) for i in range(min(len(neighbour_str), n_internal_faces))]

    def _parse_cells(self) -> List[List[int]]:
        """Parse OpenFOAM cells file or reconstruct from owner/neighbour."""
        filepath = self.case_dir / "constant" / "polyMesh" / "cells"
        if filepath.exists():
            content = filepath.read_text()
            match = re.search(r'(\d+)\s*\(([\d\s]+)\)', content, re.DOTALL)
            if match:
                n_cells = int(match.group(1))
                cells_str = match.group(2).split()
                cells = []
                idx = 0
                for _ in range(n_cells):
                    if idx >= len(cells_str):
                        break
                    size = int(cells_str[idx])
                    idx += 1
                    if idx + size > len(cells_str):
                        break
                    cell_nodes = [int(cells_str[idx + j]) for j in range(size)]
                    cells.append(cell_nodes)
                    idx += size
                return cells

        owner = self._parse_owner()
        neighbour = self._parse_neighbour()
        n_cells = max(owner) + 1 if owner else 0
        cell_faces = [[] for _ in range(n_cells)]
        for face_idx, cell_idx in enumerate(owner):
            if face_idx < len(self.faces):
                cell_faces[cell_idx].append(self.faces[face_idx])

        cells = []
        for face_list in cell_faces:
            cell_nodes = []
            for face in face_list:
                for node in face:
                    if node not in cell_nodes:
                        cell_nodes.append(node)
            cells.append(cell_nodes)
        return cells

    def _parse_boundary(self) -> Dict[str, Tuple[int, int]]:
        """Parse OpenFOAM boundary file.

        Returns dict: {patch_name: (start_face, n_faces)}
        """
        content = self._read_openfoam_file("boundary")
        patches = {}
        for match in re.finditer(r'(\w+)\s*\{[^}]*nFaces\s+(\d+)[^}]*startFace\s+(\d+)', content):
            name = match.group(1)
            n_faces = int(match.group(2))
            start_face = int(match.group(3))
            patches[name] = (start_face, n_faces)
        return patches

    def _load_openfoam_mesh(self) -> None:
        """Load OpenFOAM mesh data."""
        self.coords = self._parse_points()
        self.faces = self._parse_faces()
        self.cells = self._parse_cells()
        self.owner = self._parse_owner()
        self.neighbour = self._parse_neighbour()
        self.boundary = self._parse_boundary()

        self.xmin = float(np.min(self.coords[:, 0]))
        self.xmax = float(np.max(self.coords[:, 0]))
        self.ymin = float(np.min(self.coords[:, 1]))
        self.ymax = float(np.max(self.coords[:, 1]))
        self.zmin = float(np.min(self.coords[:, 2]))
        self.zmax = float(np.max(self.coords[:, 2]))

    def _is_wall_face(self, face_center: np.ndarray) -> bool:
        """Determine if a face center is on a wall boundary."""
        tol = self.thresholds.wall_tolerance
        if abs(face_center[2] - self.zmin) < tol:
            return True
        if abs(face_center[0] - self.xmin) < tol or abs(face_center[0] - self.xmax) < tol:
            return True
        if abs(face_center[1] - self.ymin) < tol or abs(face_center[1] - self.ymax) < tol:
            return True
        return False

    def _identify_wall_faces(self) -> None:
        """Identify wall faces from boundary patches."""
        self.wall_face_indices = []
        self.wall_face_centers = []

        wall_patches = ["buildings", "GROUND", "SIDE_NORTH", "SIDE_SOUTH", "WALL", "WALLS"]
        inlet_outlet = ["INLET", "OUTLET", "TOP", "SYMMETRY", "atm"]

        for patch_name, (start_face, n_faces) in self.boundary.items():
            # Treat as wall if not explicitly inlet/outlet/top
            is_wall = any(wall.lower() in patch_name.lower() for wall in wall_patches)
            is_not_inlet = not any(io.lower() in patch_name.lower() for io in inlet_outlet)

            if is_wall or (is_not_inlet and patch_name not in ['patch0']):
                for face_idx in range(start_face, min(start_face + n_faces, len(self.faces))):
                    face_node_ids = self.faces[face_idx]
                    if len(face_node_ids) >= 3:
                        pts = self.coords[face_node_ids[:3]]
                        center = pts.mean(axis=0)
                        self.wall_face_indices.append(face_idx)
                        self.wall_face_centers.append(center)

        # If still no wall faces, use geometric proximity
        if not self.wall_face_indices:
            for face_idx, face_node_ids in enumerate(self.faces):
                if len(face_node_ids) >= 3:
                    pts = self.coords[face_node_ids[:3]]
                    center = pts.mean(axis=0)
                    if self._is_wall_face(center):
                        self.wall_face_indices.append(face_idx)
                        self.wall_face_centers.append(center)

        self.report.num_wall_faces = len(self.wall_face_indices)

    def _analyze_surface_quality(self) -> None:
        """Analyze surface triangle quality on walls."""
        if not self.wall_face_indices:
            return

        areas = []
        for face_idx in self.wall_face_indices:
            face_node_ids = self.faces[face_idx]
            if len(face_node_ids) >= 3:
                pts = self.coords[face_node_ids[:3]]
                area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
                areas.append(area)

        if areas:
            self.report.min_surface_triangle_area = float(np.min(areas))
            self.report.avg_surface_triangle_area = float(np.mean(areas))
            self.report.max_surface_triangle_area = float(np.max(areas))
            small_area = sum(1 for a in areas if a < self.thresholds.max_surface_triangle_area)
            self.report.surface_coverage_ratio = small_area / len(areas)

    def _build_cells_from_owner(self) -> List[List[int]]:
        """Reconstruct cell node lists from faces and owner."""
        if not self.cells or len(self.cells) != max(self.owner) + 1 or any(len(c) == 0 for c in self.cells):
            n_cells = max(self.owner) + 1 if self.owner else 0
            cell_faces = [[] for _ in range(n_cells)]
            for face_idx, cell_idx in enumerate(self.owner):
                if face_idx < len(self.faces):
                    cell_faces[cell_idx].append(self.faces[face_idx])

            cells = []
            for face_list in cell_faces:
                cell_nodes = []
                for face in face_list:
                    for node in face:
                        if node not in cell_nodes:
                            cell_nodes.append(node)
                cells.append(cell_nodes)
            return cells
        return self.cells

    def _analyze_wall_cells(self) -> None:
        """Analyze cells adjacent to walls using owner array."""
        if not self.wall_face_centers or not self.owner:
            return

        wall_cell_ids = set()
        first_cell_threshold = self.thresholds.max_surface_triangle_area * 2
        cells = self._build_cells_from_owner()

        for face_idx in self.wall_face_indices:
            if face_idx >= len(self.owner):
                continue
            cell_idx = self.owner[face_idx]
            if cell_idx in wall_cell_ids:
                continue
            if cell_idx >= len(cells) or len(cells[cell_idx]) < 4:
                continue

            cell_node_ids = cells[cell_idx][:4]
            pts = self.coords[cell_node_ids]
            centroid = pts.mean(axis=0)

            min_dist = float('inf')
            for center in self.wall_face_centers:
                dist = np.linalg.norm(centroid - center)
                if dist < min_dist:
                    min_dist = dist

            vol = abs(np.dot(pts[0] - pts[3], np.cross(pts[1] - pts[3], pts[2] - pts[3]))) / 6.0
            edges = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[0] - pts[2]),
                np.linalg.norm(pts[3] - pts[0]),
                np.linalg.norm(pts[2] - pts[3]),
                np.linalg.norm(pts[3] - pts[1]),
            ]
            max_edge = max(edges)
            min_edge = min(edges)
            aspect_ratio = max_edge / min_edge if min_edge > 1e-12 else float("inf")
            mean_edge = float(np.mean(edges))
            quality = vol / (mean_edge ** 3) if mean_edge > 1e-12 else 0.0
            is_first = min_dist < first_cell_threshold

            elem_info = WallElementInfo(
                element_id=cell_idx,
                element_type=4,
                distance_to_wall=float(min_dist),
                volume=vol,
                quality=quality,
                aspect_ratio=aspect_ratio,
                is_first_cell=is_first,
            )

            if vol < 1e-6:
                elem_info.problematic = True
                elem_info.reason = "small_volume"
            elif quality < self.thresholds.min_wall_quality:
                elem_info.problematic = True
                elem_info.reason = "low_quality"
            elif aspect_ratio > self.thresholds.max_transition_ratio * 5:
                elem_info.problematic = True
                elem_info.reason = "high_aspect_ratio"

            self.report.problematic_wall_elements.append(elem_info)
            wall_cell_ids.add(cell_idx)

            if is_first:
                self.report.num_first_cells += 1

        self.report.num_wall_cells = len(wall_cell_ids)

        if self.report.problematic_wall_elements:
            vols = [e.volume for e in self.report.problematic_wall_elements]
            quals = [e.quality for e in self.report.problematic_wall_elements]
            self.report.min_first_cell_volume = float(np.min(vols))
            self.report.avg_first_cell_volume = float(np.mean(vols))
            self.report.max_first_cell_volume = float(np.max(vols))
            self.report.min_wall_quality = float(np.min(quals))
            self.report.avg_wall_quality = float(np.mean(quals))

    def _compute_distance_bins(self) -> None:
        """Compute quality metrics by distance bins from wall."""
        if not self.report.problematic_wall_elements:
            return

        bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float('inf'))]
        for low, high in bins:
            bin_name = f"{low:.1f}-{high:.1f}m"
            bin_elems = [e for e in self.report.problematic_wall_elements if low <= e.distance_to_wall < high]
            if bin_elems:
                quals = [e.quality for e in bin_elems]
                vols = [e.volume for e in bin_elems]
                self.report.wall_quality_by_distance[bin_name] = {
                    "count": len(bin_elems),
                    "min_quality": float(np.min(quals)),
                    "avg_quality": float(np.mean(quals)),
                    "avg_volume": float(np.mean(vols)),
                }

    def _check_thresholds(self) -> None:
        """Check thresholds and count problematic elements."""
        pass

    def export_wall_quality_vtk(self, path: str | Path) -> None:
        """Export wall-adjacent elements with quality scalars."""
        try:
            import pyvista as pv
        except ImportError:
            logger.warning("PyVista not available, skipping export")
            return

        if not self.report.problematic_wall_elements:
            Path(path).write_text("# No wall elements to export\n")
            return

        points = self.coords
        cells = []
        cell_types = []
        scalars_quality = []
        scalars_dist = []

        for elem in self.report.problematic_wall_elements:
            cell_idx = elem.element_id
            if cell_idx >= len(self.cells):
                continue
            nids = self.cells[cell_idx][:4]
            local_nodes = [nid for nid in nids]

            cells.extend([4, *local_nodes])
            cell_types.append(10)  # VTK_TETRA
            scalars_quality.append(elem.quality)
            scalars_dist.append(elem.distance_to_wall)

        if not cells:
            Path(path).write_text("# No cells to export\n")
            return

        grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)
        grid.cell_data["quality"] = np.array(scalars_quality)
        grid.cell_data["distance_to_wall"] = np.array(scalars_dist)
        grid.save(str(path))
        logger.info("Exported wall quality: %s", path)
