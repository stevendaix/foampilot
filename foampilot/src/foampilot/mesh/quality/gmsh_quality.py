"""Mesh quality analysis for Gmsh meshes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gmsh
import numpy as np

logger = logging.getLogger(__name__)

_GMSH_TET = 4
_GMSH_TRI = 2
_VTK_TETRA = 10
_VTK_TRIANGLE = 5


@dataclass
class QualityThresholds:
    """Thresholds for mesh quality checks."""

    min_tet_quality: float = 0.1
    min_tri_quality: float = 0.2
    max_aspect_ratio: float = 10.0
    min_volume: float = 1e-10
    min_area: float = 1e-10
    min_sicn: float = 0.2
    min_sj: float = 0.3
    min_sige: float = 0.3
    gamma_min: float = 0.2


@dataclass
class ElementQuality:
    """Quality metrics for a single element."""

    element_id: int
    element_type: int
    quality: float
    volume: float = 0.0
    area: float = 0.0
    aspect_ratio: float = 0.0
    min_angle: float = 0.0
    max_angle: float = 0.0
    sicn: float = 0.0
    sj: float = 0.0
    sige: float = 0.0
    gamma: float = 0.0
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    isotropy: float = 0.0
    bad: bool = False
    bad_reasons: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Aggregated mesh quality report."""

    num_nodes: int = 0
    num_tets: int = 0
    num_tris: int = 0
    tet_quality_min: float = 0.0
    tet_quality_max: float = 0.0
    tet_quality_mean: float = 0.0
    tri_quality_min: float = 0.0
    tri_quality_max: float = 0.0
    tri_quality_mean: float = 0.0
    volume_min: float = 0.0
    volume_max: float = 0.0
    volume_mean: float = 0.0
    area_min: float = 0.0
    area_max: float = 0.0
    area_mean: float = 0.0
    bad_tets: int = 0
    bad_tris: int = 0
    threshold_violations: Dict[str, int] = field(default_factory=dict)
    element_qualities: List[ElementQuality] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def cells(self) -> int:
        return self.num_tets + self.num_tris

    @property
    def nodes(self) -> int:
        return self.num_nodes

    @property
    def tetrahedra(self) -> int:
        return self.num_tets

    @property
    def surface_triangles(self) -> int:
        return self.num_tris

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    @property
    def volume_metrics(self) -> Dict[str, Any]:
        tet_eqs = [eq for eq in self.element_qualities if eq.element_type == _GMSH_TET]
        if not tet_eqs:
            return {}
        metrics: Dict[str, Any] = {}
        for name in ["quality", "volume", "aspect_ratio", "sicn", "sj", "sige", "gamma", "isotropy"]:
            vals = [getattr(eq, name) for eq in tet_eqs]
            metrics[name] = {
                "min": float(np.min(vals)),
                "P01": self._percentile(vals, 1),
                "P05": self._percentile(vals, 5),
                "P50": self._percentile(vals, 50),
                "P95": self._percentile(vals, 95),
                "P99": self._percentile(vals, 99),
                "max": float(np.max(vals)),
            }
        return metrics

    @property
    def surface_metrics(self) -> Dict[str, Any]:
        tri_eqs = [eq for eq in self.element_qualities if eq.element_type == _GMSH_TRI]
        if not tri_eqs:
            return {}
        metrics: Dict[str, Any] = {}
        for name in ["quality", "area", "aspect_ratio", "sicn", "sj", "sige", "gamma", "isotropy", "min_angle", "max_angle"]:
            vals = [getattr(eq, name) for eq in tri_eqs]
            metrics[name] = {
                "min": float(np.min(vals)),
                "P01": self._percentile(vals, 1),
                "P05": self._percentile(vals, 5),
                "P50": self._percentile(vals, 50),
                "P95": self._percentile(vals, 95),
                "P99": self._percentile(vals, 99),
                "max": float(np.max(vals)),
            }
        return metrics

    @property
    def gmsh_passed(self) -> bool:
        return self.bad_tets == 0 and self.bad_tris == 0

    @property
    def bad_elements(self) -> List[ElementQuality]:
        return [eq for eq in self.element_qualities if eq.bad]

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("MESH QUALITY REPORT")
        lines.append("=" * 60)
        lines.append(f"  Nodes           : {self.num_nodes}")
        lines.append(f"  Tetrahedra      : {self.num_tets}")
        lines.append(f"  Triangles       : {self.num_tris}")
        lines.append("")

        if self.num_tets > 0:
            lines.append("  Tetrahedra quality:")
            lines.append(f"    min  = {self.tet_quality_min:.6f}")
            lines.append(f"    max  = {self.tet_quality_max:.6f}")
            lines.append(f"    mean = {self.tet_quality_mean:.6f}")
            lines.append("")
            tet_eqs = [eq for eq in self.element_qualities if eq.element_type == _GMSH_TET]
            if tet_eqs:
                for name in ["sicn", "sj", "sige", "gamma", "aspect_ratio", "isotropy"]:
                    vals = [getattr(eq, name) for eq in tet_eqs]
                    lines.append(f"    {name}:")
                    lines.append(f"      min={np.min(vals):.6f} P05={self._percentile(vals,5):.6f} P50={self._percentile(vals,50):.6f} P95={self._percentile(vals,95):.6f} max={np.max(vals):.6f}")
            lines.append("")

        if self.num_tris > 0:
            lines.append("  Triangle quality:")
            lines.append(f"    min  = {self.tri_quality_min:.6f}")
            lines.append(f"    max  = {self.tri_quality_max:.6f}")
            lines.append(f"    mean = {self.tri_quality_mean:.6f}")
            lines.append("")
            tri_eqs = [eq for eq in self.element_qualities if eq.element_type == _GMSH_TRI]
            if tri_eqs:
                for name in ["sicn", "sj", "sige", "gamma", "aspect_ratio", "isotropy", "min_angle", "max_angle"]:
                    vals = [getattr(eq, name) for eq in tri_eqs]
                    lines.append(f"    {name}:")
                    lines.append(f"      min={np.min(vals):.6f} P05={self._percentile(vals,5):.6f} P50={self._percentile(vals,50):.6f} P95={self._percentile(vals,95):.6f} max={np.max(vals):.6f}")
            lines.append("")

        lines.append("  Volume:")
        lines.append(f"    min  = {self.volume_min:.6e}")
        lines.append(f"    max  = {self.volume_max:.6e}")
        lines.append(f"    mean = {self.volume_mean:.6e}")
        lines.append("")
        lines.append("  Area:")
        lines.append(f"    min  = {self.area_min:.6e}")
        lines.append(f"    max  = {self.area_max:.6e}")
        lines.append(f"    mean = {self.area_mean:.6e}")
        lines.append("")
        lines.append("  Bad elements:")
        lines.append(f"    Tets = {self.bad_tets}")
        lines.append(f"    Tris  = {self.bad_tris}")
        if self.threshold_violations:
            lines.append("")
            lines.append("  Threshold violations:")
            for name, count in self.threshold_violations.items():
                lines.append(f"    {name}: {count}")
        lines.append("=" * 60)
        return "\n".join(lines)


class GmshQualityAnalyzer:
    """Analyze mesh quality from a live Gmsh model."""

    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        self.thresholds = thresholds or QualityThresholds()
        self.report = QualityReport()

    def analyze(self) -> QualityReport:
        """Run the full quality analysis and return a QualityReport."""
        self.report = QualityReport()
        self._collect_mesh_stats()
        self._analyze_volume_elements()
        self._analyze_surface_elements()
        self._compute_statistics()
        self._check_thresholds()
        return self.report

    def _collect_mesh_stats(self) -> None:
        """Collect basic mesh statistics from the current Gmsh model."""
        nodes, _, _ = gmsh.model.mesh.getNodes()
        self.report.num_nodes = len(nodes)

        elem_types, elem_tags, _ = gmsh.model.mesh.getElements()
        for etype, tags in zip(elem_types, elem_tags):
            if etype == _GMSH_TET:
                self.report.num_tets = len(tags)
            elif etype == _GMSH_TRI:
                self.report.num_tris = len(tags)

    def _analyze_volume_elements(self) -> None:
        """Analyze all tetrahedral elements."""
        if self.report.num_tets == 0:
            return
        elem_tags, elem_nodes = gmsh.model.mesh.getElementsByType(_GMSH_TET)
        node_ids, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)
        node_map = {int(n): i for i, n in enumerate(node_ids)}

        tet_qualities: List[float] = []
        volumes: List[float] = []
        bad_count = 0

        for j, tag in enumerate(elem_tags):
            npe = 4
            start = j * npe
            end = start + npe
            nids = [int(k) for k in elem_nodes[start:end]]
            pts = coords[[node_map[nid] for nid in nids]]
            eq = self._compute_tet_quality(tag, pts)
            tet_qualities.append(eq.quality)
            volumes.append(eq.volume)
            if eq.bad:
                bad_count += 1
            self.report.element_qualities.append(eq)

        if tet_qualities:
            self.report.tet_quality_min = float(np.min(tet_qualities))
            self.report.tet_quality_max = float(np.max(tet_qualities))
            self.report.tet_quality_mean = float(np.mean(tet_qualities))
        if volumes:
            self.report.volume_min = float(np.min(volumes))
            self.report.volume_max = float(np.max(volumes))
            self.report.volume_mean = float(np.mean(volumes))
        self.report.bad_tets = bad_count

    def _analyze_surface_elements(self) -> None:
        """Analyze all triangular surface elements."""
        if self.report.num_tris == 0:
            return
        elem_tags, elem_nodes = gmsh.model.mesh.getElementsByType(_GMSH_TRI)
        node_ids, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)
        node_map = {int(n): i for i, n in enumerate(node_ids)}

        tri_qualities: List[float] = []
        areas: List[float] = []
        bad_count = 0

        for j, tag in enumerate(elem_tags):
            npe = 3
            start = j * npe
            end = start + npe
            nids = [int(k) for k in elem_nodes[start:end]]
            pts = coords[[node_map[nid] for nid in nids]]
            eq = self._compute_tri_quality(tag, pts)
            tri_qualities.append(eq.quality)
            areas.append(eq.area)
            if eq.bad:
                bad_count += 1
            self.report.element_qualities.append(eq)

        if tri_qualities:
            self.report.tri_quality_min = float(np.min(tri_qualities))
            self.report.tri_quality_max = float(np.max(tri_qualities))
            self.report.tri_quality_mean = float(np.mean(tri_qualities))
        if areas:
            self.report.area_min = float(np.min(areas))
            self.report.area_max = float(np.max(areas))
            self.report.area_mean = float(np.mean(areas))
        self.report.bad_tris = bad_count

    def _compute_tet_quality(self, tag: int, pts: np.ndarray) -> ElementQuality:
        """Compute quality metrics for a single tetrahedron."""
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
        mean_edge = float(np.mean(edge_lengths))
        aspect_ratio = max_edge / min_edge if min_edge > 1e-12 else float("inf")

        # Face areas
        faces = [
            np.cross(pts[1] - pts[0], pts[2] - pts[0]),
            np.cross(pts[1] - pts[0], pts[3] - pts[0]),
            np.cross(pts[2] - pts[0], pts[3] - pts[0]),
            np.cross(pts[1] - pts[2], pts[3] - pts[2]),
        ]
        face_areas = [0.5 * np.linalg.norm(fa) for fa in faces]
        total_face_area = sum(face_areas)

        # Inradius / Circumradius (radius ratio / gamma)
        inradius = 3.0 * vol / total_face_area if total_face_area > 1e-12 else 0.0
        circumradius = max_edge / (2.0 * np.sqrt(6.0) / 3.0) if max_edge > 1e-12 else 0.0
        gamma = inradius / circumradius if circumradius > 1e-12 else 0.0

        # SICN
        ref_vol = mean_edge ** 3 / (6.0 * np.sqrt(2.0)) if mean_edge > 1e-12 else 0.0
        sicn = vol / ref_vol if ref_vol > 1e-12 else 0.0

        # SJ (scaled Jacobian) — min over 4 nodes
        sj_vals = []
        for i in range(4):
            others = [k for k in range(4) if k != i]
            e1 = pts[others[0]] - pts[i]
            e2 = pts[others[1]] - pts[i]
            e3 = pts[others[2]] - pts[i]
            detJ = abs(np.dot(e1, np.cross(e2, e3)))
            le1, le2, le3 = np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)
            if le1 > 1e-12 and le2 > 1e-12 and le3 > 1e-12:
                sj_vals.append(detJ / (le1 * le2 * le3))
        sj = min(sj_vals) if sj_vals else 0.0

        # SIGE (generalized scaled Jacobian)
        edge_sum = sum(edge_lengths)
        gen_ref_vol = (mean_edge / 2.0) ** 3 / (3.0 * np.sqrt(2.0)) if mean_edge > 1e-12 else 0.0
        sige = vol / gen_ref_vol if gen_ref_vol > 1e-12 else 0.0

        # Isotropy
        ideal_vol = (total_face_area / (6.0 * np.sqrt(3.0))) ** 1.5 / (9.0 * np.sqrt(2.0)) if total_face_area > 1e-12 else 0.0
        isotropy = vol / ideal_vol if ideal_vol > 1e-12 else 0.0

        quality = gamma
        bad = False
        reasons: List[str] = []
        if vol < self.thresholds.min_volume:
            bad = True
            reasons.append("small_volume")
        if aspect_ratio > self.thresholds.max_aspect_ratio:
            bad = True
            reasons.append("high_aspect_ratio")
        if sicn < self.thresholds.min_sicn:
            bad = True
            reasons.append("low_sicn")
        if sj < self.thresholds.min_sj:
            bad = True
            reasons.append("low_sj")
        if sige < self.thresholds.min_sige:
            bad = True
            reasons.append("low_sige")
        if gamma < self.thresholds.gamma_min:
            bad = True
            reasons.append("low_gamma")
        if quality < self.thresholds.min_tet_quality:
            bad = True
            reasons.append("low_quality")

        return ElementQuality(
            element_id=tag,
            element_type=_GMSH_TET,
            quality=quality,
            volume=vol,
            aspect_ratio=aspect_ratio,
            sicn=sicn,
            sj=sj,
            sige=sige,
            gamma=gamma,
            inner_radius=inradius,
            outer_radius=circumradius,
            isotropy=isotropy,
            bad=bad,
            bad_reasons=reasons,
        )

    def _compute_tri_quality(self, tag: int, pts: np.ndarray) -> ElementQuality:
        """Compute quality metrics for a single triangle."""
        a = np.linalg.norm(pts[1] - pts[0])
        b = np.linalg.norm(pts[2] - pts[1])
        c = np.linalg.norm(pts[0] - pts[2])
        # FIX: use norm of cross product, not sum of components
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        max_edge = max(a, b, c)
        min_edge = min(a, b, c)
        mean_edge = float(np.mean([a, b, c]))
        aspect_ratio = max_edge / min_edge if min_edge > 1e-12 else float("inf")
        angles: List[float] = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            cos_angle = np.dot(v1, v2) / (n1 * n2 + 1e-12) if n1 > 1e-12 and n2 > 1e-12 else 1.0
            angles.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

        # Inradius / Circumradius
        s = (a + b + c) / 2.0
        inradius = area / s if s > 1e-12 else 0.0
        circumradius = a * b * c / (4.0 * area) if area > 1e-12 else 0.0
        gamma = inradius / circumradius if circumradius > 1e-12 else 0.0

        # SICN = r/R = 8*A^2 / ((a+b+c)*a*b*c)
        perimeter = a + b + c
        sicn = 8.0 * area * area / (perimeter * a * b * c) if perimeter > 1e-12 and a * b * c > 1e-12 else 0.0

        # SJ (scaled Jacobian) — min over 3 nodes
        sj_vals = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            e1 = pts[j] - pts[i]
            e2 = pts[k] - pts[i]
            le1, le2 = np.linalg.norm(e1), np.linalg.norm(e2)
            cross = np.cross(e1, e2)
            detJ = np.linalg.norm(cross)
            if le1 > 1e-12 and le2 > 1e-12:
                sj_vals.append(detJ / (le1 * le2))
        sj = min(sj_vals) if sj_vals else 0.0

        # SIGE
        sige = sicn

        # Isotropy = area / (sqrt(3)/4 * max_edge^2)
        ideal_area = np.sqrt(3.0) / 4.0 * max_edge * max_edge if max_edge > 1e-12 else 0.0
        isotropy = area / ideal_area if ideal_area > 1e-12 else 0.0

        quality = area / (max_edge ** 2) if max_edge > 1e-12 else 0.0
        bad = False
        reasons: List[str] = []
        if area < self.thresholds.min_area:
            bad = True
            reasons.append("small_area")
        if aspect_ratio > self.thresholds.max_aspect_ratio:
            bad = True
            reasons.append("high_aspect_ratio")
        if sicn < self.thresholds.min_sicn:
            bad = True
            reasons.append("low_sicn")
        if sj < self.thresholds.min_sj:
            bad = True
            reasons.append("low_sj")
        if sige < self.thresholds.min_sige:
            bad = True
            reasons.append("low_sige")
        if gamma < self.thresholds.gamma_min:
            bad = True
            reasons.append("low_gamma")
        if quality < self.thresholds.min_tri_quality:
            bad = True
            reasons.append("low_quality")

        return ElementQuality(
            element_id=tag,
            element_type=_GMSH_TRI,
            quality=quality,
            area=area,
            aspect_ratio=aspect_ratio,
            min_angle=min(angles),
            max_angle=max(angles),
            sicn=sicn,
            sj=sj,
            sige=sige,
            gamma=gamma,
            inner_radius=inradius,
            outer_radius=circumradius,
            isotropy=isotropy,
            bad=bad,
            bad_reasons=reasons,
        )

    def _compute_statistics(self) -> None:
        """Aggregate per-element qualities into report-level statistics."""
        tet_qualities = [eq.quality for eq in self.report.element_qualities if eq.element_type == _GMSH_TET]
        tri_qualities = [eq.quality for eq in self.report.element_qualities if eq.element_type == _GMSH_TRI]
        tet_volumes = [eq.volume for eq in self.report.element_qualities if eq.element_type == _GMSH_TET]
        tri_areas = [eq.area for eq in self.report.element_qualities if eq.element_type == _GMSH_TRI]

        if tet_qualities:
            self.report.tet_quality_min = float(np.min(tet_qualities))
            self.report.tet_quality_max = float(np.max(tet_qualities))
            self.report.tet_quality_mean = float(np.mean(tet_qualities))
        if tri_qualities:
            self.report.tri_quality_min = float(np.min(tri_qualities))
            self.report.tri_quality_max = float(np.max(tri_qualities))
            self.report.tri_quality_mean = float(np.mean(tri_qualities))
        if tet_volumes:
            self.report.volume_min = float(np.min(tet_volumes))
            self.report.volume_max = float(np.max(tet_volumes))
            self.report.volume_mean = float(np.mean(tet_volumes))
        if tri_areas:
            self.report.area_min = float(np.min(tri_areas))
            self.report.area_max = float(np.max(tri_areas))
            self.report.area_mean = float(np.mean(tri_areas))

    def _check_thresholds(self) -> None:
        """Count threshold violations across all analyzed elements."""
        stats: Dict[str, int] = {}
        for eq in self.report.element_qualities:
            for reason in eq.bad_reasons:
                stats[reason] = stats.get(reason, 0) + 1
        self.report.threshold_violations = stats
        self.report.bad_tets = sum(1 for eq in self.report.element_qualities if eq.element_type == _GMSH_TET and eq.bad)
        self.report.bad_tris = sum(1 for eq in self.report.element_qualities if eq.element_type == _GMSH_TRI and eq.bad)

    def export_bad_elements_vtk(self, path: str | Path) -> None:
        """Write only bad elements to a VTK file."""
        bad = [eq for eq in self.report.element_qualities if eq.bad]
        if not bad:
            Path(path).write_text("# No bad elements\n")
            return
        self._write_vtk(path, bad, scalar_name="bad_flag")

    def export_quality_vtk(self, path: str | Path) -> None:
        """Write all elements with quality scalar to a VTK file."""
        self._write_vtk(path, self.report.element_qualities, scalar_name="quality")

    def _write_vtk(self, path: str | Path, elements: List[ElementQuality], scalar_name: str) -> None:
        """Write elements to a VTK unstructured grid using pyvista."""
        try:
            import pyvista as pv
        except ImportError:
            Path(path).write_text("# pyvista not available\n")
            return

        node_ids, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)
        node_map = {int(n): i for i, n in enumerate(node_ids)}

        points = coords
        cells: List[int] = []
        cell_types: List[int] = []
        scalars: List[float] = []

        for eq in elements:
            etype = eq.element_type
            npe = 4 if etype == _GMSH_TET else 3
            vtk_type = _VTK_TETRA if etype == _GMSH_TET else _VTK_TRIANGLE

            tag = eq.element_id
            all_elem_tags, all_elem_nodes = gmsh.model.mesh.getElementsByType(etype)
            if tag not in all_elem_tags:
                continue
            idx = list(all_elem_tags).index(tag)
            start = idx * npe
            end = start + npe
            nids = [int(k) for k in all_elem_nodes[start:end]]
            local_nodes = [node_map[nid] for nid in nids]

            cells.append(npe)
            cells.extend(local_nodes)
            cell_types.append(vtk_type)
            scalars.append(eq.quality)

        if not cells:
            Path(path).write_text("# No elements to write\n")
            return

        grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)
        grid.cell_data[scalar_name] = np.array(scalars)
        grid.save(str(path))
        logger.info("Wrote VTK: %s (%d elements)", path, len(elements))
