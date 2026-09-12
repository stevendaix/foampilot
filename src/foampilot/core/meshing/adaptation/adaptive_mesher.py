"""
Adaptive mesh improvement engine for Gmsh.

This module provides a loop that analyses mesh quality, diagnoses failure
modes, applies targeted improvement actions, and tracks the adaptation
history. It is intended for use inside building_aero CFD pre-processing
workflows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gmsh
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gmsh option name constants
# ---------------------------------------------------------------------------
ALGO_MAP = {
    "Delaunay": 1,
    "HXT": 9,
    "Frontal": 4,
    "MeshAdapt": 6,
}

ALGO3D_MAP = {
    "Delaunay": 1,
    "HXT": 9,
    "Frontal": 5,
    "MeshAdapt": 7,
}

OPTIMIZER_MAP = {
    "Gmsh": 1,
    "Netgen": 2,
    "Relocate3D": 3,
    "UntangleMeshGeometry": 4,
}

# Gmsh element-type codes
_GMSH_TET = 4
_GMSH_HEX = 5
_GMSH_TRI = 2
_GMSH_QUAD = 3


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    """Snapshot of mesh quality after a generate call."""

    num_cells: int = 0
    num_faces: int = 0
    min_element_size: float = 0.0
    max_element_size: float = 0.0
    min_face_area: float = 0.0
    max_face_area: float = 0.0
    min_dihedral_angle: float = 0.0
    max_dihedral_angle: float = 0.0
    max_non_ortho: float = 0.0
    num_faces_over_threshold: int = 0
    face_threshold: float = 70.0
    non_ortho_threshold: float = 70.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationRecord:
    """Single iteration record for the adaptation history."""

    iteration: int
    cells: int
    max_non_ortho: float
    faces_over_threshold: int
    action: str
    result: str


# ---------------------------------------------------------------------------
# Size field helpers
# ---------------------------------------------------------------------------
class SizeFieldManager:
    """Manage Gmsh background size fields."""

    def __init__(self) -> None:
        self._active: Optional[int] = None

    def clear(self) -> None:
        if self._active is not None:
            try:
                gmsh.model.mesh.removeSizeField(self._active)
            except Exception:
                pass
            self._active = None

    def add_distance(
        self,
        tags: List[int],
        dim: int = 2,
        target_size: float = 1.0,
        sampling: int = 100,
    ) -> int:
        field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field, "SurfacesList", tags)
        gmsh.model.mesh.field.setNumber(field, "Sampling", sampling)
        self._finalize(field, target_size)
        return field

    def add_threshold(
        self,
        source_field: int,
        min_threshold: float = 0.0,
        max_threshold: float = 10.0,
        min_size: float = 0.5,
        max_size: float = 5.0,
    ) -> int:
        field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field, "InField", source_field)
        gmsh.model.mesh.field.setNumber(field, "LcMin", min_size)
        gmsh.model.mesh.field.setNumber(field, "LcMax", max_size)
        gmsh.model.mesh.field.setNumber(field, "DistMin", min_threshold)
        gmsh.model.mesh.field.setNumber(field, "DistMax", max_threshold)
        self._finalize(field, max_size)
        return field

    def add_min(
        self,
        field_a: int,
        field_b: int,
        target_size: float = 1.0,
    ) -> int:
        field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(field, "FieldsList", [field_a, field_b])
        self._finalize(field, target_size)
        return field

    def add_math_eval(
        self,
        expression: str,
        target_size: float = 1.0,
    ) -> int:
        field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(field, "F", expression)
        self._finalize(field, target_size)
        return field

    def add_box(
        self,
        xmin: float,
        ymin: float,
        zmin: float,
        xmax: float,
        ymax: float,
        zmax: float,
        v_in: float,
        v_out: float,
        target_size: float = 1.0,
    ) -> int:
        field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field, "VIn", v_in)
        gmsh.model.mesh.field.setNumber(field, "VOut", v_out)
        gmsh.model.mesh.field.setNumber(field, "XMin", xmin)
        gmsh.model.mesh.field.setNumber(field, "YMin", ymin)
        gmsh.model.mesh.field.setNumber(field, "ZMin", zmin)
        gmsh.model.mesh.field.setNumber(field, "XMax", xmax)
        gmsh.model.mesh.field.setNumber(field, "YMax", ymax)
        gmsh.model.mesh.field.setNumber(field, "ZMax", zmax)
        self._finalize(field, target_size)
        return field

    def _finalize(self, field: int, target_size: float) -> None:
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        self._active = field


# ---------------------------------------------------------------------------
# Core adaptive improver
# ---------------------------------------------------------------------------
class AdaptiveMeshImprover:
    """Iterative adaptive mesher for Gmsh models.

    The improver wraps a live Gmsh model.  The caller is responsible for
    building geometry, assigning physical groups, and calling
    :pymeth:`iterate` (or :pymeth:`improve`) after the initial mesh has
    been generated.

    Example
    -------
    >>> improver = AdaptiveMeshImprover(max_iterations=8, cell_budget=500_000)
    >>> improver.iterate()
    >>> history = improver.export_history()
    """

    def __init__(
        self,
        max_iterations: int = 5,
        cell_budget: Optional[int] = None,
        face_ortho_threshold: float = 70.0,
        lc_min: float = 0.1,
        lc_max: float = 100.0,
        verbose: bool = True,
    ) -> None:
        self.max_iterations = max_iterations
        self.cell_budget = cell_budget
        self.face_ortho_threshold = face_ortho_threshold
        self.lc_min = lc_min
        self.lc_max = lc_max
        self.verbose = verbose

        self.size_fields = SizeFieldManager()
        self.history: List[AdaptationRecord] = []
        self.current_report: Optional[QualityReport] = None
        self.iteration = 0
        self.algorithm_2d: str = "MeshAdapt"
        self.algorithm_3d: str = "Delaunay"
        self.optimizer: str = "Gmsh"
        self._converged = False
        self._active_wall_field: Optional[int] = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def analyze(self) -> QualityReport:
        """Analyse the current mesh and return a :class:`QualityReport`.

        Uses Gmsh element statistics to compute cell counts, face counts,
        size ranges, and a proxy for non-orthogonality based on face
        normal / centroid deviation.
        """
        report = QualityReport(face_threshold=self.face_ortho_threshold)
        try:
            nodes, coords, _ = gmsh.model.mesh.getNodes()
            report.num_cells = gmsh.model.mesh.getElementsByType(_GMSH_TET)
            report.num_faces = gmsh.model.mesh.getElementsByType(_GMSH_TRI)

            element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()
            cell_tags = []
            face_tags = []
            for etype, tags in zip(element_types, element_tags):
                if etype in (_GMSH_TET, _GMSH_HEX):
                    cell_tags.extend(tags)
                elif etype in (_GMSH_TRI, _GMSH_QUAD):
                    face_tags.extend(tags)

            report.num_cells = len(cell_tags)
            report.num_faces = len(face_tags)

            coords = np.array(coords).reshape(-1, 3)
            node_map = {int(n): i for i, n in enumerate(nodes)}

            sizes: List[float] = []
            face_areas: List[float] = []
            face_normals: List[np.ndarray] = []

            for etype, tags, nodes_list in zip(element_types, element_tags, element_nodes):
                npe = {_GMSH_TRI: 3, _GMSH_QUAD: 4, _GMSH_TET: 4, _GMSH_HEX: 8}.get(etype, 0)
                if npe == 0:
                    continue
                for j, tag in enumerate(tags):
                    start = j * npe
                    end = start + npe
                    node_ids = [int(k) for k in nodes_list[start:end]]
                    pts = coords[[node_map[nid] for nid in node_ids]]
                    if etype in (_GMSH_TRI, _GMSH_QUAD):
                        area = self._face_area(pts)
                        face_areas.append(area)
                        face_normals.append(self._face_normal(pts))
                        sizes.append(area ** 0.5)
                    elif etype in (_GMSH_TET, _GMSH_HEX):
                        vol = abs(np.dot(
                            pts[0] - pts[-1],
                            np.cross(pts[1] - pts[-1], pts[2] - pts[-1]),
                        )) / (6.0 if etype == _GMSH_TET else 1.0)
                        sizes.append(vol ** (1.0 / 3.0))

            if sizes:
                report.min_element_size = min(sizes)
                report.max_element_size = max(sizes)
            if face_areas:
                report.min_face_area = min(face_areas)
                report.max_face_area = max(face_areas)

            if face_normals:
                dihedral_angles = self._dihedral_angles(face_normals)
                if dihedral_angles:
                    report.min_dihedral_angle = min(dihedral_angles)
                    report.max_dihedral_angle = max(dihedral_angles)
                    over = sum(1 for a in dihedral_angles if a > self.face_ortho_threshold)
                    report.num_faces_over_threshold = over

            report.max_non_ortho = self._compute_non_ortho_proxy()
        except Exception as exc:
            logger.warning("Mesh analysis failed: %s", exc)

        self.current_report = report
        return report

    def diagnose(self, report: Optional[QualityReport] = None) -> List[str]:
        """Return failure-mode tags for *report* (or the last analyse)."""
        if report is None:
            report = self.current_report or self.analyze()

        modes: List[str] = []

        if self.cell_budget is not None and report.num_cells > self.cell_budget:
            modes.append("over_budget")

        if report.num_faces_over_threshold > 0:
            modes.append("high_non_ortho")

        if report.max_element_size > self.lc_max * 2.0:
            modes.append("size_too_large")

        if report.min_element_size < self.lc_min * 0.5 and report.num_cells < (self.cell_budget or float("inf")):
            modes.append("size_too_small")

        if not modes and report.num_cells == 0:
            modes.append("empty_mesh")

        if not modes:
            modes.append("good")

        return modes

    def improve(
        self,
        action: str,
        report: Optional[QualityReport] = None,
    ) -> str:
        """Apply a single improvement *action* to the current Gmsh model.

        Returns a short result string describing what was applied.
        """
        if report is None:
            report = self.current_report or self.analyze()

        try:
            if action == "refine":
                return self._action_refine(report)
            if action == "derefine":
                return self._action_derefine(report)
            if action == "smooth":
                return self._action_smooth(report)
            if action == "relocate":
                return self._action_relocate(report)
            if action == "untangle":
                return self._action_untangle(report)
            if action == "change_algorithm":
                return self._action_change_algorithm(report)
            raise ValueError(f"Unknown action: {action}")
        except Exception as exc:
            logger.warning("Action %s failed: %s", action, exc)
            return f"failed: {exc}"

    def iterate(self) -> List[AdaptationRecord]:
        """Run the adaptation loop up to :attr:`max_iterations`.

        Each iteration:
        1. Analyses the mesh.
        2. Diagnoses failure modes.
        3. Picks the highest-priority improvement action.
        4. Applies it.
        5. Records the outcome.

        Returns the accumulated history.
        """
        for i in range(self.max_iterations):
            self.iteration = i + 1
            report = self.analyze()
            modes = self.diagnose(report)

            if "good" in modes and not any(m != "good" for m in modes):
                self._log(f"Iteration {self.iteration}: mesh acceptable, stopping.")
                self._converged = True
                break

            action = self._select_action(modes, report)
            result = self.improve(action, report)

            try:
                gmsh.model.mesh.generate(3)
            except Exception as exc:
                result = f"generate_failed: {exc}"

            record = AdaptationRecord(
                iteration=self.iteration,
                cells=report.num_cells,
                max_non_ortho=report.max_non_ortho,
                faces_over_threshold=report.num_faces_over_threshold,
                action=action,
                result=result,
            )
            self.history.append(record)
            self._log(
                "Iter %d: cells=%d maxNonOrtho=%.1f faces>70=%d action=%s -> %s",
                record.iteration,
                record.cells,
                record.max_non_ortho,
                record.faces_over_threshold,
                record.action,
                record.result,
            )

            if self.cell_budget is not None and report.num_cells > self.cell_budget:
                self._log("Cell budget exceeded; stopping iteration.")
                break

        return self.history

    def export_history(self, path: Optional[Path] = None) -> str:
        """Export adaptation history as CSV text.

        If *path* is given, the CSV is written to that file.
        Returns the CSV string.
        """
        header = "iteration,cells,maxNonOrtho,faces>70,action,result\n"
        rows = []
        for rec in self.history:
            rows.append(
                f"{rec.iteration},{rec.cells},{rec.max_non_ortho:.2f},"
                f"{rec.faces_over_threshold},{rec.action},{rec.result}"
            )
        csv_text = header + "\n".join(rows) + "\n"
        if path is not None:
            Path(path).write_text(csv_text)
        return csv_text

    # ------------------------------------------------------------------
    #  Size field setup helpers
    # ------------------------------------------------------------------
    def setup_distance_field(
        self,
        surface_tags: List[int],
        target_size: float = 1.0,
        sampling: int = 100,
    ) -> None:
        self.size_fields.clear()
        self.size_fields.add_distance(surface_tags, target_size=target_size, sampling=sampling)

    def setup_threshold_field(
        self,
        source_field: int,
        min_threshold: float = 0.0,
        max_threshold: float = 10.0,
        min_size: float = 0.5,
        max_size: float = 5.0,
    ) -> None:
        self.size_fields.add_threshold(source_field, min_threshold, max_threshold, min_size, max_size)

    def setup_math_eval_field(self, expression: str, target_size: float = 1.0) -> None:
        self.size_fields.add_math_eval(expression, target_size)

    def setup_box_field(
        self,
        xmin: float,
        ymin: float,
        zmin: float,
        xmax: float,
        ymax: float,
        zmax: float,
        v_in: float,
        v_out: float,
        target_size: float = 1.0,
    ) -> None:
        self.size_fields.add_box(xmin, ymin, zmin, xmax, ymax, zmax, v_in, v_out, target_size)

    def setup_min_field(self, field_a: int, field_b: int, target_size: float = 1.0) -> None:
        self.size_fields.add_min(field_a, field_b, target_size)

    def clear_size_fields(self) -> None:
        self.size_fields.clear()
        if self._active_wall_field is not None:
            try:
                gmsh.model.mesh.removeSizeField(self._active_wall_field)
            except Exception:
                pass
            self._active_wall_field = None

    # ------------------------------------------------------------------
    #  Algorithm / optimizer setters
    # ------------------------------------------------------------------
    def set_algorithm(self, algo_2d: str, algo_3d: str) -> None:
        self.algorithm_2d = algo_2d
        self.algorithm_3d = algo_3d

    def set_optimizer(self, optimizer: str) -> None:
        self.optimizer = optimizer

    def apply_mesh_options(self) -> None:
        """Push current algorithm / optimizer choices to Gmsh."""
        gmsh.option.setNumber("Mesh.Algorithm", ALGO_MAP.get(self.algorithm_2d, 6))
        gmsh.option.setNumber("Mesh.Algorithm3D", ALGO3D_MAP.get(self.algorithm_3d, 1))
        gmsh.option.setNumber("Mesh.Optimize", OPTIMIZER_MAP.get(self.optimizer, 1))
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)
        gmsh.option.setNumber("Mesh.QualityType", 2)
        gmsh.option.setNumber("Mesh.Smoothing", 1)
        gmsh.option.setNumber("Mesh.SmoothNormals", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.lc_max)

    def setup_wall_protection(
        self,
        wall_surface_tags: List[int],
        protect_distance: float = 5.0,
        protect_size: float = 1.0,
    ) -> None:
        field_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_dist, "SurfacesList", wall_surface_tags)
        gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 100)
        field_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
        gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", protect_distance)
        gmsh.model.mesh.field.setNumber(field_thresh, "LcMin", protect_size)
        gmsh.model.mesh.field.setNumber(field_thresh, "LcMax", protect_size)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_thresh)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        self._active_wall_field = field_thresh

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str, *args: Any) -> None:
        if self.verbose:
            logger.info(msg, *args)

    @staticmethod
    def _face_area(pts: np.ndarray) -> float:
        if len(pts) == 3:
            return 0.5 * abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]).sum())
        normal = np.zeros(3)
        for i in range(1, len(pts) - 1):
            normal += np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
        return 0.5 * abs(normal.sum())

    @staticmethod
    def _face_normal(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 3:
            return np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n1 = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n2 = np.cross(pts[2] - pts[0], pts[3] - pts[0])
        return n1 + n2

    def _dihedral_angles(self, normals: List[np.ndarray], sample_size: int = 500) -> List[float]:
        angles = []
        n = len(normals)
        if n <= sample_size:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            import random
            indices = random.sample(range(n), sample_size)
            pairs = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
        for i, j in pairs:
            n1 = normals[i]
            n2 = normals[j]
            norm1 = np.linalg.norm(n1)
            norm2 = np.linalg.norm(n2)
            if norm1 > 1e-12 and norm2 > 1e-12:
                cos_angle = np.dot(n1, n2) / (norm1 * norm2)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angles.append(math.degrees(math.acos(abs(cos_angle))))
        return angles

    def _compute_non_ortho_proxy(self, sample_size: int = 500) -> float:
        normals: List[np.ndarray] = []
        element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()
        all_tri_nodes = []
        for etype, tags, nodes_list in zip(element_types, element_tags, element_nodes):
            if etype not in (_GMSH_TRI, _GMSH_QUAD):
                continue
            npe = 3 if etype == _GMSH_TRI else 4
            all_tri_nodes.append((npe, tags, nodes_list))

        total_tris = sum(len(tags) for _, tags, _ in all_tri_nodes)
        if total_tris == 0:
            return 0.0

        sample = min(sample_size, total_tris)
        step = max(1, total_tris // sample)
        sampled = 0
        for npe, tags, nodes_list in all_tri_nodes:
            for j in range(0, len(tags), step):
                if sampled >= sample_size:
                    break
                start = j * npe
                end = start + npe
                node_ids = [int(k) for k in nodes_list[start:end]]
                _, coords, _ = gmsh.model.mesh.getNodes(node_ids, includeBoundary=True)
                pts = np.array(coords).reshape(-1, 3)
                normals.append(self._face_normal(pts))
                sampled += 1
            if sampled >= sample_size:
                break

        if len(normals) < 2:
            return 0.0

        angles = self._dihedral_angles(normals, sample_size=min(sample_size, len(normals)))
        return max(angles) if angles else 0.0

    # ------------------------------------------------------------------
    #  Action implementations
    # ------------------------------------------------------------------
    def _action_refine(self, report: QualityReport) -> str:
        self.lc_max = max(self.lc_min, self.lc_max * 0.7)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.lc_max)
        if self._active_wall_field is not None:
            return f"refine: lc_max->{self.lc_max:.3f} (wall-protected)"
        return f"refine: lc_max->{self.lc_max:.3f}"

    def _action_derefine(self, report: QualityReport) -> str:
        if self.cell_budget is not None and report.num_cells > self.cell_budget:
            self.lc_max = min(self.lc_max * 1.5, self.lc_max * 2.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.lc_max)
            return f"derefine: lc_max->{self.lc_max:.3f}"
        return "derefine: skipped (within budget)"

    def _action_smooth(self, report: QualityReport) -> str:
        gmsh.option.setNumber("Mesh.Smoothing", 2)
        gmsh.option.setNumber("Mesh.SmoothNormals", 1)
        return "smooth: increased smoothing"

    def _action_relocate(self, report: QualityReport) -> str:
        gmsh.model.mesh.optimize("Relocate3D", niter=2)
        return "relocate: Relocate3D optimizer applied"

    def _action_untangle(self, report: QualityReport) -> str:
        gmsh.model.mesh.optimize("UntangleMeshGeometry", niter=2)
        return "untangle: UntangleMeshGeometry optimizer applied"

    def _action_change_algorithm(self, report: QualityReport) -> str:
        candidates_3d = ["Delaunay", "HXT", "Frontal", "MeshAdapt"]
        candidates_2d = ["MeshAdapt", "Frontal", "Delaunay"]
        current_3d = self.algorithm_3d
        next_3d = next((c for c in candidates_3d if c != current_3d), "HXT")
        self.set_algorithm(self.algorithm_2d, next_3d)
        self.apply_mesh_options()
        return f"algorithm: {self.algorithm_2d}/{next_3d}"

    def _select_action(self, modes: List[str], report: QualityReport) -> str:
        if "over_budget" in modes:
            return "derefine"
        if "high_non_ortho" in modes:
            return "relocate"
        if "size_too_small" in modes:
            return "derefine"
        if "size_too_large" in modes:
            return "refine"
        if "empty_mesh" in modes:
            return "change_algorithm"
        return "smooth"
