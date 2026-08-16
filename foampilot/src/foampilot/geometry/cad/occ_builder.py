"""OpenCASCADE / Gmsh geometry builder for lofted CAD from sections."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import gmsh
import numpy as np
from geomdl import BSpline

from foampilot.geometry.cad.bspline_fitter import BSplineFitter
from foampilot.geometry.topology.section_extractor import Section

logger = logging.getLogger(__name__)


def _evaluate_curve(curve: BSpline.Curve, n_samples: int = 80) -> np.ndarray:
    curve.delta = 1.0 / max(n_samples - 1, 1)
    pts = np.array(curve.evalpts)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    return pts


class OCCBuilder:
    def __init__(self, n_samples: int = 40, mesh_size_factor: float = 1.0):
        self.n_samples = n_samples
        self.mesh_size_factor = mesh_size_factor
        self.fitter = BSplineFitter(degree=3, n_ctrl=12)

    def build_from_sections(self, sections: List[Section], case_dir: Optional[Path] = None) -> Dict:
        if len(sections) < 2:
            raise ValueError("At least two sections are required for lofting")
        gmsh.initialize()
        gmsh.model.add("cad_loft")
        try:
            return self._build(sections, case_dir=case_dir)
        finally:
            gmsh.model.occ.synchronize()

    def _build(self, sections: List[Section], case_dir: Optional[Path] = None) -> Dict:
        curves = []
        for i, section in enumerate(sections):
            pts2d = section.to_2d()
            try:
                curve = self.fitter.fit_section(pts2d)
            except Exception as exc:
                logger.warning("Section %d skipped: %s", i, exc)
                continue
            pts3d = section.center + self._to_3d(section, _evaluate_curve(curve, self.n_samples))
            curves.append(pts3d)

        result = {"sections": len(sections), "curves": len(curves)}

        if len(curves) < 2:
            result["volume_tag"] = -1
            result["method"] = "none"
            logger.warning("Not enough curve segments for lofting or sweeping")
            return result

        # Try lofting first
        for trial_curves in [curves, curves[::2], curves[1::2], curves[:4], curves[-4:], curves[:3], curves[-3:]]:
            if len(trial_curves) < 2:
                continue
            try:
                loft_tag = self._loft(trial_curves, name="loft")
                result["volume_tag"] = loft_tag
                result["method"] = "loft"
                logger.info("Loft succeeded with %d curve segments", len(trial_curves))
                if case_dir is not None:
                    of_dir = self._mesh_and_export(loft_tag, case_dir)
                    result["openfoam_dir"] = str(of_dir)
                return result
            except Exception as exc:
                logger.debug("Loft attempt with %d curves failed: %s", len(trial_curves), exc)
                continue

        # Fallback: sweep along centerline using addPipe
        logger.info("Loft failed, trying sweep (addPipe) along centerline...")
        try:
            sweep_tag = self._sweep_along_centerline(sections, curves, name="sweep")
            result["volume_tag"] = sweep_tag
            result["method"] = "sweep"
            logger.info("Sweep succeeded")
            if case_dir is not None:
                of_dir = self._mesh_and_export(sweep_tag, case_dir)
                result["openfoam_dir"] = str(of_dir)
            return result
        except Exception as exc:
            logger.warning("Sweep also failed: %s", exc)

        # Final fallback: direct STL export
        result["volume_tag"] = -1
        result["method"] = "direct_fallback"
        logger.warning("All loft/sweep methods failed, using direct STL export")
        return result

    def _to_3d(self, section: Section, pts2d: np.ndarray) -> np.ndarray:
        x, y, z = section.local_frame()
        return (pts2d[:, 0:1] * x + pts2d[:, 1:2] * y).reshape(-1, 3)

    def _loft(self, curves: List[np.ndarray], name: str) -> int:
        wire_tags = []
        for pts in curves:
            pts = np.asarray(pts, dtype=float)
            if len(pts) < 2:
                continue
            # Resample to a reasonable number of points for the BSpline
            n_pts = min(len(pts), 20)
            if len(pts) > n_pts:
                idx = np.linspace(0, len(pts) - 1, n_pts).astype(int)
                pts = pts[idx]
            point_tags = []
            for p in pts:
                point_tags.append(gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])))
            try:
                curve_tag = gmsh.model.occ.addBSpline(point_tags, degree=3)
                wire_tag = gmsh.model.occ.addWire([curve_tag])
                wire_tags.append(wire_tag)
            except Exception as exc:
                logger.warning("Wire creation failed: %s", exc)
                continue
        if len(wire_tags) < 2:
            raise RuntimeError(f"Not enough wires for loft: {name}")
        loft = gmsh.model.occ.addThruSections(wire_tags, makeSolid=True, makeRuled=False, smoothing=True)
        loft_tag = loft[0][1] if loft else -1
        gmsh.model.occ.synchronize()
        phys_tag = gmsh.model.addPhysicalGroup(3, [loft_tag])
        gmsh.model.setPhysicalName(3, phys_tag, name)
        return loft_tag

    def _sweep_along_centerline(self, sections: List[Section], curves: List[np.ndarray], name: str) -> int:
        """Sweep a profile curve along the centerline using addPipe."""
        if not sections:
            raise RuntimeError("No sections for sweep")

        # Create profile from the first section's 2D contour
        first_sec = sections[0]
        pts2d = first_sec.to_2d()
        if len(pts2d) < 3:
            raise RuntimeError("First section has too few points for profile")

        profile_tag = self._create_profile_curve(first_sec)

        # Create the path from all centerline points
        path_points = np.array([s.center for s in sections])
        if len(path_points) < 2:
            raise RuntimeError("Not enough centerline points for sweep path")

        path_tag = self._create_path_curve(path_points)

        gmsh.model.occ.synchronize()
        pipe_result = gmsh.model.occ.addPipe([(1, profile_tag)], [(1, path_tag)], makeSolid=True, trihedronMode=1)
        if not pipe_result:
            raise RuntimeError("addPipe returned no result")
        vol_tag = pipe_result[0][1]
        gmsh.model.occ.synchronize()

        phys_tag = gmsh.model.addPhysicalGroup(3, [vol_tag])
        gmsh.model.setPhysicalName(3, phys_tag, name)
        return vol_tag

    def _create_profile_curve(self, section: Section) -> int:
        """Create a BSpline curve from a section's 2D contour projected to 3D."""
        pts2d = section.to_2d()
        centroid = pts2d.mean(axis=0)
        angles = np.arctan2(pts2d[:, 1] - centroid[1], pts2d[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts2d = pts2d[order]

        pts3d = section.center + self._to_3d(section, pts2d)

        point_tags = []
        for p in pts3d:
            point_tags.append(gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])))

        curve_tag = gmsh.model.occ.addBSpline(point_tags, degree=min(3, len(point_tags) - 1))
        return curve_tag

    def _create_path_curve(self, path_points: np.ndarray) -> int:
        """Create a 3D BSpline curve through the given path points."""
        path_points = np.asarray(path_points, dtype=float)

        n_ctrl = min(len(path_points), 10)
        if len(path_points) > n_ctrl:
            idx = np.linspace(0, len(path_points) - 1, n_ctrl).astype(int)
            path_points = path_points[idx]

        point_tags = []
        for p in path_points:
            point_tags.append(gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])))

        curve_tag = gmsh.model.occ.addBSpline(point_tags, degree=3)
        return curve_tag

    def _mesh_and_export(self, vol_tag: int, case_dir: Path) -> Path:
        from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", self.mesh_size_factor)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen", niter=2)

        poly_dir = Path(case_dir) / "constant" / "polyMesh"
        poly_dir.mkdir(parents=True, exist_ok=True)

        exporter = DirectOpenFOAMExporter(case_dir)
        exporter.export_single_region(region_name="fluid")
        return poly_dir
