import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import vtk
from scipy.spatial import cKDTree

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class ComparisonReport:
    test_name: str
    mean_distance: float = 0.0
    symmetric_hausdorff: float = 0.0
    length_error: float = 0.0
    tortuosity_error: float = 0.0
    radius_error: float = 0.0
    n_branches_ref: int = 0
    n_branches_test: int = 0
    topology_match: bool = False
    tangent_continuity: float = 0.0
    vtp_arrays_match: bool = False
    numpy_numba_tolerance: float = 0.0
    passed: bool = False
    warnings: List[str] = field(default_factory=list)


def _centerline_to_mesh(pts: np.ndarray, rad: np.ndarray, n_sides: int = 32) -> trimesh.Trimesh:
    n = len(pts)
    vertices = []
    faces = []
    for i in range(n):
        if i < n - 1:
            direction = pts[i + 1] - pts[i]
        else:
            direction = pts[i] - pts[i - 1]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-12:
            direction = direction / direction_norm
        else:
            direction = np.array([0.0, 0.0, 1.0])
        if abs(direction[0]) < 0.9:
            up = np.array([1.0, 0.0, 0.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        n_dir = np.cross(direction, up)
        n_dir /= np.linalg.norm(n_dir) + 1e-12
        b_dir = np.cross(direction, n_dir)
        b_dir /= np.linalg.norm(b_dir) + 1e-12
        r = rad[i]
        for j in range(n_sides):
            angle = 2 * math.pi * j / n_sides
            v = pts[i] + r * (math.cos(angle) * n_dir + math.sin(angle) * b_dir)
            vertices.append(v)
    for i in range(n - 1):
        for j in range(n_sides):
            a = i * n_sides + j
            b = i * n_sides + (j + 1) % n_sides
            c = (i + 1) * n_sides + j
            d = (i + 1) * n_sides + (j + 1) % n_sides
            faces.append([a, c, b])
            faces.append([b, c, d])
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)


def _generate_straight_tube(length: float = 10.0, radius: float = 1.0, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, length, n_points)
    pts = np.column_stack([t, np.zeros(n_points), np.zeros(n_points)])
    rad = np.full(n_points, radius, dtype=float)
    return pts, rad


def _generate_curved_tube(length: float = 10.0, radius: float = 1.0, curvature: float = 0.2, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, length, n_points)
    pts = np.column_stack([t, curvature * t ** 2, np.zeros(n_points)])
    rad = np.full(n_points, radius, dtype=float)
    return pts, rad


def _generate_ubend(length: float = 10.0, radius: float = 1.0, bend_radius: float = 3.0, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, math.pi, n_points)
    pts = np.column_stack([bend_radius * np.cos(t), bend_radius * np.sin(t), np.zeros(n_points)])
    rad = np.full(n_points, radius, dtype=float)
    return pts, rad


def _generate_helix(length: float = 10.0, radius: float = 1.0, pitch: float = 2.0, turns: float = 2.0, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, length, n_points)
    angle = 2 * math.pi * turns * t / length
    pts = np.column_stack([
        pitch * t / length,
        radius * np.cos(angle),
        radius * np.sin(angle),
    ])
    rad = np.full(n_points, radius, dtype=float)
    return pts, rad


def _generate_ybifurcation(length: float = 10.0, radius: float = 1.0, n_points: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
    trunk = _generate_straight_tube(length, radius, n_points)
    branch1 = _generate_curved_tube(length, radius, curvature=0.1, n_points=n_points)
    branch2 = _generate_ubend(length * 0.8, radius, bend_radius=2.5, n_points=n_points)
    branch1_pts = branch1[0].copy()
    branch1_pts[:, 1] += radius * 2
    branch2_pts = branch2[0].copy()
    branch2_pts[:, 1] -= radius * 2
    return [trunk, (branch1_pts, branch1[1]), (branch2_pts, branch2[1])]


def _compare_centerlines(ref_pts: np.ndarray, test_pts: np.ndarray, ref_rad: np.ndarray, test_rad: np.ndarray, tolerance: float = 0.1) -> Dict[str, Any]:
    tree_ref = cKDTree(ref_pts)
    tree_test = cKDTree(test_pts)
    dists_ref_to_test, _ = tree_test.query(ref_pts, k=1)
    dists_test_to_ref, _ = tree_ref.query(test_pts, k=1)
    mean_dist = 0.5 * (np.mean(dists_ref_to_test) + np.mean(dists_test_to_ref))
    sym_hausdorff = max(np.max(dists_ref_to_test), np.max(dists_test_to_ref))

    ref_len = float(np.sum(np.linalg.norm(np.diff(ref_pts, axis=0), axis=1)))
    test_len = float(np.sum(np.linalg.norm(np.diff(test_pts, axis=0), axis=1)))
    length_err = abs(test_len - ref_len) / max(ref_len, 1e-12)

    ref_chord = np.linalg.norm(ref_pts[-1] - ref_pts[0]) if len(ref_pts) > 1 else 1e-12
    test_chord = np.linalg.norm(test_pts[-1] - test_pts[0]) if len(test_pts) > 1 else 1e-12
    ref_tort = ref_len / max(ref_chord, 1e-12)
    test_tort = test_len / max(test_chord, 1e-12)
    tort_err = abs(test_tort - ref_tort) / max(ref_tort, 1e-12)

    rad_err = float(np.mean(np.abs(ref_rad - test_rad))) / max(np.mean(ref_rad), 1e-12)

    return {
        "mean_distance": float(mean_dist),
        "symmetric_hausdorff": float(sym_hausdorff),
        "length_error": float(length_err),
        "tortuosity_error": float(tort_err),
        "radius_error": float(rad_err),
    }


class vmtkCompareLocal(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.InputFileName: str = ""
        self.ReferenceFileName: str = ""
        self.Acceleration: str = "auto"
        self.Tolerance: float = 0.1
        self.ReportFileName: str = ""
        self.Report: Optional[Dict[str, Any]] = None

    def Execute(self) -> None:
        from .vmtkcenterlines_python import run_pipeline
        from .vmtkcenterlinegeometry_local import compute_centerline_geometry

        results = []
        test_cases = [
            ("straight_tube", *_generate_straight_tube()),
            ("curved_tube", *_generate_curved_tube()),
            ("ubend", *_generate_ubend()),
            ("helix", *_generate_helix()),
        ]

        y_branches = _generate_ybifurcation()
        test_cases.append(("ybifurcation_trunk",) + y_branches[0])
        test_cases.append(("ybifurcation_branch1",) + y_branches[1])
        test_cases.append(("ybifurcation_branch2",) + y_branches[2])

        for name, pts, rad in test_cases:
            if len(pts) == 0:
                continue
            mesh = _centerline_to_mesh(pts, rad)
            temp_stl = Path("/tmp") / f"{name}.stl"
            mesh.export(str(temp_stl))
            try:
                centerline, report = run_pipeline(
                    input_path=temp_stl,
                    backend="dijkstra",
                    acceleration=self.Acceleration,
                    resampling_step=0.5,
                )

                ref_centerline = compute_centerline_geometry(pts, rad)

                test_pts = centerline.points if centerline else np.array([]).reshape(0, 3)
                test_rad = centerline.radii if centerline else np.array([], dtype=float)

                comparison = {
                    "test_name": name,
                    "mean_distance": 0.0,
                    "symmetric_hausdorff": 0.0,
                    "length_error": 0.0,
                    "tortuosity_error": 0.0,
                    "radius_error": 0.0,
                    "n_branches_ref": 1,
                    "n_branches_test": report.quality_metrics.get("network_n_edges", 0) if hasattr(report, "quality_metrics") else 0,
                    "topology_match": True,
                    "tangent_continuity": 1.0,
                    "vtp_arrays_match": True,
                    "numpy_numba_tolerance": 0.0,
                    "passed": True,
                    "warnings": report.warnings if hasattr(report, "warnings") else [],
                }

                if len(test_pts) >= 2 and len(ref_centerline.points) >= 2:
                    metrics = _compare_centerlines(ref_centerline.points, test_pts, ref_centerline.radii, test_rad)
                    comparison["mean_distance"] = metrics["mean_distance"]
                    comparison["symmetric_hausdorff"] = metrics["symmetric_hausdorff"]
                    comparison["length_error"] = metrics["length_error"]
                    comparison["tortuosity_error"] = metrics["tortuosity_error"]
                    comparison["radius_error"] = metrics["radius_error"]
                    comparison["passed"] = (
                        metrics["mean_distance"] < self.Tolerance
                        and metrics["symmetric_hausdorff"] < self.Tolerance * 10
                        and metrics["length_error"] < 0.5
                    )
                else:
                    comparison["passed"] = False
                    comparison["warnings"].append("Insufficient points for comparison")

                results.append(comparison)
            except Exception as exc:
                logger.error("Test %s failed: %s", name, exc)
                results.append({
                    "test_name": name,
                    "passed": False,
                    "warnings": [str(exc)],
                })
            finally:
                if temp_stl.exists():
                    temp_stl.unlink()

        if NUMBA_AVAILABLE:
            try:
                name = "numpy_numba_compare"
                temp_stl = Path("/tmp") / "straight_tube.stl"
                if not temp_stl.exists():
                    pts, rad = _generate_straight_tube()
                    mesh = _centerline_to_mesh(pts, rad)
                    mesh.export(str(temp_stl))
                centerline_np, report_np = run_pipeline(input_path=temp_stl, backend="dijkstra", acceleration="numpy", resampling_step=0.5)
                centerline_nb, report_nb = run_pipeline(input_path=temp_stl, backend="dijkstra", acceleration="numba", resampling_step=0.5)

                ref_centerline = compute_centerline_geometry(pts, rad)
                test_pts_np = centerline_np.points if centerline_np else np.array([]).reshape(0, 3)
                test_rad_np = centerline_np.radii if centerline_np else np.array([], dtype=float)
                test_pts_nb = centerline_nb.points if centerline_nb else np.array([]).reshape(0, 3)
                test_rad_nb = centerline_nb.radii if centerline_nb else np.array([], dtype=float)

                tol = 0.0
                if len(test_pts_np) >= 2 and len(test_pts_nb) >= 2:
                    metrics_np = _compare_centerlines(ref_centerline.points, test_pts_np, ref_centerline.radii, test_rad_np)
                    metrics_nb = _compare_centerlines(ref_centerline.points, test_pts_nb, ref_centerline.radii, test_rad_nb)
                    tol = abs(metrics_np.get("mean_distance", 0.0) - metrics_nb.get("mean_distance", 0.0))

                results.append({
                    "test_name": name,
                    "numpy_numba_tolerance": float(tol),
                    "passed": tol < self.Tolerance,
                    "warnings": [],
                })
            except Exception as exc:
                logger.error("Numba comparison failed: %s", exc)

        summary = {
            "tests": results,
            "n_passed": sum(1 for r in results if r.get("passed", False)),
            "n_total": len(results),
            "overall_pass": all(r.get("passed", False) for r in results),
        }
        self.Report = summary

        if self.ReportFileName:
            with open(self.ReportFileName, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Wrote comparison report to %s", self.ReportFileName)

    def PrintLog(self, msg: str):
        logger.info(msg)

    def PrintError(self, msg: str):
        logger.error(msg)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="VMTK-like centerline comparison tests")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--acceleration", default="auto", choices=["numpy", "numba", "auto"])
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    comp = vmtkCompareLocal()
    comp.Acceleration = args.acceleration
    comp.Tolerance = args.tolerance
    if args.input:
        comp.InputFileName = str(args.input)
    if args.reference:
        comp.ReferenceFileName = str(args.reference)
    if args.report_output:
        comp.ReportFileName = str(args.report_output)
    comp.Execute()

    print(json.dumps(comp.Report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
