#!/usr/bin/env python3
"""Focused benchmark on the main VMTK aorta test case.

Runs foampilot pipeline on aorta-surface-open-ends.stl and compares against
aorta-centerline.vtp from vmtk-test-data. Reports per-phase timings and
all geometric accuracy metrics.
"""
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import vtk
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "foampilot" / "src"))

from foampilot.geometry.topology.vmtk.vmtkcenterlines_python import (
    PipelineReport,
    run_pipeline,
)
from foampilot.geometry.topology.vmtk.vmtkcenterlinegeometry_local import Centerline


@dataclass
class PhaseTiming:
    preprocess: float = 0.0
    capping: float = 0.0
    delaunay: float = 0.0
    internal_tets: float = 0.0
    voronoi: float = 0.0
    poles: float = 0.0
    fast_marching: float = 0.0
    resampling: float = 0.0
    sections: float = 0.0
    network: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocess_s": self.preprocess,
            "capping_s": self.capping,
            "delaunay_s": self.delaunay,
            "internal_tets_s": self.internal_tets,
            "voronoi_s": self.voronoi,
            "poles_s": self.poles,
            "fast_marching_s": self.fast_marching,
            "resampling_s": self.resampling,
            "sections_s": self.sections,
            "network_s": self.network,
            "total_s": self.total,
        }


@dataclass
class QualityMetrics:
    n_points: int = 0
    length_mm: float = 0.0
    mean_radius_mm: float = 0.0
    mean_distance_mm: float = 0.0
    hausdorff_mm: float = 0.0
    length_error_pct: float = 0.0
    tortuosity_error_pct: float = 0.0
    radius_error_pct: float = 0.0
    anatomical_plausibility: str = "UNKNOWN"
    status: str = "PASS"
    voronoi_n_points: int = 0
    voronoi_n_edges: int = 0
    n_internal_tets: int = 0
    n_seed_component: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_points": self.n_points,
            "length_mm": self.length_mm,
            "mean_radius_mm": self.mean_radius_mm,
            "mean_distance_mm": self.mean_distance_mm,
            "hausdorff_mm": self.hausdorff_mm,
            "length_error_pct": self.length_error_pct,
            "tortuosity_error_pct": self.tortuosity_error_pct,
            "radius_error_pct": self.radius_error_pct,
            "anatomical_plausibility": self.anatomical_plausibility,
            "status": self.status,
            "voronoi_n_points": self.voronoi_n_points,
            "voronoi_n_edges": self.voronoi_n_edges,
            "n_internal_tets": self.n_internal_tets,
            "n_seed_component": self.n_seed_component,
        }


def read_vtp_points_radii(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    pd = reader.GetOutput()
    n = pd.GetNumberOfPoints()
    pts = np.array([pd.GetPoint(i) for i in range(n)], dtype=float)
    rad_arr = pd.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if rad_arr is not None:
        rad = np.array([rad_arr.GetTuple1(i) for i in range(n)], dtype=float)
    else:
        rad = np.full(n, np.nan, dtype=float)
    return pts, rad


def compute_centerline_length(cl: Centerline) -> float:
    if len(cl.points) < 2:
        return 0.0
    seg = np.diff(cl.points, axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def compute_tortuosity(cl: Centerline) -> float:
    length = compute_centerline_length(cl)
    if length <= 0 or len(cl.points) < 2:
        return 0.0
    chord = float(np.linalg.norm(cl.points[-1] - cl.points[0]))
    return length / max(chord, 1e-12)


def compute_tortuosity_from_points(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    return length / max(chord, 1e-12)


def run_benchmark(input_path: Path, ref_path: Path) -> Tuple[PhaseTiming, QualityMetrics]:
    t0 = time.perf_counter()
    cl, report = run_pipeline(
        input_path=input_path,
        backend="python_fmm",
        acceleration="numpy",
        resampling_step=1.0,
    )
    total = time.perf_counter() - t0

    phase_timings = PhaseTiming(
        preprocess=float(report.phase_timings.get("preprocess", 0.0)),
        capping=float(report.phase_timings.get("capping", 0.0)),
        delaunay=float(report.phase_timings.get("delaunay", 0.0)),
        internal_tets=float(report.phase_timings.get("internal_tets", 0.0)),
        voronoi=float(report.phase_timings.get("voronoi", 0.0)),
        poles=float(report.phase_timings.get("poles", 0.0)),
        fast_marching=float(report.phase_timings.get("fast_marching", 0.0)),
        resampling=float(report.phase_timings.get("resampling", 0.0)),
        sections=float(report.phase_timings.get("sections", 0.0)),
        network=float(report.phase_timings.get("network", 0.0)),
        total=total,
    )

    warnings = list(report.warnings)
    for w in warnings:
        logger.warning("Pipeline warning: %s", w)

    if cl is None or len(cl.points) < 2:
        logger.error("Empty or invalid centerline")
        return phase_timings, QualityMetrics(
            status="ERROR",
        )

    metrics = QualityMetrics(
        n_points=len(cl.points),
        length_mm=compute_centerline_length(cl),
        mean_radius_mm=float(np.nanmean(cl.radii)) if len(cl.radii) > 0 else 0.0,
        voronoi_n_points=int(report.quality_metrics.get("voronoi_n_points", 0)),
        voronoi_n_edges=int(report.quality_metrics.get("voronoi_n_edges", 0)),
        n_internal_tets=int(report.quality_metrics.get("n_internal_tets", 0)),
        n_seed_component=int(report.quality_metrics.get("n_seed_component", 0)),
    )

    anatomical = "PASS"
    if metrics.mean_radius_mm < 0.5:
        anatomical = "FAIL: radius too small"
    elif metrics.length_mm < 50.0:
        anatomical = "FAIL: length too small"
    metrics.anatomical_plausibility = anatomical
    if anatomical != "PASS":
        metrics.status = "FAIL"

    if ref_path.exists():
        try:
            ref_pts, ref_rad = read_vtp_points_radii(ref_path)
            if len(ref_pts) < 2:
                raise ValueError("Reference has fewer than 2 points")

            test_pts = cl.points
            test_rad = cl.radii

            tree_ref = cKDTree(ref_pts)
            tree_test = cKDTree(test_pts)
            dists_ref_to_test, _ = tree_test.query(ref_pts, k=1)
            dists_test_to_ref, _ = tree_ref.query(test_pts, k=1)
            mean_dist = 0.5 * (float(np.mean(dists_ref_to_test)) + float(np.mean(dists_test_to_ref)))
            sym_hausdorff = float(max(np.max(dists_ref_to_test), np.max(dists_test_to_ref)))

            ref_len = float(np.sum(np.linalg.norm(np.diff(ref_pts, axis=0), axis=1)))
            test_len = metrics.length_mm
            length_err = abs(test_len - ref_len) / max(ref_len, 1e-12)

            ref_tort = compute_tortuosity_from_points(ref_pts)
            test_tort = compute_tortuosity(cl)
            tort_err = abs(test_tort - ref_tort) / max(ref_tort, 1e-12)

            ref_mean_rad = float(np.nanmean(ref_rad))
            test_mean_rad = metrics.mean_radius_mm
            rad_err = abs(test_mean_rad - ref_mean_rad) / max(ref_mean_rad, 1e-12)

            metrics.mean_distance_mm = mean_dist
            metrics.hausdorff_mm = sym_hausdorff
            metrics.length_error_pct = length_err * 100.0
            metrics.tortuosity_error_pct = tort_err * 100.0
            metrics.radius_error_pct = rad_err * 100.0

            logger.info(
                "Reference comparison: mean_dist=%.3f mm, hausdorff=%.3f mm, len_err=%.1f%%, rad_err=%.1f%%",
                mean_dist, sym_hausdorff, length_err * 100, rad_err * 100,
            )
        except Exception as exc:
            logger.error("Reference comparison failed: %s", exc, exc_info=True)
            warnings.append(f"Reference comparison failed: {exc}")
            metrics.status = "ERROR"
    else:
        logger.warning("No reference found at %s", ref_path)

    return phase_timings, metrics


def print_report(case_name: str, timings: PhaseTiming, metrics: QualityMetrics) -> None:
    print("\n" + "=" * 100)
    print(f"CASE: {case_name}")
    print("=" * 100)
    print("PER-PHASE TIMINGS")
    print("-" * 100)
    for phase, secs in timings.to_dict().items():
        print(f"  {phase:<24} {secs:10.3f} s")
    print("-" * 100)
    print("GEOMETRIC ACCURACY vs VMTK REFERENCE")
    print("-" * 100)
    print(f"  {'n_points':<24} {metrics.n_points:>10}")
    print(f"  {'length_mm':<24} {metrics.length_mm:>10.3f}")
    print(f"  {'mean_radius_mm':<24} {metrics.mean_radius_mm:>10.3f}")
    print(f"  {'mean_distance_mm':<24} {metrics.mean_distance_mm:>10.3f}")
    print(f"  {'hausdorff_mm':<24} {metrics.hausdorff_mm:>10.3f}")
    print(f"  {'length_error_pct':<24} {metrics.length_error_pct:>10.2f}")
    print(f"  {'tortuosity_error_pct':<24} {metrics.tortuosity_error_pct:>10.2f}")
    print(f"  {'radius_error_pct':<24} {metrics.radius_error_pct:>10.2f}")
    print(f"  {'anatomical_plausibility':<24} {metrics.anatomical_plausibility:>10}")
    print(f"  {'status':<24} {metrics.status:>10}")
    print(f"  {'voronoi_n_points':<24} {metrics.voronoi_n_points:>10}")
    print(f"  {'voronoi_n_edges':<24} {metrics.voronoi_n_edges:>10}")
    print(f"  {'n_internal_tets':<24} {metrics.n_internal_tets:>10}")
    print(f"  {'n_seed_component':<24} {metrics.n_seed_component:>10}")
    print("=" * 100)


def main() -> int:
    input_path = ROOT / "test" / "vmtk-test-data" / "input" / "aorta-surface-open-ends.stl"
    ref_path = ROOT / "test" / "vmtk-test-data" / "input" / "aorta-centerline.vtp"

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return 1
    if not ref_path.exists():
        logger.error("Reference not found: %s", ref_path)
        return 1

    logger.info("Running benchmark on %s", input_path.name)
    timings, metrics = run_benchmark(input_path, ref_path)
    print_report("aorta-surface-open-ends", timings, metrics)

    report = {
        "case": "aorta-surface-open-ends",
        "input": str(input_path),
        "reference": str(ref_path),
        "timings": timings.to_dict(),
        "metrics": metrics.to_dict(),
    }
    out = Path("/tmp/vmtk_benchmark_aorta.json")
    out.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", out)

    return 0 if metrics.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
