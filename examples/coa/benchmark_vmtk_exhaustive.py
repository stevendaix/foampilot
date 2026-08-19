#!/usr/bin/env python3
"""Exhaustive benchmark for foampilot VMTK centerline extraction pipeline.

Runs the full pipeline on every available surface in vmtk-test-data and
compares against VMTK reference centerlines when available. Reports per-phase
timings, geometric accuracy, and physical plausibility metrics.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "foampilot" / "src"))

from foampilot.geometry.topology.vmtk.vmtkcenterlines_python import (
    PipelineReport,
    run_pipeline,
)
from foampilot.geometry.topology.vmtk.vmtkcenterlinegeometry_local import Centerline


@dataclass
class CaseResult:
    case_name: str
    status: str
    input_path: str
    ref_path: Optional[str]
    n_points: int
    length_mm: float
    mean_radius_mm: float
    mean_distance_mm: float
    hausdorff_mm: float
    length_error_pct: float
    tortuosity_error_pct: float
    radius_error_pct: float
    total_time_s: float
    phase_timings: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    anatomical_plausibility: str = "UNKNOWN"


def read_vtp_points_radii(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read points and MaximumInscribedSphereRadius from a VTP file."""
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


def compute_metrics(
    test_cl: Centerline,
    ref_pts: np.ndarray,
    ref_rad: np.ndarray,
) -> Dict[str, Any]:
    test_pts = test_cl.points
    test_rad = test_cl.radii

    tree_ref = cKDTree(ref_pts)
    tree_test = cKDTree(test_pts)
    dists_ref_to_test, _ = tree_test.query(ref_pts, k=1)
    dists_test_to_ref, _ = tree_ref.query(test_pts, k=1)
    mean_dist = 0.5 * (float(np.mean(dists_ref_to_test)) + float(np.mean(dists_test_to_ref)))
    sym_hausdorff = float(max(np.max(dists_ref_to_test), np.max(dists_test_to_ref)))

    test_len = compute_centerline_length(test_cl)
    ref_len = float(np.sum(np.linalg.norm(np.diff(ref_pts, axis=0), axis=1)))
    length_err = abs(test_len - ref_len) / max(ref_len, 1e-12)

    ref_tort = compute_tortuosity_from_points(ref_pts)
    test_tort = test_cl.tortuosity if test_cl.tortuosity is not None else compute_tortuosity(test_cl)
    tort_err = abs(test_tort - ref_tort) / max(ref_tort, 1e-12)

    ref_mean_rad = float(np.nanmean(ref_rad))
    test_mean_rad = float(np.nanmean(test_rad))
    rad_err = abs(test_mean_rad - ref_mean_rad) / max(ref_mean_rad, 1e-12)

    anatomical = "PASS"
    if test_mean_rad < 0.5:
        anatomical = "FAIL: radius too small"
    elif test_len < 50.0:
        anatomical = "FAIL: length too small"
    elif sym_hausdorff > 50.0:
        anatomical = "FAIL: too far from reference"

    status = "PASS" if anatomical == "PASS" else "FAIL"

    return {
        "n_points": int(len(test_pts)),
        "length_mm": float(test_len),
        "mean_radius_mm": float(test_mean_rad),
        "mean_distance_mm": float(mean_dist),
        "hausdorff_mm": float(sym_hausdorff),
        "length_error_pct": float(length_err * 100.0),
        "tortuosity_error_pct": float(tort_err * 100.0),
        "radius_error_pct": float(rad_err * 100.0),
        "anatomical_plausibility": anatomical,
        "status": status,
    }


# Explicit mapping of input surfaces to reference centerlines.
# The vmtk-test-data repo stores some references under input/ as well.
KNOWN_CASES: Dict[str, str] = {
    "aorta-surface-open-ends": "aorta-centerline.vtp",
    "aorta-surface-branch-split": "aorta-centerline-branches.vtp",
}


def discover_test_cases(
    input_dir: Path,
    ref_dir: Path,
) -> List[Tuple[Path, Optional[Path]]]:
    """Map input surfaces to reference centerlines."""
    surface_exts = {".stl", ".vtp"}
    ref_ext = ".vtp"

    input_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in surface_exts
        and "centerline" not in p.stem.lower()
    )
    ref_files = sorted(
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ref_ext
    )

    ref_by_name = {p.stem: p for p in ref_files}

    cases: List[Tuple[Path, Optional[Path]]] = []
    for inp in input_files:
        stem = inp.stem
        ref_name = KNOWN_CASES.get(stem)
        if ref_name is None:
            ref_path = None
        else:
            ref_path = ref_by_name.get(ref_name.replace(".vtp", ""))
            if ref_path is None:
                ref_path = (input_dir / ref_name).resolve()
        cases.append((inp, ref_path))

    return cases


def run_single_case(
    input_path: Path,
    ref_path: Optional[Path],
    backend: str = "python_eikonal",
    acceleration: str = "numpy",
    resampling_step: float = 1.0,
) -> CaseResult:
    case_name = input_path.stem
    logger.info("Processing case: %s", case_name)

    try:
        cl, report = run_pipeline(
            input_path=input_path,
            backend=backend,
            acceleration=acceleration,
            resampling_step=resampling_step,
        )
    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", case_name, exc, exc_info=True)
        return CaseResult(
            case_name=case_name,
            status="ERROR",
            input_path=str(input_path),
            ref_path=str(ref_path) if ref_path else None,
            n_points=0,
            length_mm=0.0,
            mean_radius_mm=0.0,
            mean_distance_mm=0.0,
            hausdorff_mm=0.0,
            length_error_pct=0.0,
            tortuosity_error_pct=0.0,
            radius_error_pct=0.0,
            total_time_s=0.0,
            warnings=[f"Pipeline exception: {exc}"],
        )

    total_time = float(report.phase_timings.get("total_elapsed_s", sum(report.phase_timings.values())))
    warnings = list(report.warnings)

    phase_timings = {
        phase: float(report.phase_timings.get(phase, 0.0))
        for phase in [
            "preprocess",
            "capping",
            "delaunay",
            "internal_tets",
            "voronoi",
            "poles",
            "fast_marching",
            "resampling",
            "sections",
            "network",
        ]
    }

    quality_metrics = {
        "voronoi_n_points": report.quality_metrics.get("voronoi_n_points", 0),
        "voronoi_n_edges": report.quality_metrics.get("voronoi_n_edges", 0),
        "n_internal_tets": report.quality_metrics.get("n_internal_tets", 0),
        "n_seed_component": report.quality_metrics.get("n_seed_component", 0),
    }

    if cl is None or len(cl.points) < 2:
        logger.warning("Empty or invalid centerline for %s", case_name)
        return CaseResult(
            case_name=case_name,
            status="ERROR",
            input_path=str(input_path),
            ref_path=str(ref_path) if ref_path else None,
            n_points=0,
            length_mm=0.0,
            mean_radius_mm=0.0,
            mean_distance_mm=0.0,
            hausdorff_mm=0.0,
            length_error_pct=0.0,
            tortuosity_error_pct=0.0,
            radius_error_pct=0.0,
            total_time_s=total_time,
            phase_timings=phase_timings,
            quality_metrics=quality_metrics,
            warnings=warnings + ["Empty centerline returned"],
        )

    metrics = {
        "n_points": int(len(cl.points)),
        "length_mm": compute_centerline_length(cl),
        "mean_radius_mm": float(np.nanmean(cl.radii)) if len(cl.radii) > 0 else 0.0,
        "mean_distance_mm": 0.0,
        "hausdorff_mm": 0.0,
        "length_error_pct": 0.0,
        "tortuosity_error_pct": 0.0,
        "radius_error_pct": 0.0,
        "anatomical_plausibility": "NO_REF",
        "status": "NO_REF",
    }

    anatomical = "NO_REF"
    if ref_path is not None and ref_path.exists():
        anatomical = "PASS"
        if metrics["mean_radius_mm"] < 0.5:
            anatomical = "FAIL: radius too small"
        elif metrics["length_mm"] < 50.0:
            anatomical = "FAIL: length too small"
    metrics["anatomical_plausibility"] = anatomical
    if anatomical != "PASS":
        metrics["status"] = anatomical

    if ref_path is not None and ref_path.exists():
        try:
            ref_pts, ref_rad = read_vtp_points_radii(ref_path)
            if len(ref_pts) < 2:
                raise ValueError("Reference has fewer than 2 points")

            comp = compute_metrics(cl, ref_pts, ref_rad)
            metrics.update(comp)
        except Exception as exc:
            logger.error("Reference comparison failed for %s: %s", case_name, exc, exc_info=True)
            warnings.append(f"Reference comparison failed: {exc}")
            metrics["status"] = "ERROR"
    else:
        logger.info("No reference for %s (expected path: %s)", case_name, ref_path)

    return CaseResult(
        case_name=case_name,
        status=metrics["status"],
        input_path=str(input_path),
        ref_path=str(ref_path) if ref_path else None,
        n_points=metrics["n_points"],
        length_mm=metrics["length_mm"],
        mean_radius_mm=metrics["mean_radius_mm"],
        mean_distance_mm=metrics["mean_distance_mm"],
        hausdorff_mm=metrics["hausdorff_mm"],
        length_error_pct=metrics["length_error_pct"],
        tortuosity_error_pct=metrics["tortuosity_error_pct"],
        radius_error_pct=metrics["radius_error_pct"],
        total_time_s=total_time,
        phase_timings=phase_timings,
        quality_metrics=quality_metrics,
        warnings=warnings,
        anatomical_plausibility=metrics["anatomical_plausibility"],
    )


def print_table(results: List[CaseResult]) -> None:
    headers = [
        "Case",
        "Status",
        "Points",
        "Length(mm)",
        "MeanR(mm)",
        "MeanDist(mm)",
        "Hausdorff(mm)",
        "LenErr(%)",
        "TortErr(%)",
        "RadErr(%)",
        "Total(s)",
    ]
    rows = []
    for r in results:
        rows.append([
            r.case_name,
            r.status,
            str(r.n_points),
            f"{r.length_mm:.2f}",
            f"{r.mean_radius_mm:.3f}",
            f"{r.mean_distance_mm:.3f}",
            f"{r.hausdorff_mm:.3f}",
            f"{r.length_error_pct:.2f}",
            f"{r.tortuosity_error_pct:.2f}",
            f"{r.radius_error_pct:.2f}",
            f"{r.total_time_s:.3f}",
        ])

    col_widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, col_widths))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = project_root / "test" / "vmtk-test-data" / "input"
    ref_dir = project_root / "test" / "vmtk-test-data" / "centerlinereference"
    output_json = Path("/tmp/vmtk_exhaustive_benchmark.json")

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 1
    if not ref_dir.is_dir():
        logger.error("Reference directory not found: %s", ref_dir)
        return 1

    cases = discover_test_cases(input_dir, ref_dir)
    logger.info("Discovered %d test cases", len(cases))

    results: List[CaseResult] = []
    for input_path, ref_path in cases:
        result = run_single_case(
            input_path=input_path,
            ref_path=ref_path,
            backend="python_fmm",
            acceleration="numpy",
            resampling_step=1.0,
        )
        results.append(result)

    print("\n" + "=" * 140)
    print("VMTK EXHAUSTIVE BENCHMARK (foampilot)")
    print("=" * 140)
    print_table(results)
    print("=" * 140)

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errored = sum(1 for r in results if r.status == "ERROR")
    no_ref = sum(1 for r in results if r.status == "NO_REF")
    print(f"\nSummary: {passed} passed, {failed} failed, {errored} errored, {no_ref} no-reference")

    json_report = []
    for r in results:
        json_report.append({
            "case_name": r.case_name,
            "status": r.status,
            "input_path": r.input_path,
            "ref_path": r.ref_path,
            "n_points": r.n_points,
            "length_mm": r.length_mm,
            "mean_radius_mm": r.mean_radius_mm,
            "mean_distance_mm": r.mean_distance_mm,
            "hausdorff_mm": r.hausdorff_mm,
            "length_error_pct": r.length_error_pct,
            "tortuosity_error_pct": r.tortuosity_error_pct,
            "radius_error_pct": r.radius_error_pct,
            "total_time_s": r.total_time_s,
            "phase_timings": r.phase_timings,
            "quality_metrics": r.quality_metrics,
            "warnings": r.warnings,
            "anatomical_plausibility": r.anatomical_plausibility,
        })

    output_json.write_text(json.dumps(json_report, indent=2))
    logger.info("Full JSON report saved to %s", output_json)

    if failed > 0 or errored > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
