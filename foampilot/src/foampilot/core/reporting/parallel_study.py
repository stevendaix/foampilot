#!/usr/bin/env python3
"""
Systematic parallelization study framework for OpenFOAM building-aero cases.

Test matrix:
    Processor counts : 1, 2, 4, 6, 8
    Methods          : scotch, simple (all factor combinations of n_procs)

For each configuration the framework:
    1. Writes decomposeParDict and runs decomposePar
    2. Runs the simulation (or reads an existing log)
    3. Measures convergence, CPU time, processor-boundary faces, cell imbalance
    4. Exports processor-boundary faces to VTK/VTP for ParaView inspection
    5. Aggregates results into a JSON report

Outputs
-------
    parallel_study_report.json   — aggregated metrics for every configuration
    processor_boundaries.vtp     — unified VTK file with all processor interfaces

Usage
-----
    PYTHONPATH=src python3 -m foampilot.report.parallel_study \\
        --case cases/wind_0deg \\
        --max-time 30 \\
        --output-dir .

    # Analyse-only mode (no simulation, reads existing logs / processor dirs):
    PYTHONPATH=src python3 -m foampilot.report.parallel_study \\
        --case cases/wind_0deg \\
        --analyze-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOLVER_NAME = "incompressibleFluud"
try:
    from foampilot.solver.base_solver import BaseSolver
    SOLVER_MODULE = BaseSolver.SOLVER_MODULES.get("incompressibleFluid", "incompressibleFluid")
except Exception:
    SOLVER_MODULE = "incompressibleFluid"

BASELINE_FV_SOLUTION = """FoamFile
{
    version     2.0;
    format     ascii;
    class     dictionary;
    object     fvSolution;
}

solvers
{
    p
    {
        solver PCG;
        preconditioner DIC;
        tolerance 1e-06;
        relTol 0.1;
    }
    U
    {
        solver smoothSolver;
        tolerance 1e-05;
        relTol 0.1;
        smoother symGaussSeidel;
    }
    k
    {
        solver smoothSolver;
        tolerance 1e-05;
        relTol 0.1;
        smoother symGaussSeidel;
    }
    epsilon
    {
        solver smoothSolver;
        tolerance 1e-05;
        relTol 0.1;
        smoother symGaussSeidel;
    }
    nut
    {
        solver smoothSolver;
        tolerance 1e-05;
        relTol 0.1;
        smoother symGaussSeidel;
    }
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors 3;
    nCorrectors 2;
    nNonOrthogonalCorrectors 3;
    pRefPoint (0 0 0);
    pRefValue 0;
}
relaxationFactors
{
    fields
    {
        p 0.2;
    }
    equations
    {
        U 0.5;
        "(k|epsilon|omega).*" 0.5;
    }
}
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DecompositionConfig:
    n_procs: int
    method: str
    direction: tuple[int, int, int] | None = None

    @property
    def label(self) -> str:
        if self.method == "scotch":
            return f"{self.n_procs}proc_scotch"
        return (
            f"{self.n_procs}proc_simple_"
            f"{self.direction[0]}_{self.direction[1]}_{self.direction[2]}"
        )


@dataclass
class RunMetrics:
    name: str
    n_procs: int
    method: str
    direction: tuple[int, int, int] | None

    # Timing
    elapsed_s: float = 0.0
    timed_out: bool = False
    decompose_failed: bool = False
    decompose_error: str | None = None

    # Convergence (from log)
    last_time: float | None = None
    time_steps_completed: int = 0
    crashed: bool = False
    continuity_blowup: tuple[float, float] | None = None
    crash_location: str | None = None
    final_residuals: dict[str, float] = field(default_factory=dict)
    convergence_reached: bool = False

    # Mesh metrics
    total_cells: int = 0
    total_faces: int = 0
    proc_boundary_faces: int = 0
    cell_imbalance: float = 0.0
    max_cells_proc: int = 0
    min_cells_proc: int = 0
    mean_cells_proc: float = 0.0
    std_cells_proc: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = list(d["direction"]) if d["direction"] else None
        return d


# ---------------------------------------------------------------------------
# Helper: valid simple directions for a given processor count
# ---------------------------------------------------------------------------
def _factor_tuples(n: int) -> list[tuple[int, int, int]]:
    """Return all (nx, ny, nz) triples whose product is exactly n, with
    nx >= ny >= nz >= 1 to avoid duplicate permutations."""
    triples = []
    for nx in range(1, int(round(n ** (1/3))) + 2):
        if n % nx != 0:
            continue
        rem = n // nx
        for ny in range(1, int(math.sqrt(rem)) + 2):
            if rem % ny != 0:
                continue
            nz = rem // ny
            if nx * ny * nz == n and nx >= ny >= nz:
                triples.append((nx, ny, nz))
    return triples


# ---------------------------------------------------------------------------
# Helper: write decomposeParDict
# ---------------------------------------------------------------------------
def _write_decompose_par_dict(case_dir: Path, cfg: DecompositionConfig) -> None:
    dpd = case_dir / "system" / "decomposeParDict"
    if cfg.method == "scotch":
        content = f"""FoamFile
{{
    version     2.0;
    format     ascii;
    class     dictionary;
    object     decomposeParDict;
}}

numberOfSubdomains {cfg.n_procs};
method scotch;
"""
    else:
        nx, ny, nz = cfg.direction
        content = f"""FoamFile
{{
    version     2.0;
    format     ascii;
    class     dictionary;
    object     decomposeParDict;
}}

numberOfSubdomains {cfg.n_procs};
method simple;

simpleCoeffs
{{
    n       ({nx} {ny} {nz});
    delta   0.001;
}}
"""
    dpd.write_text(content)


# ---------------------------------------------------------------------------
# Helper: clean case between runs
# ---------------------------------------------------------------------------
def _clean_case(case_dir: Path) -> None:
    for d in case_dir.glob("processor*"):
        if d.is_dir():
            shutil.rmtree(d)
    for d in case_dir.iterdir():
        if d.is_dir() and d.name not in ("0", "constant", "dynamicCode"):
            try:
                float(d.name)
                shutil.rmtree(d)
            except ValueError:
                pass
    for log in case_dir.glob("log.*"):
        log.unlink()
    vtk = case_dir / "VTK"
    if vtk.exists():
        shutil.rmtree(vtk)


# ---------------------------------------------------------------------------
# Helper: ensure baseline fvSolution
# ---------------------------------------------------------------------------
def _ensure_baseline_fv_solution(case_dir: Path) -> None:
    fv = case_dir / "system" / "fvSolution"
    fv.write_text(BASELINE_FV_SOLUTION)


# ---------------------------------------------------------------------------
# Helper: run decomposePar
# ---------------------------------------------------------------------------
def _run_decompose_par(case_dir: Path, cfg: DecompositionConfig) -> tuple[bool, str]:
    _write_decompose_par_dict(case_dir, cfg)
    for d in case_dir.glob("processor*"):
        if d.is_dir():
            shutil.rmtree(d)
    result = subprocess.run(
        ["decomposePar", "-case", str(case_dir)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Helper: run simulation (serial or parallel)
# ---------------------------------------------------------------------------
def _run_simulation(
    case_dir: Path,
    cfg: DecompositionConfig,
    log_path: Path,
    timeout: int = 600,
) -> dict:
    n_procs = cfg.n_procs
    method = cfg.method
    direction = cfg.direction

    if n_procs > 1:
        _write_decompose_par_dict(case_dir, cfg)
        for d in case_dir.glob("processor*"):
            if d.is_dir():
                shutil.rmtree(d)
        result = subprocess.run(
            ["decomposePar", "-case", str(case_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {
                "returncode": result.returncode,
                "elapsed": 0.0,
                "timed_out": False,
                "decompose_failed": True,
                "decompose_error": result.stderr,
            }

    if n_procs == 1:
        cmd = ["foamRun", "-solver", SOLVER_MODULE]
    else:
        cmd = [
            "mpirun", "--oversubscribe", "-np", str(n_procs),
            "foamRun", "-solver", SOLVER_MODULE, "-parallel",
        ]

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=case_dir,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc.wait(timeout=timeout)
        elapsed = time.time() - start
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        elapsed = time.time() - start
        returncode = -1

    return {
        "returncode": returncode,
        "elapsed": elapsed,
        "timed_out": returncode == -1,
        "decompose_failed": False,
    }


# ---------------------------------------------------------------------------
# Helper: parse OpenFOAM log for convergence / crash info
# ---------------------------------------------------------------------------
def _parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {
            "last_time": None,
            "time_steps_completed": 0,
            "crashed": True,
            "error": "no log",
            "continuity_blowup": None,
            "crash_location": None,
            "final_residuals": {},
        }

    content = log_path.read_text(errors="replace")
    lines = content.splitlines()

    last_time = None
    time_steps = []
    for line in lines:
        m = re.search(r"Time = (\d+\.?\d*)s", line)
        if m:
            t = float(m.group(1))
            last_time = t
            time_steps.append(t)

    crashed = any(
        "Floating point exception" in l
        or "FOAM FATAL ERROR" in l
        or "FOAM exiting" in l
        for l in lines
    )

    continuity_blowup = None
    for line in lines:
        m = re.search(r"time step continuity errors : sum local = ([0-9.e+-]+)", line)
        if m:
            val = float(m.group(1))
            if val > 1e10:
                continuity_blowup = (time_steps[-1] if time_steps else None, val)
                break

    crash_location = None
    for line in lines:
        if "Floating point exception" in line or "FOAM FATAL ERROR" in line:
            for l in lines[max(0, lines.index(line) - 5):]:
                if "::" in l and ("solve" in l or "correct" in l or "smooth" in l):
                    crash_location = l.strip()
                    break
            break

    final_residuals = {}
    for line in lines:
        m = re.search(r"GAMG.*?tolerance\s+=\s+([0-9.e+-]+)", line)
        if m:
            final_residuals["p_GAMG"] = float(m.group(1))
        m = re.search(r"Final residual = ([0-9.e+-]+)", line)
        if m:
            final_residuals["final"] = float(m.group(1))
            break

    convergence_reached = False
    for line in lines:
        if "reached convergence" in line.lower() or "solution converged" in line.lower():
            convergence_reached = True
            break

    return {
        "last_time": last_time,
        "time_steps_completed": len(time_steps),
        "crashed": crashed,
        "error": None,
        "continuity_blowup": continuity_blowup,
        "crash_location": crash_location,
        "final_residuals": final_residuals,
        "convergence_reached": convergence_reached,
    }


# ---------------------------------------------------------------------------
# Mesh metrics: cell counts per processor, boundary faces, imbalance
# ---------------------------------------------------------------------------
def _read_processor_mesh_metrics(case_dir: Path) -> dict:
    """Return metrics extracted from the decomposed processor directories."""
    proc_dirs = sorted(case_dir.glob("processor*"))
    if not proc_dirs:
        return {
            "total_cells": 0,
            "total_faces": 0,
            "proc_boundary_faces": 0,
            "cell_imbalance": 0.0,
            "max_cells_proc": 0,
            "min_cells_proc": 0,
            "mean_cells_proc": 0.0,
            "std_cells_proc": 0.0,
            "cells_per_proc": [],
            "boundary_faces_per_proc": [],
        }

    cells_per_proc = []
    boundary_faces_per_proc = []
    total_faces = 0

    for proc_dir in proc_dirs:
        mesh_dir = proc_dir / "constant" / "polyMesh"
        if not mesh_dir.exists():
            continue

        # Cell count from owner file
        owner_path = mesh_dir / "owner"
        if owner_path.exists():
            try:
                from foampilot.utilities.read_mesh import OpenFoamFile
                owner = OpenFoamFile(str(mesh_dir), name="owner", verbose=False)
                n_cells = int(owner.nb_cell)
            except Exception:
                n_cells = 0
        else:
            n_cells = 0
        cells_per_proc.append(n_cells)

        # Face count and processor-boundary face count from boundary file
        boundary_path = mesh_dir / "boundary"
        proc_boundary_faces = 0
        n_faces_total = 0
        if boundary_path.exists():
            try:
                from foampilot.utilities.read_mesh import OpenFoamFile
                bf = OpenFoamFile(str(mesh_dir), name="boundary", verbose=False)
                if hasattr(bf, "boundaryface") and bf.boundaryface:
                    for patch_name, patch_data in bf.boundaryface.items():
                        name_str = patch_name.decode() if isinstance(patch_name, bytes) else str(patch_name)
                        if name_str.startswith("processor"):
                            proc_boundary_faces += int(patch_data[b"nFaces"])
                        n_faces_total += int(patch_data[b"nFaces"])
            except Exception:
                pass
        boundary_faces_per_proc.append(proc_boundary_faces)
        total_faces += n_faces_total

    arr = np.array(cells_per_proc, dtype=float)
    total_cells = int(arr.sum())
    imbalance = float((arr.max() - arr.min()) / (arr.mean() + 1e-12))

    return {
        "total_cells": total_cells,
        "total_faces": total_faces,
        "proc_boundary_faces": int(sum(boundary_faces_per_proc)),
        "cell_imbalance": imbalance,
        "max_cells_proc": int(arr.max()) if arr.size else 0,
        "min_cells_proc": int(arr.min()) if arr.size else 0,
        "mean_cells_proc": float(arr.mean()) if arr.size else 0.0,
        "std_cells_proc": float(arr.std()) if arr.size else 0.0,
        "cells_per_proc": [int(x) for x in arr],
        "boundary_faces_per_proc": boundary_faces_per_proc,
    }


# ---------------------------------------------------------------------------
# VTK / VTP export for processor boundaries
# ---------------------------------------------------------------------------
def _export_processor_boundaries_vtp(case_dir: Path, output_path: Path) -> None:
    """Export a single VTP file containing all processor-boundary faces."""
    if not case_dir.exists():
        return

    # Try foamToVTK first (constant + all regions)
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.exists():
        try:
            subprocess.run(
                ["foamToVTK", "-case", str(case_dir), "-constant", "-allRegions"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            pass

    import pyvista as pv
    meshes: list[pv.PolyData] = []
    proc_dirs = sorted(case_dir.glob("processor*"))
    if not proc_dirs:
        # Serial case — no processor boundaries
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pv.PolyData()
        empty.save(str(output_path))
        return

    for proc_dir in proc_dirs:
        proc_name = proc_dir.name
        region_vtk = vtk_dir / proc_name / f"{proc_name}_constant.vtk"
        if not region_vtk.exists():
            # Fallback: try latest time step VTK for this processor
            time_dirs = sorted(
                [d for d in vtk_dir.glob(f"{proc_name}_*") if d.is_dir()],
                reverse=True,
            )
            if time_dirs:
                candidate = time_dirs[0] / f"{time_dirs[0].name}_{time_dirs[0].name.split('_')[-1]}.vtk"
                if candidate.exists():
                    region_vtk = candidate

        if not region_vtk.exists():
            continue

        try:
            mesh = pv.read(str(region_vtk))
        except Exception:
            continue

        # Try to identify processor-boundary blocks
        # In foamToVTK output they may appear as separate blocks or as cell data
        blocks = mesh if isinstance(mesh, pv.MultiBlock) else [mesh]
        for block in blocks:
            name = block.name if hasattr(block, "name") else ""
            if "processor" in name.lower() or "proc" in name.lower():
                block.cell_data["Processor"] = np.full(block.n_cells, int(proc_name.replace("processor", "")))
                meshes.append(block)
                continue

            # If cell data contains a "processor" array, extract those cells
            if "processor" in [k.lower() for k in block.cell_data.keys()]:
                proc_key = next(k for k in block.cell_data.keys() if k.lower() == "processor")
                try:
                    unique_procs = np.unique(block.cell_data[proc_key])
                    for pid in unique_procs:
                        sub = block.extract_cells(block.cell_data[proc_key] == pid)
                        sub.cell_data["Processor"] = np.full(sub.n_cells, int(pid))
                        meshes.append(sub)
                except Exception:
                    pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if meshes:
        combined = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
        combined.save(str(output_path))
    else:
        empty = pv.PolyData()
        empty.save(str(output_path))


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def _print_comparison(results: list[RunMetrics]) -> None:
    print(f"\n{'=' * 90}")
    print(f"  PARALLEL STUDY — COMPARISON")
    print(f"{'=' * 90}")
    header = (
        f"  {'Config':<42} {'Procs':>5} {'Status':>8} {'t_last':>8} "
        f"{'Steps':>6} {'CPU s':>8} {'ProcFaces':>10} {'Imb':>7}"
    )
    print(header)
    print(f"  {'-' * 90}")
    for r in results:
        status = "TIMEOUT" if r.timed_out else ("CRASH" if r.crashed else "OK")
        t_last = f"{r.last_time:.0f}s" if r.last_time is not None else "N/A"
        print(
            f"  {r.name:<42} {r.n_procs:>5} {status:>8} {t_last:>8} "
            f"{r.time_steps_completed:>6} {r.elapsed_s:>8.1f} "
            f"{r.proc_boundary_faces:>10} {r.cell_imbalance:>7.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main study class
# ---------------------------------------------------------------------------
class ParallelStudy:
    def __init__(
        self,
        case_dir: Path,
        output_dir: Path,
        procs: list[int] | None = None,
        max_time: int = 30,
        solver: str = SOLVER_MODULE,
    ):
        self.case_dir = Path(case_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.procs = procs or [1, 2, 4, 6, 8]
        self.max_time = max_time
        self.solver = solver
        self.results: list[RunMetrics] = []

        if not self.case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {self.case_dir}")

    # ------------------------------------------------------------------
    # Build test matrix
    # ------------------------------------------------------------------
    def build_matrix(self) -> list[DecompositionConfig]:
        tests: list[DecompositionConfig] = []
        for n_proc in self.procs:
            if n_proc == 1:
                tests.append(DecompositionConfig(n_procs=1, method="scotch"))
            else:
                tests.append(DecompositionConfig(n_procs=n_proc, method="scotch"))
                for direction in _factor_tuples(n_proc):
                    tests.append(
                        DecompositionConfig(
                            n_procs=n_proc,
                            method="simple",
                            direction=direction,
                        )
                    )
        # deduplicate
        seen: set[tuple] = set()
        unique: list[DecompositionConfig] = []
        for t in tests:
            key = (t.n_procs, t.method, t.direction)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    # ------------------------------------------------------------------
    # Execute the full study
    # ------------------------------------------------------------------
    def run(self, analyze_only: bool = False) -> list[RunMetrics]:
        matrix = self.build_matrix()
        print(f"\n{'=' * 70}")
        print(f"  Parallel study — {self.case_dir.name}")
        print(f"  Max time per run: {self.max_time}s")
        print(f"{'=' * 70}")
        print(f"  Tests to run: {len(matrix)}")
        for t in matrix:
            dir_str = str(t.direction) if t.direction else "N/A"
            print(f"    {t.label:<42}  procs={t.n_procs}  method={t.method}  dir={dir_str}")
        print()

        self.results = []
        for i, cfg in enumerate(matrix, 1):
            print(f"[{i}/{len(matrix)}] Running: {cfg.label}")
            metrics = self._run_single(cfg, analyze_only=analyze_only)
            self.results.append(metrics)

            status = "TIMEOUT" if metrics.timed_out else ("CRASH" if metrics.crashed else "OK")
            print(
                f"  -> {status}  t_last={metrics.last_time}  "
                f"steps={metrics.time_steps_completed}  "
                f"elapsed={metrics.elapsed_s:.1f}s  "
                f"procFaces={metrics.proc_boundary_faces}  "
                f"imb={metrics.cell_imbalance:.3f}"
            )
            if metrics.continuity_blowup:
                print(
                    f"     continuity blowup at t={metrics.continuity_blowup[0]}s: "
                    f"{metrics.continuity_blowup[1]:.2e}"
                )
            if metrics.crash_location:
                print(f"     crash in: {metrics.crash_location}")
            print()

        _print_comparison(self.results)
        self._write_report()
        self._export_vtp()
        return self.results

    # ------------------------------------------------------------------
    # Single configuration
    # ------------------------------------------------------------------
    def _run_single(self, cfg: DecompositionConfig, analyze_only: bool = False) -> RunMetrics:
        name = cfg.label
        n_procs = cfg.n_procs
        method = cfg.method
        direction = cfg.direction

        log_path = self.case_dir / f"log.study_{name}"

        # Analyze-only mode: skip simulation, read existing artifacts
        if analyze_only:
            stats = {
                "returncode": 0,
                "elapsed": 0.0,
                "timed_out": False,
                "decompose_failed": False,
            }
        else:
            _clean_case(self.case_dir)
            _ensure_baseline_fv_solution(self.case_dir)
            stats = _run_simulation(self.case_dir, cfg, log_path, timeout=self.max_time)

        crash_info = _parse_log(log_path)
        mesh_metrics = _read_processor_mesh_metrics(self.case_dir)

        return RunMetrics(
            name=name,
            n_procs=n_procs,
            method=method,
            direction=direction,
            elapsed_s=round(stats.get("elapsed", 0.0), 2),
            timed_out=stats.get("timed_out", False),
            decompose_failed=stats.get("decompose_failed", False),
            decompose_error=stats.get("decompose_error"),
            last_time=crash_info.get("last_time"),
            time_steps_completed=crash_info.get("time_steps_completed", 0),
            crashed=crash_info.get("crashed", False),
            continuity_blowup=crash_info.get("continuity_blowup"),
            crash_location=crash_info.get("crash_location"),
            final_residuals=crash_info.get("final_residuals", {}),
            convergence_reached=crash_info.get("convergence_reached", False),
            total_cells=mesh_metrics.get("total_cells", 0),
            total_faces=mesh_metrics.get("total_faces", 0),
            proc_boundary_faces=mesh_metrics.get("proc_boundary_faces", 0),
            cell_imbalance=mesh_metrics.get("cell_imbalance", 0.0),
            max_cells_proc=mesh_metrics.get("max_cells_proc", 0),
            min_cells_proc=mesh_metrics.get("min_cells_proc", 0),
            mean_cells_proc=mesh_metrics.get("mean_cells_proc", 0.0),
            std_cells_proc=mesh_metrics.get("std_cells_proc", 0.0),
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def _write_report(self) -> None:
        report_path = self.output_dir / "parallel_study_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case": str(self.case_dir),
            "solver": self.solver,
            "max_time_per_run_s": self.max_time,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "results": [r.to_dict() for r in self.results],
        }
        report_path.write_text(json.dumps(payload, indent=2))
        print(f"  Report saved: {report_path}")

    # ------------------------------------------------------------------
    # VTP export
    # ------------------------------------------------------------------
    def _export_vtp(self) -> None:
        vtp_path = self.output_dir / "processor_boundaries.vtp"
        _export_processor_boundaries_vtp(self.case_dir, vtp_path)
        if vtp_path.exists():
            print(f"  Processor boundaries VTP: {vtp_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systematic parallelization study for OpenFOAM building-aero cases",
    )
    parser.add_argument(
        "--case",
        default="cases/wind_0deg",
        help="Case directory (relative to cwd or absolute)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for JSON report and VTP output",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=30,
        help="Max seconds per run before timeout",
    )
    parser.add_argument(
        "--procs",
        default="1,2,4,6,8",
        help="Comma-separated processor counts",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Do not run simulations; only analyse existing logs / processor dirs",
    )
    args = parser.parse_args()

    case_path = Path(args.case)
    if not case_path.is_absolute():
        case_path = (Path.cwd() / case_path).resolve()

    if not case_path.exists():
        print(f"ERROR: case directory not found: {case_path}")
        sys.exit(1)

    procs = [int(x.strip()) for x in args.procs.split(",") if x.strip()]

    study = ParallelStudy(
        case_dir=case_path,
        output_dir=Path(args.output_dir).resolve(),
        procs=procs,
        max_time=args.max_time,
    )
    study.run(analyze_only=args.analyze_only)


if __name__ == "__main__":
    main()
