#!/usr/bin/env python3
"""
Run all wind rose CFD cases sequentially (or check convergence).

Reads case directories from `cases/` and runs simpleFoam for each.

Usage:
    PYTHONPATH=src python3 run_all_cases.py [--nb-proc 4] [--max-cases N]
"""

import argparse
import os
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot import postprocess, utilities
from foampilot.postprocess import FoamPostProcessing


def run_case(case_dir: Path, nb_proc: int = 2, check_only: bool = False, sigfpe: bool = False):
    """Run a single CFD case. Check convergence if already solved."""
    print(f"\n{'=' * 60}")
    print(f"Case: {case_dir.name}")
    print(f"{'=' * 60}")

    # Load metadata
    metadata = {}
    meta_path = case_dir / "case_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    if metadata:
        print(f"  Direction: {metadata.get('direction_deg', 'N/A')}°  Speed: {metadata.get('speed_10m', 'N/A')} m/s  Freq: {metadata.get('frequency', 'N/A')}")
    else:
        print(f"  No metadata found for {case_dir.name}")

    log_file = case_dir / "log.incompressibleFluid"

    if log_file.exists() and not check_only:
        print("  Log exists — skipping.")
        return True

    if check_only and not log_file.exists():
        print("  No log file — case not yet run.")
        return False

    if not check_only:
        # Run the simulation
        from foampilot import Meshing
        from foampilot.solver import Solver

        solver = Solver(case_dir)
        solver.compressible = False
        solver.with_gravity = False
        solver.turbulence_model = metadata.get("turbulence_model", "kOmegaSST")
        solver.transient = False

        if sigfpe:
            os.environ["FOAM_SIGFPE"] = "1"
        solver.run_simulation(nb_proc=nb_proc, log_filename=str(log_file))

    # --- Post-processing: check residuals ---
    if log_file.exists():
        residuals_post = utilities.ResidualsPost(log_file)
        residuals_post.process(export_csv=True, export_json=True,
                                export_png=False, export_html=False)

        # Load residuals JSON
        residuals_json = case_dir / "residuals" / "log_residuals.json"
        if residuals_json.exists():
            with open(residuals_json) as f:
                residuals = json.load(f)
            final = residuals.get("final", {})
            all_converged = True
            for field, val in final.items():
                converged = val < 1e-4
                status = "OK" if converged else "FAIL"
                if not converged:
                    all_converged = False
                print(f"  Residual {field}: {val:.2e} [{status}]")
            return all_converged
        else:
            print("  Could not load residuals JSON — convergence unknown.")
            return False
    else:
        print(f"  Log file not found: {log_file}")
        return False


def foam_to_vtk(case_dir: Path):
    """Convert OpenFOAM results to VTK for post-processing."""
    foam_post = FoamPostProcessing(case_path=case_dir)
    foam_post.foamToVTK()
    print(f"  VTK conversion done: {case_dir / 'VTK'}")


def main():
    parser = argparse.ArgumentParser(description="Run all wind rose CFD cases")
    parser.add_argument("--cases-dir", default="cases", help="Directory containing case subfolders")
    parser.add_argument("--nb-proc", type=int, default=2, help="Number of parallel processes")
    parser.add_argument("--sigfpe", action="store_true", help="Enable FOAM_SIGFPE")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit to first N cases")
    parser.add_argument("--check-only", action="store_true", help="Only check convergence, don't run")
    parser.add_argument("--foam-to-vtk", action="store_true", help="Also run foamToVTK after simulation")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    case_dirs = sorted([d for d in cases_dir.iterdir() if d.is_dir() and d.name.startswith("wind_")])

    if not case_dirs:
        print(f"No cases found in {cases_dir}. Run generate_wind_cases.py first.")
        sys.exit(1)

    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]

    print(f"Processing {len(case_dirs)} case(s)...")

    results = {}
    for case_dir in case_dirs:
        start = time.time()
        try:
            converged = run_case(case_dir, nb_proc=args.nb_proc, check_only=args.check_only, sigfpe=args.sigfpe)
            results[case_dir.name] = {
                "converged": converged,
                "elapsed": time.time() - start,
            }
            if converged and args.foam_to_vtk:
                foam_to_vtk(case_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[case_dir.name] = {"error": str(e), "elapsed": time.time() - start}

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    n_ok = sum(1 for r in results.values() if r.get("converged"))
    n_fail = sum(1 for r in results.values() if r.get("error"))
    n_unknown = len(results) - n_ok - n_fail
    print(f"  Converged: {n_ok}")
    print(f"  Failed:    {n_fail}")
    print(f"  Unknown:   {n_unknown}")

    # Save summary
    summary_path = cases_dir.parent / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
