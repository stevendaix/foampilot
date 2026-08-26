"""Reproduction Foampilot du tutoriel DLBFoam 2D planar hydrogen flame.

Le dossier ``case_template`` contient uniquement les données d'entrée publiées.
Tous les dictionnaires et champs sont ré-écrits dans ``case`` par
OpenFOAMDictAddFile avant l'exécution. Le calcul réactif nécessite les
bibliothèques DLBFoam/FickianTransportFoam et le mécanisme PyJac compilé.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "case_template"
DEFAULT_CASE = ROOT / "case"
REQUIRED = ("0", "constant", "system")


def _foam_header_version(text: str) -> str:
    """Normalize cosmetic OpenFOAM header versions without changing physics."""
    return (text.replace("Version:  9", "Version:  13")
                .replace("Version:  10", "Version:  13")
                .replace("Version: 9", "Version: 13")
                .replace("Version: 10", "Version: 13"))


def prepare_case(case_dir: Path, solver_name: str = "reactingFoam") -> Path:
    """Generate a clean OpenFOAM case through FoamPilot's raw dictionary writer."""
    if not TEMPLATE.is_dir():
        raise FileNotFoundError(f"Missing input template: {TEMPLATE}")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    writer = OpenFOAMDictAddFile(object_name="dlfoam_hydrogen_planar")
    for source in sorted(TEMPLATE.rglob("*")):
        if not source.is_file() or source.name == "README.md":
            continue
        relative = source.relative_to(TEMPLATE)
        content = source.read_text(encoding="utf-8")
        content = _foam_header_version(content)
        if relative == Path("system/controlDict"):
            content = content.replace("application     reactingFoam;", f"application     {solver_name};")
        folder = str(relative.parent) if str(relative.parent) != "." else ""
        writer.write_raw(relative.name, case_dir, content, folder=folder)

    for directory in REQUIRED:
        if not (case_dir / directory).exists():
            raise RuntimeError(f"Generated case is incomplete: missing {directory}/")
    return case_dir


def command_available(command: str) -> bool:
    return shutil.which(command) is not None


def run_case(case_dir: Path, solver_name: str, nproc: int) -> None:
    """Run blockMesh, decomposition, solver and reconstruction through FoamPilot."""
    solver = Solver(case_dir)
    solver.run_command(["blockMesh"], "log.blockMesh")
    if nproc > 1:
        solver.run_command(["decomposePar", "-force"], "log.decomposePar")
        solver.run_command(["mpirun", "-np", str(nproc), solver_name, "-parallel"], f"log.{solver_name}")
        solver.run_command(["reconstructPar", "-latestTime"], "log.reconstructPar")
    else:
        solver.run_command([solver_name], f"log.{solver_name}")


def validate_case(case_dir: Path, solver_name: str) -> None:
    """Fail loudly when the requested solver did not produce a usable result."""
    log = case_dir / f"log.{solver_name}"
    if not log.exists():
        raise RuntimeError(f"Solver log was not produced: {log}")
    latest = sorted(
        (p for p in case_dir.iterdir() if p.is_dir() and p.name.replace(".", "", 1).isdigit()),
        key=lambda p: float(p.name),
    )
    if not latest:
        raise RuntimeError("The solver produced no numeric time directory")
    print(f"Validated latest OpenFOAM time: {latest[-1].name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--solver", default=os.environ.get("DLFOAM_SOLVER", "reactingFoam"))
    parser.add_argument("--np", type=int, default=1, help="MPI ranks; use 1 for serial execution")
    parser.add_argument("--run", action="store_true", help="run OpenFOAM after generating the case")
    args = parser.parse_args()

    case_dir = prepare_case(args.case_dir, args.solver)
    print(f"Generated case with FoamPilot: {case_dir}")
    if not args.run:
        print("Preparation only. Add --run to execute OpenFOAM.")
        return

    if not command_available("blockMesh"):
        raise RuntimeError("OpenFOAM 13 is not sourced: blockMesh is unavailable")
    if not command_available(args.solver):
        raise RuntimeError(f"Requested solver is unavailable: {args.solver}")
    run_case(case_dir, args.solver, args.np)
    validate_case(case_dir, args.solver)


if __name__ == "__main__":
    main()
