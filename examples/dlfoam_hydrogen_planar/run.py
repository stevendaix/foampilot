"""Reproduction Foampilot du tutoriel DLBFoam 2D planar hydrogen flame.

Le dossier ``case_template`` contient uniquement les données d'entrée publiées.
Tous les dictionnaires et champs sont ré-écrits dans ``case`` par
OpenFOAMDictAddFile avant l'exécution. Le calcul réactif nécessite les
bibliothèques DLBFoam/FickianTransportFoam et le mécanisme PyJac compilé.
"""
from __future__ import annotations

import argparse
import os
import re
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


def prepare_case(
    case_dir: Path,
    solver_name: str = "foamRun",
    nx: int | None = None,
    ny: int | None = None,
    end_time: float | None = None,
    write_interval: int | None = None,
) -> Path:
    """Generate a clean OpenFOAM case through FoamPilot's raw dictionary writer."""
    if not TEMPLATE.is_dir():
        raise FileNotFoundError(f"Missing input template: {TEMPLATE}")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    writer = OpenFOAMDictAddFile(object_name="dlfoam_hydrogen_planar")
    for source in sorted(TEMPLATE.rglob("*")):
        relative = source.relative_to(TEMPLATE)
        # Build products are machine-specific and must never be copied or committed.
        if (not source.is_file() or source.name == "README.md"
                or ("Make" in relative.parts and source.name not in {"files", "options"})
                or "lib" in relative.parts):
            continue
        content = source.read_text(encoding="utf-8")
        # OpenFOAM 13 gives an included file its own dictionary name. Expand
        # these fragments here so each field keeps its own profile (H2.dat,
        # T.dat, U.dat, ...) instead of trying readInternalScalarField.H.dat.
        for include_name in (
            "readInletValue.H",
            "readInternalScalarField.H",
            "readInternalVectorXField.H",
        ):
            include_path = TEMPLATE / "0" / include_name
            if include_path.is_file() and include_name in content:
                include_text = include_path.read_text(encoding="utf-8")
                content = re.sub(
                    rf'#include\s+"{re.escape(include_name)}";?',
                    include_text,
                    content,
                )
        content = _foam_header_version(content)
        content = content.replace(
            "coefficientWilkeMultiComponentMixture",
            "coefficientWilkeMulticomponentMixture",
        )
        content = content.replace(
            '#include "fvCFD.H"',
            '#include "fvMesh.H"\n        #include "IOdictionary.H"\n        #include "volFields.H"\n        #include <fstream>',
        )
        content = content.replace("ifstream dataFile", "std::ifstream dataFile")
        content = content.replace("std::std::ifstream", "std::ifstream")
        content = content.replace("Info <<", "Foam::Info <<")
        content = content.replace("Foam::Foam::Info <<", "Foam::Info <<")
        content = content.replace(" << endl", " << Foam::endl")
        # OpenFOAM 13 resolves dict.name() to the included fragment. Bind the
        # published profile explicitly to the field being generated instead.
        profile_name = f"{source.name}.dat"
        profile = f'"{profile_name}"'
        profile_path = f'(static_cast<const IOdictionary&>(dict).db().time().path() / "0" / "{profile_name}").c_str()'
        content = content.replace('dict.name() + ".dat"', profile)
        content = content.replace('dict.name()+".dat"', profile)
        content = content.replace('dict.parent().name() + ".dat"', profile)
        content = content.replace('dict.parent().name()+".dat"', profile)
        content = content.replace(f'std::ifstream dataFile({profile});',
                                  f'std::ifstream dataFile({profile_path});')
        content = content.replace(f'read_2_column_data_file({profile},',
                                  f'read_2_column_data_file({profile_path},')
        content = content.replace("$(LIB_SRC)/finiteVolume/lnInclude",
                                  "$(WM_PROJECT_DIR)/src/finiteVolume/lnInclude")
        content = content.replace("$(LIB_SRC)/meshTools/lnInclude",
                                  "$(WM_PROJECT_DIR)/src/meshTools/lnInclude")
        if "$(FOAM_CASE)/0" in content and "$(WM_PROJECT_DIR)/src/OpenFOAM/lnInclude" not in content:
            content = content.replace(
                "codeOptions\n    #{",
                "codeOptions\n    #{\n        -I$(WM_PROJECT_DIR)/src/OpenFOAM/lnInclude \\",
            )
        if relative == Path("system/controlDict"):
            content = content.replace("application     reactingFoam;", f"application     {solver_name};")
            if end_time is not None:
                content = re.sub(r"endTime\s+[^;]+;", f"endTime         {end_time};", content)
            if write_interval is not None:
                content = re.sub(r"writeInterval\s+[^;]+;", f"writeInterval   {write_interval};", content)
        if relative == Path("system/blockMeshDict"):
            if nx is not None:
                content = content.replace("Nx              2000;", f"Nx              {nx};")
            if ny is not None:
                content = content.replace("Ny              2000;", f"Ny              {ny};")
        folder = str(relative.parent) if str(relative.parent) != "." else ""
        writer.write_raw(relative.name, case_dir, content, folder=folder)

    for directory in REQUIRED:
        if not (case_dir / directory).exists():
            raise RuntimeError(f"Generated case is incomplete: missing {directory}/")
    return case_dir


def command_available(command: str) -> bool:
    return shutil.which(command) is not None


def run_case(
    case_dir: Path,
    solver_name: str,
    nproc: int,
    foam_solver: str = "multicomponentFluid",
) -> None:
    """Build the mechanism and run mesh, solver and reconstruction through FoamPilot."""
    solver = Solver(case_dir)
    mechanism = case_dir / "constant" / "mech"
    if not (mechanism / "Allwmake").is_file():
        raise RuntimeError(f"PyJac build script is missing: {mechanism / 'Allwmake'}")
    subprocess.run(["bash", "./Allwmake"], cwd=mechanism, check=True)
    solver.run_command(["blockMesh"], "log.blockMesh")
    if nproc > 1:
        solver.run_command(["decomposePar", "-force"], "log.decomposePar")
        command = ["mpirun", "-np", str(nproc), solver_name]
        if solver_name == "foamRun":
            command += ["-solver", foam_solver]
        command += ["-parallel"]
        solver.run_command(command, f"log.{solver_name}")
        solver.run_command(["reconstructPar", "-latestTime"], "log.reconstructPar")
    else:
        command = [solver_name]
        if solver_name == "foamRun":
            command += ["-solver", foam_solver]
        solver.run_command(command, f"log.{solver_name}")


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
    parser.add_argument("--solver", default=os.environ.get("DLFOAM_SOLVER", "foamRun"))
    parser.add_argument("--foam-solver", default="multicomponentFluid")
    parser.add_argument("--np", type=int, default=1, help="MPI ranks; use 1 for serial execution")
    parser.add_argument("--run", action="store_true", help="run OpenFOAM after generating the case")
    parser.add_argument("--nx", type=int, help="optional mesh override for a bounded smoke test")
    parser.add_argument("--ny", type=int, help="optional mesh override for a bounded smoke test")
    parser.add_argument("--end-time", type=float, help="optional endTime override for a bounded smoke test")
    parser.add_argument("--write-interval", type=int, help="optional writeInterval override for a bounded smoke test")
    args = parser.parse_args()

    if (args.nx is not None and args.nx < 1) or (args.ny is not None and args.ny < 1):
        parser.error("--nx and --ny must be positive")
    if args.end_time is not None and args.end_time <= 0:
        parser.error("--end-time must be positive")
    if args.write_interval is not None and args.write_interval < 1:
        parser.error("--write-interval must be positive")
    case_dir = prepare_case(
        args.case_dir, args.solver, args.nx, args.ny,
        args.end_time, args.write_interval,
    )
    print(f"Generated case with FoamPilot: {case_dir}")
    if not args.run:
        print("Preparation only. Add --run to execute OpenFOAM.")
        return

    if not command_available("blockMesh"):
        raise RuntimeError("OpenFOAM 13 is not sourced: blockMesh is unavailable")
    if not command_available(args.solver):
        raise RuntimeError(f"Requested solver is unavailable: {args.solver}")
    run_case(case_dir, args.solver, args.np, args.foam_solver)
    validate_case(case_dir, args.solver)


if __name__ == "__main__":
    main()
