#!/usr/bin/env python3
"""Run the OpenFOAM 13 DamBreak VOF tutorial entirely through FoamPilot.

FoamPilot writes the input dictionaries, creates the mesh through its
``BaseSolver.run_command`` API, validates the case, then calls its standard
``run_simulation`` API.  The script does not call ``foamRun`` or ``Allrun``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "foampilot" / "src"))

from foampilot.solver.base_solver import BaseSolver
from foampilot.tutorials import OpenFOAM13Environment, validate_generated_case
from foampilot.utilities import OpenFOAMDictAddFile


MODULE = "incompressibleVoF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-case", type=Path,
        default=Path("/opt/openfoam13/tutorials/incompressibleVoF/damBreakLaminar"),
        help="Official OpenFOAM 13 DamBreak source case.",
    )
    parser.add_argument("--run-root", type=Path, default=ROOT / ".runs/dambreak_vof")
    parser.add_argument("--end-time", type=float, default=0.05)
    parser.add_argument("--write-interval", type=float, default=0.01)
    return parser.parse_args()


def set_entry(content: str, key: str, value: str | int | float) -> str:
    pattern = rf"^(?P<prefix>\s*{re.escape(key)}\s+)[^;]+;"
    rendered, count = re.subn(pattern, rf"\g<prefix>{value};", content, flags=re.MULTILINE)
    if count == 1:
        return rendered
    if count > 1:
        raise ValueError(f"Multiple {key} entries in controlDict")
    return content.rstrip() + f"\n{key} {value};\n"


def write_foampilot_inputs(source: Path, target: Path, end_time: float, write_interval: float) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    records: list[dict[str, str]] = []
    for directory in ("0", "constant", "system"):
        for path in sorted((source / directory).rglob("*")):
            if not path.is_file() or "polyMesh" in path.parts or "dynamicCode" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            relative = path.relative_to(source)
            if str(relative) == "system/controlDict":
                content = set_entry(content, "application", "foamRun")
                content = set_entry(content, "solver", MODULE)
                content = set_entry(content, "endTime", end_time)
                content = set_entry(content, "writeInterval", write_interval)
            OpenFOAMDictAddFile(path.name).write_raw(path.name, target, content, folder=str(relative.parent))
            records.append({
                "path": str(relative),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "role": "field" if directory == "0" else "model" if directory == "constant" else "numerics_or_mesh",
            })
    (target / "foampilot-input-manifest.json").write_text(
        json.dumps({
            "generator": "FoamPilot",
            "executor": "BaseSolver.run_simulation",
            "openfoam_target": "13",
            "solver_module": MODULE,
            "inputs": records,
            "mesh_command": ["blockMesh"],
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source_case.resolve()
    target = (args.run_root / "damBreak").resolve()
    write_foampilot_inputs(source, target, args.end_time, args.write_interval)

    environment = OpenFOAM13Environment().environment()
    os.environ.update(environment)
    solver = BaseSolver(
        case_path=target,
        solver_name=MODULE,
        compressible=False,
        with_gravity=True,
        is_vof=True,
        transient=True,
        turbulence_model="kEpsilon",
    )
    solver.setup_case()
    validation = validate_generated_case(target, is_vof=True)
    if not validation.valid:
        raise SystemExit("Invalid FoamPilot case: " + "; ".join((*validation.missing_files, *validation.warnings)))
    solver.run_command(["blockMesh"], "log.blockMesh")
    solver.run_command(["checkMesh"], "log.checkMesh")
    solver.run_simulation(log_filename="log.foampilot")
    print(f"FoamPilot completed DamBreak VOF: {target}")


if __name__ == "__main__":
    main()
