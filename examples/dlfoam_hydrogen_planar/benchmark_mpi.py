#!/usr/bin/env python3
"""Benchmark reproducible for multicomponentFluid with OpenMPI."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run.py"
OUT = ROOT / "benchmark_results"
BASE = OUT / "base"
NPROCS = (1, 2, 3)
NX = 2
NY = 2
END_TIME = 1.1e-6
WRITE_INTERVAL = 1


def run(cmd: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> float:
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def write_decompose_dict(case: Path, nproc: int) -> None:
    """Use a decomposition compatible with the tested rank count."""
    (case / "system" / "decomposeParDict").write_text(
        """FoamFile\n{\n    version 2;\n    format ascii;\n    class dictionary;\n    object decomposeParDict;\n}\nnumberOfSubdomains %d;\nmethod simple;\nsimpleCoeffs\n{\n    n (%d 1 1);\n    delta 1e-10;\n}\n""" % (nproc, nproc),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if BASE.exists():
        shutil.rmtree(BASE)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT.parents[1] / "foampilot" / "src")
    run(
        ["python3", str(RUNNER), "--case-dir", str(BASE), "--nx", str(NX), "--ny", str(NY),
         "--end-time", str(END_TIME), "--write-interval", str(WRITE_INTERVAL)],
        ROOT, OUT / "prepare.log", env,
    )
    records: list[dict[str, object]] = []
    for nproc in NPROCS:
        case = OUT / f"np{nproc}"
        if case.exists():
            shutil.rmtree(case)
        shutil.copytree(BASE, case)
        write_decompose_dict(case, nproc)
        mech = case / "constant" / "mech"
        build_time = run(["bash", "./Allwmake"], mech, OUT / f"np{nproc}.mechanism.log", env)
        mesh_time = run(["blockMesh"], case, OUT / f"np{nproc}.blockMesh.log", env)
        solver_log = OUT / f"np{nproc}.solver.log"
        solver = ["foamRun", "-solver", "multicomponentFluid"]
        if nproc == 1:
            command = solver
        else:
            subprocess.run(["decomposePar", "-force"], cwd=case, env=env, check=True,
                           stdout=(OUT / f"np{nproc}.decompose.log").open("w"),
                           stderr=subprocess.STDOUT)
            command = ["mpirun", "--allow-run-as-root", "-np", str(nproc), *solver, "-parallel"]
        solver_time = run(command, case, solver_log, env)
        if nproc > 1:
            run(["reconstructPar", "-latestTime"], case, OUT / f"np{nproc}.reconstruct.log", env)
        log_text = solver_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"ExecutionTime = ([0-9.eE+-]+) s", log_text)
        records.append({
            "nproc": nproc,
            "cells": NX * NY,
            "wall_seconds_solver_command": solver_time,
            "foam_execution_seconds": float(match.group(1)) if match else None,
            "build_seconds": build_time,
            "blockmesh_seconds": mesh_time,
            "completed": "End" in log_text,
            "latest_time": str(max((p for p in case.iterdir() if p.is_dir() and p.name.replace('.', '', 1).isdigit()), key=lambda p: float(p.name)).name),
        })
    (OUT / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
