"""Python replacement for Dakota in the geometric-variation tutorial.

The default campaign reproduces dakota.in: ten Latin-Hypercube samples over
angle1, angle2 and length. Each sample is run in an isolated case directory,
and the two Dakota responses are written to ``python_optimization.csv`` and
``python_optimization.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
RUNS = ROOT / "python_optimization_runs"
DEFAULT_TARGET = 320.0


@dataclass(frozen=True)
class Design:
    angle1: float
    angle2: float
    length: float


@dataclass
class Evaluation:
    sample: int
    angle1: float
    angle2: float
    length: float
    tmin: float | None = None
    tmax: float | None = None
    taverage: float | None = None
    objective_average: float | None = None
    objective_distribution: float | None = None
    status: str = "pending"
    error: str = ""


class OptimizationError(RuntimeError):
    pass


def latin_hypercube(samples: int, dimensions: int, seed: int) -> list[list[float]]:
    """Return a deterministic Latin-Hypercube design in [0, 1]^dimensions."""
    try:
        from scipy.stats import qmc

        return qmc.LatinHypercube(d=dimensions, seed=seed).random(samples).tolist()
    except ImportError:
        import random

        rng = random.Random(seed)
        result = [[(i + rng.random()) / samples for i in range(samples)] for _ in range(dimensions)]
        for column in result:
            rng.shuffle(column)
        return [list(row) for row in zip(*result)]


def designs(samples: int, seed: int) -> list[Design]:
    lower = (0.0, 0.0, 0.005)
    upper = (180.0, 180.0, 0.03)
    return [
        Design(*(lo + fraction * (hi - lo) for lo, hi, fraction in zip(lower, upper, row)))
        for row in latin_hypercube(samples, 3, seed)
    ]


def write_case(case: Path, design: Design) -> None:
    if case.exists():
        shutil.rmtree(case)
    writer = OpenFOAMDictAddFile(object_name=f"python_optimization_{case.name}")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            text = source.read_text(encoding="utf-8")
            if relative.as_posix() == "system/blockMeshDict":
                half = design.length / 2.0
                x1 = 0.035 - half
                x2 = 0.035 + half
                text = re.sub(r"^x1\s+[^;]+;", f"x1 {x1:.9g};", text, flags=re.MULTILINE)
                text = re.sub(r"^x2\s+[^;]+;", f"x2 {x2:.9g};", text, flags=re.MULTILINE)
            writer.write_raw(relative.name, case, text, folder=str(relative.parent))


def command(solver: Solver, cmd: list[str], log: str) -> None:
    solver.run_command(cmd, log)


def prepare_geometry(case: Path, design: Design) -> None:
    required = (
        case / "constant" / "triSurface" / "baffle1_original.stl",
        case / "constant" / "triSurface" / "baffle2_original.stl",
    )
    missing = [str(path.relative_to(case)) for path in required if not path.is_file()]
    if missing:
        raise OptimizationError("Missing geometry assets: " + ", ".join(missing))
    rotate = case / "system" / "rotateBaffles"
    rotate.write_text(
        (case / "system" / "rotateBafflesDict")
        .read_text(encoding="utf-8")
        .replace("angle1", f"{design.angle1:.9g}")
        .replace("angle2", f"{design.angle2:.9g}"),
        encoding="utf-8",
    )
    subprocess.run(
        ["bash", "-e", str(rotate)],
        cwd=case,
        check=True,
        env=Solver(case)._command_environment(),
    )


def parse_number(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def postprocess_value(case: Path, command_name: str, fallback_file: str) -> float | None:
    output = case / fallback_file
    if output.is_file():
        value = parse_number(output.read_text(encoding="utf-8", errors="replace"))
        if value is not None:
            return value
    result = subprocess.run(
        ["postProcess", "-latestTime", "-func", command_name],
        cwd=case,
        env=Solver(case)._command_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    return parse_number(result.stdout + "\n" + result.stderr)


def average_temperature(case: Path) -> float | None:
    candidates = sorted(case.glob("postProcessing/Taverage/*/surfaceFieldValue.dat"))
    if not candidates:
        return None
    lines = [line for line in candidates[-1].read_text(encoding="utf-8", errors="replace").splitlines() if line and not line.startswith("#")]
    if not lines:
        return None
    fields = lines[-1].split()
    return float(fields[-1]) if fields else None


def evaluate(sample: int, design: Design, target: float) -> Evaluation:
    case = RUNS / f"sample_{sample:04d}"
    result = Evaluation(sample, design.angle1, design.angle2, design.length)
    try:
        write_case(case, design)
        solver = Solver(case)
        prepare_geometry(case, design)
        command(solver, ["blockMesh"], "log.blockMesh")
        command(solver, ["extrudeMesh"], "log.extrudeMesh")
        command(solver, ["createPatch", "-overwrite"], "log.createPatch")
        command(solver, ["renumberMesh", "-constant", "-overwrite"], "log.renumberMesh")
        command(solver, ["foamRun"], "log.foamRun")
        result.tmin = postprocess_value(case, "patchOutletMin", "Tmin")
        result.tmax = postprocess_value(case, "patchOutletMax", "Tmax")
        result.taverage = average_temperature(case)
        if result.taverage is None or result.tmin is None or result.tmax is None:
            raise OptimizationError("Unable to extract outlet temperature responses")
        result.objective_average = abs(target - result.taverage)
        result.objective_distribution = abs(result.tmax - result.tmin)
        result.status = "success"
    except (OSError, subprocess.CalledProcessError, OptimizationError, ValueError) as exc:
        result.status = "failed"
        result.error = str(exc)
    return result


def write_results(results: Iterable[Evaluation], output_dir: Path) -> None:
    rows = [asdict(result) for result in results]
    (output_dir / "python_optimization.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "python_optimization.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else list(Evaluation.__annotations__))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10, help="number of Latin-Hypercube evaluations")
    parser.add_argument("--seed", type=int, default=124523, help="deterministic sampling seed")
    parser.add_argument("--target-temperature", type=float, default=DEFAULT_TARGET, help="target outlet mean temperature")
    parser.add_argument("--keep-runs", action="store_true", help="keep isolated case directories")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    RUNS.mkdir(parents=True, exist_ok=True)
    results = [evaluate(index, design, args.target_temperature) for index, design in enumerate(designs(args.samples, args.seed), 1)]
    write_results(results, ROOT)
    successful = [item for item in results if item.status == "success"]
    if successful:
        best = min(successful, key=lambda item: (item.objective_average or math.inf, item.objective_distribution or math.inf))
        print(json.dumps({"successful": len(successful), "total": len(results), "best": asdict(best)}, indent=2))
    else:
        print(f"No successful evaluations out of {len(results)}")
    if not args.keep_runs:
        for child in RUNS.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
