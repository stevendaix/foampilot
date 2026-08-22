"""Utilities for integrating external OpenFOAM 13 tutorial repositories.

The module deliberately keeps tutorial repositories as *inputs* and makes
FoamPilot the owner of case generation, execution and validation.  It does
not copy or edit tutorial dictionaries; instead it provides a manifest and a
safe execution boundary that can be used by declarative FoamPilot tutorials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Sequence


_LARGE_GEOMETRY_SUFFIXES = {".stl", ".step", ".stp", ".iges", ".igs", ".unv"}


@dataclass(frozen=True)
class TutorialSpec:
    """Metadata for one tutorial case discovered in a repository."""

    name: str
    family: str
    source_path: Path
    has_run_script: bool
    requires_external_geometry: bool
    geometry_files: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CaseValidation:
    """Result of validating a FoamPilot-generated OpenFOAM case."""

    case_path: Path
    required_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.missing_files and not self.warnings


class OpenFOAM13Environment:
    """Resolve and expose an OpenFOAM 13 shell environment.

    OpenFOAM is installed system-wide by the official Ubuntu package.  The
    helper loads ``etc/bashrc`` in a subprocess and returns the resulting
    environment, avoiding global shell mutation and making test runs
    reproducible.
    """

    def __init__(self, bashrc: str | Path = "/opt/openfoam13/etc/bashrc"):
        self.bashrc = Path(bashrc)

    def environment(self) -> dict[str, str]:
        if not self.bashrc.exists():
            raise FileNotFoundError(f"OpenFOAM 13 bashrc not found: {self.bashrc}")
        command = f". {self._quote(self.bashrc)} >/dev/null 2>&1 && env -0"
        result = subprocess.run(
            ["bash", "-lc", command], check=True, capture_output=True
        )
        environment = dict(
            item.split("=", 1)
            for item in result.stdout.decode().split("\0")
            if "=" in item
        )
        environment["FOAMPILOT_OPENFOAM_VERSION"] = "13"
        return environment

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        log_path: str | Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        env = self.environment()
        output = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            output = open(log_path, "w", encoding="utf-8")
        try:
            return subprocess.run(
                list(command), cwd=Path(cwd), env=env, text=True,
                stdout=output, stderr=subprocess.STDOUT if output else None,
                check=check,
            )
        finally:
            if output is not None:
                output.close()

    @staticmethod
    def _quote(path: Path) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"


class OpenFOAMTutorialManifest:
    """Discover and classify cases from ``OpenFOAMTutorials``."""

    def __init__(self, repository: str | Path):
        self.repository = Path(repository).resolve()

    def discover(self) -> tuple[TutorialSpec, ...]:
        cases = self.repository / "cases"
        if not cases.is_dir():
            raise FileNotFoundError(f"Tutorial cases directory not found: {cases}")
        discovered: list[TutorialSpec] = []
        for family_path in sorted(p for p in cases.iterdir() if p.is_dir()):
            for case_path in sorted(p for p in family_path.iterdir() if p.is_dir()):
                geometry = tuple(
                    str(path.relative_to(case_path))
                    for path in sorted(case_path.rglob("*"))
                    if path.is_file() and path.suffix.lower() in _LARGE_GEOMETRY_SUFFIXES
                )
                references_geometry = self._references_geometry(case_path)
                discovered.append(
                    TutorialSpec(
                        name=case_path.name,
                        family=family_path.name,
                        source_path=case_path,
                        has_run_script=(case_path / "run").is_file(),
                        requires_external_geometry=bool(geometry) or references_geometry,
                        geometry_files=geometry,
                    )
                )
        return tuple(discovered)

    @staticmethod
    def _references_geometry(case_path: Path) -> bool:
        for path in case_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {"", ".dict"}:
                continue
            try:
                text = path.read_text(errors="ignore").lower()
            except OSError:
                continue
            if any(token in text for token in (".stl", ".step", ".stp", ".unv")):
                return True
        return False


def validate_generated_case(
    case_path: str | Path,
    *,
    compressible: bool = False,
    required_files: Iterable[str] | None = None,
) -> CaseValidation:
    """Validate the minimum generated-case contract used by FoamPilot.

    The check is intentionally structural and deterministic: it catches
    missing dictionaries before an expensive solver run and explicitly
    verifies ``nu`` for incompressible cases.
    """

    root = Path(case_path).resolve()
    required = tuple(required_files or (
        "system/controlDict", "system/fvSchemes", "system/fvSolution",
        "constant", "0",
    ))
    missing = tuple(item for item in required if not (root / item).exists())
    warnings: list[str] = []
    if not compressible:
        transport = root / "constant" / "transportProperties"
        if not transport.exists():
            warnings.append("constant/transportProperties is missing for an incompressible case")
        elif "nu" not in transport.read_text(errors="ignore"):
            warnings.append("constant/transportProperties does not declare nu")
    control = root / "system" / "controlDict"
    if control.exists() and "application" not in control.read_text(errors="ignore"):
        warnings.append("system/controlDict does not declare application")
    return CaseValidation(root, required, missing, tuple(warnings))


def run_foampilot_case(
    solver,
    *,
    environment: OpenFOAM13Environment | None = None,
    processors: int = 1,
    log_filename: str | None = None,
) -> None:
    """Write and run a configured FoamPilot solver using OpenFOAM 13."""

    solver.setup_case()
    solver.write_case()
    validation = validate_generated_case(
        solver.case_path,
        compressible=getattr(solver, "compressible", False),
    )
    if not validation.valid:
        raise ValueError(
            "Generated case failed validation: "
            + "; ".join((*validation.missing_files, *validation.warnings))
        )
    (environment or OpenFOAM13Environment()).run(
        ["foamRun", "-solver", solver.foamrun_module],
        cwd=solver.case_path,
        log_path=Path(solver.case_path) / (log_filename or "log.foamRun"),
    )
