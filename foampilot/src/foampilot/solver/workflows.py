"""User-oriented OpenFOAM execution workflows.

The classes in this module are deliberately thin facades over the existing
``BaseSolver`` primitives.  They centralise environment resolution and common
execution workflows without introducing tutorial-specific behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
from typing import Mapping, Sequence

from .base_solver import BaseSolver


@dataclass(frozen=True)
class OpenFOAMEnvironment:
    """Resolve an OpenFOAM installation without mutating the parent shell.

    Parameters
    ----------
    bashrc:
        Path to the installation ``etc/bashrc``.  If omitted, the standard
        Foundation 13 package path is used.
    version:
        Optional version label exposed as ``FOAMPILOT_OPENFOAM_VERSION``.
    overrides:
        Explicit environment values applied after sourcing ``bashrc``.
    """

    bashrc: Path | str = Path("/opt/openfoam13/etc/bashrc")
    version: str | None = "13"
    overrides: Mapping[str, str] | None = None

    def environment(self) -> dict[str, str]:
        bashrc = Path(self.bashrc)
        if not bashrc.is_file():
            raise FileNotFoundError(f"OpenFOAM bashrc not found: {bashrc}")
        command = f". {self._quote(bashrc)} >/dev/null 2>&1 && env -0"
        result = subprocess.run(
            ["bash", "-lc", command], check=True, capture_output=True
        )
        environment = dict(
            item.split("=", 1)
            for item in result.stdout.decode().split("\0")
            if "=" in item
        )
        if self.version is not None:
            environment["FOAMPILOT_OPENFOAM_VERSION"] = str(self.version)
        if self.overrides:
            environment.update({str(k): str(v) for k, v in self.overrides.items()})
        return environment

    @staticmethod
    def _quote(path: Path) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"


class RunWorkflow:
    """High-level execution facade attached to one :class:`BaseSolver`.

    ``run_command`` remains available for uncommon OpenFOAM operations.  This
    facade provides names for the common user intents and ensures all commands
    use one resolved environment.
    """

    def __init__(
        self,
        solver: BaseSolver,
        *,
        environment: OpenFOAMEnvironment | Mapping[str, str] | None = None,
    ) -> None:
        self.solver = solver
        self._environment = environment

    @property
    def environment(self) -> dict[str, str]:
        if isinstance(self._environment, OpenFOAMEnvironment):
            return self._environment.environment()
        if self._environment is not None:
            return {str(k): str(v) for k, v in self._environment.items()}
        return os.environ.copy()

    def utility(
        self,
        name: str,
        args: Sequence[str] = (),
        *,
        log_filename: str | None = None,
    ) -> None:
        """Run a named OpenFOAM utility in the managed case."""
        if not name or Path(name).name != name:
            raise ValueError("utility name must be a single executable name")
        log = log_filename or f"log.{name}"
        self.solver.run_command([name, *map(str, args)], log, self.environment)

    def foam_run(
        self,
        module: str | None = None,
        *,
        log_filename: str | None = None,
    ) -> None:
        """Run ``foamRun -solver <module>`` for the managed solver."""
        selected = module or self.solver.foamrun_module
        if not selected or Path(selected).name != selected:
            raise ValueError("solver module must be a single name")
        log = log_filename or f"log.{selected}"
        self.solver.run_command(["foamRun", "-solver", selected], log, self.environment)

    def serial(self, *, log_filename: str | None = None) -> None:
        """Run the configured solver once in serial mode."""
        self.foam_run(log_filename=log_filename)

    def parallel(
        self,
        processes: int,
        *,
        decompose: bool = True,
        reconstruct: bool = True,
        log_filename: str | None = None,
    ) -> None:
        """Run the configured ``foamRun`` solver through an MPI workflow."""
        if processes < 2:
            raise ValueError("parallel execution requires at least two processes")
        log = log_filename or f"log.{self.solver.foamrun_module}.parallel"
        if decompose:
            self.solver.run_command(
                ["decomposePar", "-force"], log + ".decompose", self.environment
            )
        self.solver.run_command(
            [
                "mpirun", "--oversubscribe", "-np", str(processes),
                "foamRun", "-solver", self.solver.foamrun_module, "-parallel",
            ],
            log,
            self.environment,
        )
        if reconstruct:
            self.solver.run_command(
                ["reconstructPar"], log + ".reconstruct", self.environment
            )

    def run(self, *, processes: int = 1, **kwargs: object) -> None:
        """Run serially or in parallel based on ``processes``."""
        if processes == 1:
            self.serial(log_filename=kwargs.get("log_filename"))
        elif processes > 1:
            self.parallel(processes, **kwargs)
        else:
            raise ValueError("processes must be positive")


__all__ = ["OpenFOAMEnvironment", "RunWorkflow"]
