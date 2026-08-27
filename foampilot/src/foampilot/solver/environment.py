"""OpenFOAM environment resolution for FoamPilot-managed commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Mapping


@dataclass(frozen=True)
class OpenFOAMEnvironment:
    """Resolve an OpenFOAM installation without mutating the parent shell.

    The object is intentionally limited to environment resolution. Command
    execution remains the responsibility of ``BaseSolver`` so FoamPilot does
    not introduce a second execution abstraction over the existing solver API.
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
            environment.update({str(key): str(value) for key, value in self.overrides.items()})
        return environment

    @staticmethod
    def _quote(path: Path) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"


__all__ = ["OpenFOAMEnvironment"]
