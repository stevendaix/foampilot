"""Validation of Foundation OpenFOAM 13 marine case inputs.

This module intentionally performs structural checks only.  It does not parse
all OpenFOAM syntax and it does not replace ``checkMesh`` or the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarineCaseConfig:
    """Validated high-level configuration for a marine case."""

    case_path: Path
    mode: str
    solver: str

    VALID_MODES = frozenset({"dtc_moving", "maneuvering", "propeller_mrf"})

    @classmethod
    def from_case(cls, case_path: str | Path) -> "MarineCaseConfig":
        root = Path(case_path)
        control = root / "system" / "controlDict"
        if not control.is_file():
            raise FileNotFoundError(f"Missing required file: {control}")

        text = control.read_text(encoding="utf-8")
        solver = _lookup_word(text, "solver") or _lookup_word(text, "application")
        if not solver:
            raise ValueError("controlDict must define solver or application")

        marine = root / "constant" / "marineProperties"
        mode = "dtc_moving"
        if marine.is_file():
            mode = _lookup_word(marine.read_text(encoding="utf-8"), "mode") or mode

        if mode not in cls.VALID_MODES:
            allowed = ", ".join(sorted(cls.VALID_MODES))
            raise ValueError(f"Unsupported marine mode {mode!r}; expected one of: {allowed}")

        return cls(root, mode, solver)

    def required_files(self) -> tuple[Path, ...]:
        """Return the minimum structural files for the selected mode."""
        required = [
            self.case_path / "system" / "controlDict",
            self.case_path / "system" / "fvSchemes",
            self.case_path / "system" / "fvSolution",
            self.case_path / "constant" / "g",
        ]
        if self.mode == "dtc_moving":
            required.append(self.case_path / "constant" / "dynamicMeshDict")
        elif self.mode == "maneuvering":
            required.extend(
                (
                    self.case_path / "constant" / "dynamicMeshDict",
                    self.case_path / "constant" / "marineControls",
                )
            )
        elif self.mode == "propeller_mrf":
            required.extend(
                (
                    self.case_path / "constant" / "MRFProperties",
                    self.case_path / "constant" / "fvModels",
                )
            )
        return tuple(required)

    def validate_files(self) -> None:
        missing = [path for path in self.required_files() if not path.is_file()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Marine case is incomplete:\n{formatted}")

    def validate_foundation13(self) -> None:
        """Reject legacy OpenCFD overset entries that are not Foundation 13 native."""
        control = (self.case_path / "system" / "controlDict").read_text(encoding="utf-8")
        dynamic = self.case_path / "constant" / "dynamicMeshDict"
        dynamic_text = dynamic.read_text(encoding="utf-8") if dynamic.is_file() else ""
        legacy_markers = ("overInterDyMFoam", "dynamicOversetFvMesh", "rigidBodyDynamics")
        found = [marker for marker in legacy_markers if marker in control or marker in dynamic_text]
        if found:
            markers = ", ".join(found)
            raise ValueError(
                f"Legacy OpenCFD overset entries are not native Foundation 13: {markers}. "
                "Use incompressibleVoF with the Foundation 13 mover or port an overset runtime."
            )


def _lookup_word(text: str, key: str) -> str | None:
    """Read a simple unquoted OpenFOAM word entry from text."""
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line or not line.startswith(key):
            continue
        fields = line.rstrip(";").split()
        if len(fields) >= 2 and fields[0] == key:
            return fields[1].strip('"')
    return None
