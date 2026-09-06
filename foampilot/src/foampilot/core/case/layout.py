"""Generic OpenFOAM case layout primitives.

This module intentionally has no dependency on workflows, examples, solvers,
or a particular OpenFOAM distribution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_CASE_DIRECTORIES = ("0", "0.orig", "constant", "system")


class CaseLayout:
    """Create and validate the filesystem layout of an OpenFOAM case."""

    def __init__(self, case_path: str | Path) -> None:
        self.case_path = Path(case_path).expanduser().resolve()

    def ensure(self, extra_directories: Iterable[str] = ("triSurface", "geometry", "postProcessing")) -> Path:
        """Create the standard case directories and return the resolved root."""
        for name in (*DEFAULT_CASE_DIRECTORIES, *tuple(extra_directories)):
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"invalid case subdirectory: {name!r}")
            (self.case_path / name).mkdir(parents=True, exist_ok=True)
        return self.case_path

    def validate(self, required: Iterable[str] = DEFAULT_CASE_DIRECTORIES) -> tuple[Path, ...]:
        """Return missing required paths without creating them."""
        return tuple(self.case_path / name for name in required if not (self.case_path / name).is_dir())


def create_case_structure(case_path: str | Path, *, extra_dirs: Iterable[str] = ("triSurface", "geometry", "postProcessing")) -> Path:
    """Compatibility-friendly function for creating a generic case layout."""
    return CaseLayout(case_path).ensure(extra_dirs)
