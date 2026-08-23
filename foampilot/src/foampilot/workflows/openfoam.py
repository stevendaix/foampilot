"""Reusable orchestration primitives for OpenFOAM case workflows.

The standard ``foamRun`` interface is convenient for OpenFOAM Foundation releases
using solver modules.  A substantial number of published tutorials, however,
are written for OpenCFD releases and invoke legacy executables directly (for
example ``overInterDyMFoam`` and ``rhoSimpleFoam``).  This module keeps those
workflows explicit, inspectable and testable from Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import os
import shutil
import subprocess


@dataclass(frozen=True)
class CommandStep:
    """One OpenFOAM command executed from a case-relative directory.

    Parameters
    ----------
    name:
        Stable identifier used in logs and reports.
    command:
        Executable followed by its arguments.  Shell parsing is intentionally
        not supported so commands remain portable and auditable.
    cwd:
        Directory relative to the workflow root where the command is run.
    environment:
        Optional environment additions for the command.
    """

    name: str
    command: tuple[str, ...]
    cwd: Path | str = Path(".")
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A workflow step requires a non-empty name.")
        if not self.command or not self.command[0].strip():
            raise ValueError(f"Workflow step '{self.name}' has no executable.")
        object.__setattr__(self, "cwd", Path(self.cwd))
        object.__setattr__(self, "command", tuple(self.command))


@dataclass(frozen=True)
class CopyStep:
    """Copy a file or a directory inside a case workflow.

    Both paths are resolved relative to the workflow root.  Copying is
    deliberately modelled as an explicit step because dynamic-mesh studies
    commonly switch dictionaries between preparation and simulation phases.
    """

    name: str
    source: Path | str
    destination: Path | str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A workflow step requires a non-empty name.")
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "destination", Path(self.destination))


@dataclass(frozen=True)
class RemoveStep:
    """Remove case-generated paths before a deterministic rerun."""

    name: str
    paths: tuple[Path | str, ...]
    cwd: Path | str = Path(".")

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A workflow step requires a non-empty name.")
        object.__setattr__(self, "paths", tuple(Path(path) for path in self.paths))
        object.__setattr__(self, "cwd", Path(self.cwd))


@dataclass(frozen=True)
class RestoreInitialFieldsStep:
    """Restore OpenFOAM initial fields from ``0.orig`` or ``*.orig`` files.

    Official OpenFOAM tutorials commonly preserve initial fields either in a
    sibling ``0.orig`` directory or as ``.orig`` files inside ``0``. This step
    reproduces ``restore0Dir`` without sourcing the shell-only RunFunctions.
    """

    name: str
    source_directory: Path | str = Path("0.orig")
    destination_directory: Path | str = Path("0")

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A workflow step requires a non-empty name.")
        object.__setattr__(self, "source_directory", Path(self.source_directory))
        object.__setattr__(self, "destination_directory", Path(self.destination_directory))


WorkflowStep = CommandStep | CopyStep | RemoveStep | RestoreInitialFieldsStep


@dataclass(frozen=True)
class StepResult:
    """Result of one workflow step."""

    name: str
    status: str
    detail: str


class OpenFOAMWorkflow:
    """Execute an auditable ordered sequence of OpenFOAM case operations.

    The class does not generate shell scripts or depend on OpenFOAM's
    ``RunFunctions`` helper.  Consequently, a workflow can be previewed on a
    machine without OpenFOAM and executed with both Foundation and OpenCFD
    installations when the referenced commands are available.
    """

    def __init__(self, root: str | Path, name: str = "openfoam-workflow") -> None:
        self.root = Path(root).expanduser().resolve()
        self.name = name
        self._steps: list[WorkflowStep] = []

    @property
    def steps(self) -> tuple[WorkflowStep, ...]:
        """Immutable view of the ordered workflow steps."""
        return tuple(self._steps)

    def add_command(
        self,
        name: str,
        *command: str,
        cwd: str | Path = ".",
        environment: Mapping[str, str] | None = None,
    ) -> "OpenFOAMWorkflow":
        """Append a direct executable invocation to the workflow."""
        self._steps.append(
            CommandStep(
                name=name,
                command=tuple(command),
                cwd=Path(cwd),
                environment=environment or {},
            )
        )
        return self

    def add_copy(
        self,
        name: str,
        source: str | Path,
        destination: str | Path,
        *,
        required: bool = True,
    ) -> "OpenFOAMWorkflow":
        """Append a file/directory copy step."""
        self._steps.append(
            CopyStep(name=name, source=Path(source), destination=Path(destination), required=required)
        )
        return self

    def add_remove(
        self,
        name: str,
        *paths: str | Path,
        cwd: str | Path = ".",
    ) -> "OpenFOAMWorkflow":
        """Append deterministic cleanup of files or directories."""
        self._steps.append(RemoveStep(name=name, paths=tuple(paths), cwd=Path(cwd)))
        return self

    def add_restore_initial_fields(
        self,
        name: str = "restore-initial-fields",
        *,
        source_directory: str | Path = "0.orig",
        destination_directory: str | Path = "0",
    ) -> "OpenFOAMWorkflow":
        """Append restoration of OpenFOAM initial fields from ``.orig`` data."""
        self._steps.append(
            RestoreInitialFieldsStep(
                name=name,
                source_directory=Path(source_directory),
                destination_directory=Path(destination_directory),
            )
        )
        return self

    def validate(self) -> None:
        """Check the internal workflow declaration before any execution."""
        if not self.name.strip():
            raise ValueError("A workflow requires a non-empty name.")
        names = [step.name for step in self._steps]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError("Workflow step names must be unique: " + ", ".join(duplicates))
        for step in self._steps:
            if isinstance(step, CommandStep) and step.cwd.is_absolute():
                raise ValueError(f"Step '{step.name}' must use a relative working directory.")
            if isinstance(step, RemoveStep) and step.cwd.is_absolute():
                raise ValueError(f"Step '{step.name}' must use a relative working directory.")
            if isinstance(step, CopyStep) and (step.source.is_absolute() or step.destination.is_absolute()):
                raise ValueError(f"Copy step '{step.name}' must use paths relative to the workflow root.")
            if isinstance(step, RestoreInitialFieldsStep) and (
                step.source_directory.is_absolute() or step.destination_directory.is_absolute()
            ):
                raise ValueError(f"Restore step '{step.name}' must use paths relative to the workflow root.")

    def preview(self) -> str:
        """Render a shell-independent textual representation of the workflow."""
        self.validate()
        lines = [f"Workflow: {self.name}", f"Root: {self.root}"]
        for index, step in enumerate(self._steps, start=1):
            if isinstance(step, CommandStep):
                lines.append(
                    f"{index:02d}. [{step.name}] cd {step.cwd} && {' '.join(step.command)}"
                )
            elif isinstance(step, CopyStep):
                lines.append(
                    f"{index:02d}. [{step.name}] copy {step.source} -> {step.destination}"
                )
            elif isinstance(step, RestoreInitialFieldsStep):
                lines.append(
                    f"{index:02d}. [{step.name}] restore {step.source_directory} -> {step.destination_directory}"
                )
            else:
                lines.append(
                    f"{index:02d}. [{step.name}] remove "
                    + ", ".join(str(path) for path in step.paths)
                    + f" (cwd: {step.cwd})"
                )
        return "\n".join(lines)

    def run(
        self,
        *,
        dry_run: bool = False,
        environment: Mapping[str, str] | None = None,
        log_dir: str | Path | None = None,
    ) -> list[StepResult]:
        """Run all steps in order and return their individual results.

        ``dry_run`` validates and returns the declared sequence without touching
        the filesystem or invoking OpenFOAM.  Normal execution stops at the
        first failing command, preserving its captured log file for diagnosis.
        """
        self.validate()
        if dry_run:
            return [
                StepResult(step.name, "dry-run", self._describe(step)) for step in self._steps
            ]

        if not self.root.is_dir():
            raise FileNotFoundError(f"Workflow root does not exist: {self.root}")

        target_log_dir = Path(log_dir) if log_dir else self.root / "logs"
        if not target_log_dir.is_absolute():
            target_log_dir = self.root / target_log_dir
        target_log_dir.mkdir(parents=True, exist_ok=True)

        base_environment = os.environ.copy()
        if environment:
            base_environment.update({key: str(value) for key, value in environment.items()})

        results: list[StepResult] = []
        for index, step in enumerate(self._steps, start=1):
            if isinstance(step, CopyStep):
                self._copy(step)
                results.append(StepResult(step.name, "completed", self._describe(step)))
                continue
            if isinstance(step, RemoveStep):
                self._remove(step)
                results.append(StepResult(step.name, "completed", self._describe(step)))
                continue
            if isinstance(step, RestoreInitialFieldsStep):
                self._restore_initial_fields(step)
                results.append(StepResult(step.name, "completed", self._describe(step)))
                continue

            step_environment = base_environment.copy()
            step_environment.update({key: str(value) for key, value in step.environment.items()})
            cwd = self.root / step.cwd
            if not cwd.is_dir():
                raise FileNotFoundError(f"Working directory for step '{step.name}' does not exist: {cwd}")
            log_path = target_log_dir / f"{index:02d}_{step.name}.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                try:
                    subprocess.run(
                        list(step.command),
                        cwd=cwd,
                        env=step_environment,
                        text=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                except FileNotFoundError as error:
                    raise RuntimeError(
                        f"Executable '{step.command[0]}' is unavailable for step '{step.name}'. "
                        f"See {log_path} and load a compatible OpenFOAM environment."
                    ) from error
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"Step '{step.name}' failed with exit code {error.returncode}. "
                        f"See {log_path}."
                    ) from error
            results.append(StepResult(step.name, "completed", str(log_path)))
        return results

    def _copy(self, step: CopyStep) -> None:
        source = self.root / step.source
        destination = self.root / step.destination
        if not source.exists():
            if step.required:
                raise FileNotFoundError(f"Copy source for step '{step.name}' does not exist: {source}")
            return
        if source.is_dir():
            if destination.exists() and not destination.is_dir():
                raise FileExistsError(f"Cannot copy directory {source} onto file {destination}")
            shutil.copytree(source, destination, dirs_exist_ok=True)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_dir():
            destination = destination / source.name
        shutil.copy2(source, destination)

    def _restore_initial_fields(self, step: RestoreInitialFieldsStep) -> None:
        source_directory = self.root / step.source_directory
        destination_directory = self.root / step.destination_directory
        if source_directory.is_dir():
            shutil.copytree(source_directory, destination_directory, dirs_exist_ok=True)
            return

        destination_directory.mkdir(parents=True, exist_ok=True)
        original_fields = sorted(destination_directory.glob("*.orig"))
        if not original_fields:
            raise FileNotFoundError(
                f"Initial field source for step '{step.name}' is missing: "
                f"expected {source_directory} or .orig files in {destination_directory}"
            )
        for original_field in original_fields:
            field_name = original_field.name.removesuffix(".orig")
            shutil.copy2(original_field, destination_directory / field_name)

    def _remove(self, step: RemoveStep) -> None:
        base_path = self.root / step.cwd
        if not base_path.is_dir():
            raise FileNotFoundError(f"Cleanup directory for step '{step.name}' does not exist: {base_path}")
        for relative_path in step.paths:
            path = base_path / relative_path
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()

    @staticmethod
    def _describe(step: WorkflowStep) -> str:
        if isinstance(step, CommandStep):
            return f"cd {step.cwd} && {' '.join(step.command)}"
        if isinstance(step, CopyStep):
            return f"copy {step.source} -> {step.destination}"
        if isinstance(step, RestoreInitialFieldsStep):
            return f"restore {step.source_directory} -> {step.destination_directory}"
        return "remove " + ", ".join(str(path) for path in step.paths)
