from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from typing import List

from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.constant.constantDirectory import ConstantDirectory
from foampilot.boundaries.boundaries_dict import Boundary


class BaseSolver(ABC):
    """
    Base class for solvers. Provides common case management and helpers.
    Subclasses must implement update_case_specific_attributes().
    """

    def __init__(self, case_path: str | Path, solver_name: str):
        self.case_path = Path(case_path)
        self.solver_name = solver_name

        # Initialize subcomponents
        self.system = SystemDirectory(self)
        self.constant = ConstantDirectory(self)
        self.boundary = Boundary(self)

        # Generic flags (can be overridden by subclasses)
        self.compressible = False
        self.with_gravity = True

    def ensure_dirs(self) -> None:
        (self.case_path / "system").mkdir(parents=True, exist_ok=True)
        (self.case_path / "constant").mkdir(parents=True, exist_ok=True)
        (self.case_path / "0").mkdir(parents=True, exist_ok=True)

    def setup_case(self) -> None:
        """
        Prepare the case directory and apply solver-specific attribute updates.
        """
        self.ensure_dirs()
        self.update_case_specific_attributes()

    @abstractmethod
    def update_case_specific_attributes(self) -> None:
        """
        Implement in subclass to set solver-specific flags/parameters.
        """
        raise NotImplementedError

    def write_case(self) -> None:
        """
        Write case files (delegates to components).
        """
        # SystemDirectory.write and ConstantDirectory.write exist in repo
        try:
            self.system.write()
        except Exception:
            # keep behavior tolerant if a subsystem is not fully configured
            pass

        try:
            self.constant.write()
        except Exception:
            pass

        # Boundary writing is handled by boundary helper methods (no generic write assumed)

    def run_command(self, cmd: List[str], log_filename: str) -> None:
        """
        Run a command in the case directory and write stdout+stderr to log file.
        Raises CalledProcessError on non-zero exit.
        """
        if not self.case_path.exists() or not self.case_path.is_dir():
            raise FileNotFoundError(f"Case path {self.case_path} does not exist or is not a directory.")

        log_path = self.case_path / log_filename
        cmd_display = " ".join(cmd)
        print(f"Running command in {self.case_path}: {cmd_display} -> log: {log_path}")

        with open(log_path, "w", encoding="utf-8") as log_file:
            subprocess.run(
                cmd,
                cwd=self.case_path,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def run_simulation(self, log_filename: str | None = None) -> None:
        """
        Default run_simulation uses the solver_name as the command.
        Subclasses can override if they need a different command.
        """
        if log_filename is None:
            log_filename = f"log.{self.solver_name}"

        try:
            self.run_command([self.solver_name], log_filename)
            print(f"✅ Simulation {self.solver_name} finished successfully. Log: {self.case_path / log_filename}")
        except subprocess.CalledProcessError:
            print(f"❌ Simulation {self.solver_name} failed. See log: {self.case_path / log_filename}")
            raise RuntimeError(f"Simulation failed (see {self.case_path / log_filename})")