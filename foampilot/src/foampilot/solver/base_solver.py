import gzip
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.constant.constantDirectory import ConstantDirectory
from foampilot.boundaries.boundaries_dict import Boundary
from foampilot.base.cases_variables import CaseFieldsManager
from foampilot.solver.marine_case import MarineCaseConfig
from foampilot.openfoam.execution.environment import OpenFOAMEnvironment
from foampilot.openfoam.execution.runner import OpenFOAMRunner
from foampilot.openfoam.solvers.registry import SolverRegistry
from foampilot.openfoam.solvers.configs import config_from_flags

logger = logging.getLogger(__name__)

class BaseSolver:
    """Base solver class with all common functionality."""

    def __init__(
        self,
        case_path: str | Path,
        solver_name: str,
        compressible: bool = False,
        with_gravity: bool = False,
        is_vof: bool = False,
        is_solid: bool = False,
        energy_activated: bool = False,
        transient: bool = False,
        turbulence_model: Optional[str] = None,
        with_moving_mesh: bool = False,
    ):
        self.case_path = Path(case_path)
        self.solver_name = solver_name
        self.foamrun_module = SolverRegistry.get_module(solver_name)

        # Flags
        self.compressible = compressible
        self.with_gravity = with_gravity
        self.is_vof = is_vof
        self.is_solid = is_solid
        self.energy_activated = energy_activated
        self.transient = transient
        self.turbulence_model = turbulence_model
        self.with_moving_mesh = with_moving_mesh
        self._sub_solver = None

        # --- Field manager ---
        self.fields_manager = CaseFieldsManager(
            is_solid=is_solid,
            with_gravity=with_gravity,
            is_vof=is_vof,
            energy_activated=energy_activated,
            turbulence_model=turbulence_model,
            with_moving_mesh=with_moving_mesh,
        )

        # --- Subcomponents ---
        self.system = SystemDirectory(self)
        self.constant = ConstantDirectory(self)
        self.boundary = Boundary(self, fields_manager=self.fields_manager, turbulence_model=turbulence_model)

        # --- Execution backend ---
        self._env = OpenFOAMEnvironment()
        self._runner = OpenFOAMRunner(case_path=self.case_path, env=self._env)

        self.config = config_from_flags(
            solver_name,
            compressible=compressible,
            with_gravity=with_gravity,
            is_vof=is_vof,
            is_solid=is_solid,
            energy_activated=energy_activated,
            transient=transient,
            turbulence_model=turbulence_model,
            with_moving_mesh=with_moving_mesh,
        )

    @property
    def simulation_type(self) -> str:
        """Return the simulation type string used by fvSchemes/fvSolution."""
        return self.config.get_simulation_type()

    @property
    def energy_variable(self) -> str:
        """Return the primary energy/temperature variable name."""
        return self.config.get_energy_variable()

    @property
    def sub_solver(self) -> Optional[str]:
        """Return the subSolver name for the ``functions`` solver module."""
        return self._sub_solver

    @sub_solver.setter
    def sub_solver(self, value: Optional[str]):
        self._sub_solver = value

    def update_case_specific_attributes(self):
        """Default: do nothing"""
        pass

    # ---------- Marine case validation ----------
    def validate_marine_case(self, strict: bool = True) -> MarineCaseConfig:
        """Validate the structural inputs of a Foundation 13 marine case.

        Validation is opt-in and never replaces OpenFOAM's ``checkMesh``.
        """
        config = MarineCaseConfig.from_case(self.case_path)
        if strict:
            config.validate_files()
        return config

    # ---------- Directory and setup ----------
    def ensure_dirs(self) -> None:
        (self.case_path / "system").mkdir(parents=True, exist_ok=True)
        (self.case_path / "constant").mkdir(parents=True, exist_ok=True)
        (self.case_path / "0").mkdir(parents=True, exist_ok=True)

    def setup_case(self) -> None:
        self.ensure_dirs()
        self.update_case_specific_attributes()

    # ---------- Case writing ----------
    def write_case(self) -> None:
        try:
            self.system.write()
        except Exception:
            pass
        try:
            self.constant.write()
        except Exception:
            pass

    # ---------- Reference assets ----------
    def import_reference_asset(self, source_path: str | Path, destination: str | Path) -> Path:
        """Copy a non-dictionary reference asset into the case.

        The destination is relative to the case unless an absolute path is
        provided. Executable assets retain their executable permission.
        """
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Reference asset not found: {source}")
        target = Path(destination)
        if not target.is_absolute():
            target = self.case_path / target
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".gz" and target.suffix != ".gz":
            with gzip.open(source, "rb") as source_stream, target.open("wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
        else:
            shutil.copy2(source, target)
        if source.stat().st_mode & 0o111:
            target.chmod(target.stat().st_mode | 0o111)
        return target

    def copy_case_tree(
        self,
        source_case: str | Path,
        source_relative: str | Path,
        destination_relative: str | Path,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Copy a file or directory between FoamPilot-managed case trees."""
        source = Path(source_case) / source_relative
        target = self.case_path / destination_relative
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if target.exists() and not overwrite:
                raise FileExistsError(target)
            shutil.copytree(source, target, dirs_exist_ok=overwrite)
        else:
            if target.exists() and not overwrite:
                raise FileExistsError(target)
            shutil.copy2(source, target)
        return target

    def write_text_asset(self, destination: str | Path, content: str) -> Path:
        """Write a FoamPilot-managed generated text asset into the case."""
        target = Path(destination)
        if not target.is_absolute():
            target = self.case_path / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def remove_case_asset(self, destination: str | Path) -> None:
        """Remove a FoamPilot-managed file or directory inside the case."""
        target = Path(destination)
        if not target.is_absolute():
            target = self.case_path / target
        try:
            target.relative_to(self.case_path)
        except ValueError as exc:
            raise ValueError("Case asset must be inside the FoamPilot case") from exc
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    def merge_mesh_points(self, points_tmp: str | Path, *, header_lines: int = 17) -> Path:
        """Merge a FoamPilot-managed temporary points list into mesh points.

        ``datToFoam`` can emit an auxiliary ``points.tmp`` list for meshes
        whose generated points file must retain its OpenFOAM header.  This
        helper preserves the requested header and appends the temporary list,
        without invoking shell text-processing commands.
        """
        points_path = self.case_path / "constant" / "polyMesh" / "points"
        tmp_path = Path(points_tmp)
        if not tmp_path.is_absolute():
            tmp_path = self.case_path / tmp_path
        if not points_path.is_file():
            raise FileNotFoundError(f"Mesh points file not found: {points_path}")
        if not tmp_path.is_file():
            raise FileNotFoundError(f"Temporary mesh points file not found: {tmp_path}")
        points_text = points_path.read_text(encoding="utf-8", errors="replace")
        tmp_text = tmp_path.read_text(encoding="utf-8", errors="replace")
        lines = points_text.splitlines(keepends=True)
        header = "".join(lines[:header_lines]).replace("format      binary;", "format      ascii;")
        points_path.write_text(header + tmp_text, encoding="utf-8")
        tmp_path.unlink()
        return points_path

    def update_mesh_patch_types(self, patch_types: Dict[str, str]) -> Path:
        """Update patch types in ``constant/polyMesh/boundary``.

        This is intended for mesh workflows where a generator creates generic
        ``patch`` entries and a solver-specific stage must convert selected
        entries (for example to ``wedge``).  The operation is performed by
        FoamPilot on the generated boundary file and preserves all unrelated
        patch content.
        """
        boundary_path = self.case_path / "constant" / "polyMesh" / "boundary"
        if not boundary_path.is_file():
            raise FileNotFoundError(f"Mesh boundary file not found: {boundary_path}")
        content = boundary_path.read_text(encoding="utf-8")
        for patch_name, patch_type in patch_types.items():
            pattern = (r"(" + re.escape(patch_name) + r"\s*\{\s*type\s+)\w+(\s*;)")
            content, count = re.subn(pattern, r"\g<1>" + patch_type + r"\g<2>", content, count=1)
            if count != 1:
                raise ValueError(f"Patch '{patch_name}' not found in {boundary_path}")
        boundary_path.write_text(content, encoding="utf-8")
        return boundary_path

    # ---------- Running simulation ----------
    def _command_environment(self) -> Dict[str, str]:
        return self._env.command_environment()

    def run_command(
        self,
        cmd: Sequence[str],
        log_filename: str,
        cwd: str | Path | None = None,
        env: Dict[str, str] | None = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        return self._runner.run_command(cmd, log_filename, cwd=cwd, env=env, environment=environment)

    def run_external(
        self,
        cmd: Sequence[str],
        log_filename: str,
        cwd: str | Path | None = None,
        env: Dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        return self._runner.run_external(cmd, log_filename, cwd=cwd, env=env)

    def run_command_async(self, cmd: Sequence[str], log_filename: str):
        return self._runner.run_command_async(cmd, log_filename)

    def wait_command(self, process, check: bool = True) -> int:
        return self._runner.wait_command(process, check=check)

    @staticmethod
    def openfoam_version() -> str | None:
        return OpenFOAMEnvironment.openfoam_version()

    def require_openfoam(self, major: str | int | None = None) -> str:
        return self._env.require_openfoam(major=major)

    def validate_results(self, log_filename: str | None = None) -> Path:
        return self._runner.validate_results(self.solver_name, log_filename=log_filename)

    def check_solver_module_exists(self) -> bool:
        return self._runner.check_solver_module_exists(self.foamrun_module)

    def run_simulation(self, nb_proc: int = 1, log_filename: str | None = None):
        self._runner.run_simulation(self.solver_name, self.foamrun_module, nb_proc=nb_proc, log_filename=log_filename)

    def _run_legacy_solver(self, nb_proc: int, log_filename: str | None = None) -> None:
        self._runner._run_legacy_solver(self.solver_name, nb_proc, log_filename)

    def run_parallel(self, nb_proc: int, log_filename: str | None = None, force_decompose: bool = False):
        self._runner.run_parallel(self.foamrun_module, nb_proc, log_filename=log_filename, force_decompose=force_decompose)
