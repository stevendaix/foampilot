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

logger = logging.getLogger(__name__)

class BaseSolver:
    """Base solver class with all common functionality."""

    SOLVER_MODULES = {
        # Single-phase modules
        "fluid": "fluid",
        "incompressibleFluid": "incompressibleFluid",
        "multicomponentFluid": "multicomponentFluid",
        # Multiphase/VoF flow modules
        "compressibleVoF": "compressibleVoF",
        "incompressibleVoF": "incompressibleVoF",
        # Solid modules
        "solidDisplacement": "solidDisplacement",
        # Utility modules
        "functions": "functions",
        "movingMesh": "movingMesh",
        # OpenFOAM-14 solvers
        "icoFoam": "icoFoam",
        "simpleFoam": "simpleFoam",
        "pimpleFoam": "pimpleFoam",
        "pimpleDyMFoam": "pimpleDyMFoam",
        "rhoCentralFoam": "rhoCentralFoam",
        "sonicFoam": "sonicFoam",
        "reactingFoam": "reactingFoam",
        "scalarTransportFoam": "scalarTransportFoam",
        "chtMultiRegionFoam": "chtMultiRegionFoam",
        "chtMultiRegionSimpleFoam": "chtMultiRegionSimpleFoam",
        "compressibleSinglePhasePorosityFoam": "compressibleSinglePhasePorosityFoam",
        "porousSimpleFoam": "porousSimpleFoam",
        # Legacy OpenCFD solvers used in marine/overset studies
        "overInterDyMFoam": "overInterDyMFoam",
        "rhoSimpleFoam": "rhoSimpleFoam",
    }

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
        self.foamrun_module = self.SOLVER_MODULES.get(solver_name, solver_name)

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

    @property
    def simulation_type(self) -> str:
        """Return the simulation type string used by fvSchemes/fvSolution."""
        if self.is_solid:
            return "solid"
        if self.compressible:
            return "compressible"
        if self.is_vof:
            return "vof"
        return "incompressible"

    @property
    def energy_variable(self) -> str:
        """Return the primary energy/temperature variable name."""
        if self.compressible and not self.is_vof:
            return "h"
        return "T"

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
    def run_command(
        self,
        cmd: Sequence[str],
        log_filename: str,
        cwd: str | Path | None = None,
        env: Dict[str, str] | None = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command and persist its combined log.

        ``cwd`` defaults to the case directory. Both ``env`` and the legacy
        ``environment`` keyword are merged over the parent environment.
        """
        workdir = Path(cwd) if cwd is not None else self.case_path
        log_path = self.case_path / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Running command: %s -> log: %s", " ".join(cmd), log_path)
        process_environment = os.environ.copy()
        if env:
            process_environment.update(env)
        if environment:
            process_environment.update(environment)
        with log_path.open("w", encoding="utf-8") as log_file:
            return subprocess.run(
                list(cmd), cwd=workdir, env=process_environment, text=True,
                stdout=log_file, stderr=subprocess.STDOUT, check=True,
            )

    def run_external(
        self,
        cmd: Sequence[str],
        log_filename: str,
        cwd: str | Path | None = None,
        env: Dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run an external build or preprocessing command with FoamPilot logging."""
        return self.run_command(cmd, log_filename, cwd=cwd, env=env)

    def run_command_async(self, cmd: Sequence[str], log_filename: str):
        """Start a FoamPilot-managed command and return its process handle."""
        log_path = self.case_path / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Starting async command: %s -> log: %s", " ".join(cmd), log_path)
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            list(cmd), cwd=self.case_path, text=True,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        process._foampilot_log_file = log_file
        return process

    def wait_command(self, process, check: bool = True) -> int:
        """Wait for a process returned by :meth:`run_command_async`."""
        returncode = process.wait()
        log_file = getattr(process, "_foampilot_log_file", None)
        if log_file is not None:
            log_file.close()
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, process.args)
        return returncode

    @staticmethod
    def openfoam_version() -> str | None:
        """Return the sourced OpenFOAM major version, if discoverable."""
        version = os.environ.get("WM_PROJECT_VERSION")
        if version:
            return version
        foam_version = shutil.which("foamVersion")
        if not foam_version:
            return None
        result = subprocess.run([foam_version], capture_output=True, text=True, check=False)
        output = (result.stdout or result.stderr).strip()
        return output.removeprefix("OpenFOAM-") or None

    def require_openfoam(self, major: str | int | None = None) -> str:
        """Require a sourced OpenFOAM environment and optionally a major version."""
        version = self.openfoam_version()
        if not version:
            raise RuntimeError("OpenFOAM is not sourced or foamVersion is unavailable")
        if major is not None and str(version) != str(major):
            raise RuntimeError(f"OpenFOAM {major} is required, found {version}")
        return version

    def validate_results(self, log_filename: str | None = None) -> Path:
        """Validate that a solver log ended and produced a numeric time directory."""
        log_path = self.case_path / (log_filename or f"log.{self.solver_name}")
        if not log_path.is_file():
            raise RuntimeError(f"Solver log was not produced: {log_path}")
        if "End" not in log_path.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"Solver did not finish successfully: {log_path}")
        times = []
        for path in self.case_path.iterdir():
            if not path.is_dir():
                continue
            try:
                float(path.name)
            except ValueError:
                continue
            times.append(path)
        if not times:
            raise RuntimeError("The solver produced no numeric time directory")
        return max(times, key=lambda p: float(p.name))

    def check_solver_module_exists(self) -> bool:
        foam_modules = os.getenv("FOAM_MODULES", "")
        if not foam_modules:
            # OpenFOAM 13 commonly exposes modules through foamRun rather than
            # a standalone path; use the executable as the fallback check.
            return shutil.which("foamRun") is not None
        module_dir = Path(foam_modules)
        module_candidates = (
            module_dir / self.foamrun_module,
            module_dir / f"lib{self.foamrun_module}.so",
            module_dir / f"lib{self.foamrun_module}.dylib",
        )
        if not any(candidate.exists() for candidate in module_candidates):
            logger.warning("Solver module '%s' not found in %s", self.foamrun_module, foam_modules)
            return False
        return True

    def get_turbulence_configuration(self):
        """
        Normalize turbulence configuration.

        Returns
        -------
        simulationType : str
            'laminar', 'RAS', or 'LES'
        model : Optional[str]
            Turbulence model name or None
        """
        # --- DEFAULT / LAMINAR ------------------------------------------
        if self.turbulence_model is None:
            return "laminar", None

        if isinstance(self.turbulence_model, str):
            model = self.turbulence_model.strip()

            if model.lower() == "laminar":
                return "laminar", None

            # --- LES ------------------------------------------------------
            if model.lower().startswith("les:"):
                return "LES", model.split(":", 1)[1]

            # --- RAS ------------------------------------------------------
            return "RAS", model

        raise ValueError(f"Invalid turbulence_model: {self.turbulence_model}")

    def run_simulation(self, nb_proc: int = 1, log_filename: str | None = None):
        # --- Legacy OpenCFD solvers run directly without foamRun ---
        legacy_solvers = {"overInterDyMFoam", "rhoSimpleFoam", "simpleFoam", "pimpleFoam", "marineFoam"}
        if self.solver_name in legacy_solvers:
            self._run_legacy_solver(nb_proc, log_filename)
            return

        # --- parallel execution ---
        if nb_proc >= 2:
            return self.run_parallel(nb_proc, log_filename)

        # --- serial execution ---
        if log_filename is None:
            log_filename = f"log.{self.solver_name}"

        if not self.check_solver_module_exists():
            raise RuntimeError(
                f"Solver module '{self.foamrun_module}' is not available."
            )

        logger.info("Running simulation in serial mode (1 proc)")

        self.run_command(["foamRun", "-solver", self.foamrun_module], log_filename)

    def _run_legacy_solver(self, nb_proc: int, log_filename: str | None = None) -> None:
        if log_filename is None:
            log_filename = f"log.{self.solver_name}"

        if nb_proc >= 2:
            logger.info("Parallel legacy solver run with %d processors", nb_proc)
            with open(self.case_path / log_filename, "w", encoding="utf-8") as log_file:
                log_file.write("=== decomposePar ===\n")
                subprocess.run(
                    ["decomposePar", "-case", str(self.case_path)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                log_file.write(f"\n=== mpirun {self.solver_name} ===\n")
                subprocess.run(
                    ["mpirun", "--oversubscribe", "-np", str(nb_proc), self.solver_name, "-parallel"],
                    cwd=self.case_path,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                log_file.write("\n=== reconstructPar ===\n")
                subprocess.run(
                    ["reconstructPar", "-case", str(self.case_path)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:
            logger.info("Serial legacy solver run")
            self.run_command([self.solver_name], log_filename)

    def run_parallel(self, nb_proc: int, log_filename: str | None = None, force_decompose: bool = False):

        if log_filename is None:
            log_filename = f"log.{self.solver_name}"
        log_path = self.case_path / log_filename

        logger.info("Parallel run with %d processors", nb_proc)

        # Ask the system directory to prepare decomposeParDict
        if hasattr(self.system, "ensure_decomposeParDict"):
            self.system.ensure_decomposeParDict(nb_proc)
            # Only write decomposeParDict; do NOT overwrite existing system files
            system_path = Path(self.case_path) / "system"
            system_path.mkdir(parents=True, exist_ok=True)
            if self.system.decomposeParDict is not None:
                self.system.decomposeParDict.write(system_path / "decomposeParDict")

        # 1. decomposePar
        logger.info("Running decomposePar ...")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=== decomposePar ===\n")
            subprocess.run(
                ["decomposePar", "-force", "-case", str(self.case_path)] if force_decompose else ["decomposePar", "-case", str(self.case_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True
            )

        # 2. mpirun with foamRun
        logger.info("Running mpirun simulation ...")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== mpirun foamRun ===\n")
            mpi_command = ["mpirun"]
            if os.getenv("FOAMPILOT_MPI_OVERSUBSCRIBE", "1").lower() not in {"0", "false", "no"}:
                mpi_command.append("--oversubscribe")
            mpi_command += ["-np", str(nb_proc), "foamRun", "-solver", self.foamrun_module, "-parallel"]
            subprocess.run(
                mpi_command,
                cwd=self.case_path,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True
            )

        # 3. reconstructPar
        logger.info("Running reconstructPar ...")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== reconstructPar ===\n")
            subprocess.run(
                ["reconstructPar", "-case", str(self.case_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True
            )

        logger.info("Parallel simulation finished ! (log: %s)", log_path)
