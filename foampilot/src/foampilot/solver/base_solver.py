from pathlib import Path
import logging
import subprocess
from typing import List, Optional

from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.constant.constantDirectory import ConstantDirectory
from foampilot.boundaries.boundaries_dict import Boundary
from foampilot.base.cases_variables import CaseFieldsManager
import os

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
        with_radiation: bool = False,
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
        self.with_radiation = with_radiation
        self._sub_solver = None

        # --- Field manager ---
        self.fields_manager = CaseFieldsManager(
            is_solid=is_solid,
            with_gravity=with_gravity,
            is_vof=is_vof,
            energy_activated=energy_activated,
            turbulence_model=turbulence_model,
            with_radiation=with_radiation,
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
        """Return the primary energy/temperature variable name.

        ``T`` for incompressible and Boussinesq flows; ``h`` for compressible.
        """
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

    # ---------- Running simulation ----------
    def run_command(self, cmd: List[str], log_filename: str) -> None:
        log_path = self.case_path / log_filename
        logger.info("Running command: %s -> log: %s", " ".join(cmd), log_path)
        with open(log_path, "w", encoding="utf-8") as log_file:
            subprocess.run(
                cmd,
                cwd=self.case_path,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def check_solver_module_exists(self) -> bool:
        foam_modules = os.getenv("FOAM_MODULES", "")
        if not foam_modules:
            logger.warning("$FOAM_MODULES environment variable is not set.")
            return False
        module_path = Path(foam_modules) / self.foamrun_module
        if not module_path.exists():
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

        # --- parallel execution ---
        if nb_proc >= 2:
            return self.run_parallel(nb_proc)

        # --- serial execution ---
        if log_filename is None:
            log_filename = f"log.{self.solver_name}"

        if not self.check_solver_module_exists():
            raise RuntimeError(
                f"Solver module '{self.foamrun_module}' is not available."
            )

        logger.info("Running simulation in serial mode (1 proc)")

        self.run_command(["foamRun", "-solver", self.foamrun_module], log_filename)

    def run_parallel(self, nb_proc: int, log_filename: str | None = None):

        if log_filename is None:
            log_filename = f"log.{self.solver_name}"
        log_path = self.case_path / log_filename

        logger.info("Parallel run with %d processors", nb_proc)

        # Ask the system directory to prepare decomposeParDict
        if hasattr(self.system, "ensure_decomposeParDict"):
            self.system.ensure_decomposeParDict(nb_proc)
            self.system.write()

        # 1. decomposePar
        logger.info("Running decomposePar ...")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=== decomposePar ===\n")
            subprocess.run(
                ["decomposePar", "-case", str(self.case_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True
            )

        # 2. mpirun with foamRun
        logger.info("Running mpirun simulation ...")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== mpirun foamRun ===\n")
            subprocess.run(
                ["mpirun", "-np", str(nb_proc), "foamRun", "-solver", self.foamrun_module, "-parallel"],
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