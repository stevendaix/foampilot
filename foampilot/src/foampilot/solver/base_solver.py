from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import os
from typing import List, Dict, ClassVar, Type

from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.constant.constantDirectory import ConstantDirectory
from foampilot.boundaries.boundaries_dict import Boundary
from foampilot.base.cases_variables import CaseFieldsManager


class BaseSolver(ABC):
    SOLVER_MODULES: ClassVar[Dict[str, str]] = {
        # Single-phase modules
        "fluid": "fluid",
        "incompressibleDenseParticleFluid": "incompressibleDenseParticleFluid",
        "incompressibleFluid": "incompressibleFluid",
        "multicomponentFluid": "multicomponentFluid",
        "shockFluid": "shockFluid",
        "XiFluid": "XiFluid",
        # Multiphase/VoF flow modules
        "compressibleMultiphaseVoF": "compressibleMultiphaseVoF",
        "compressibleVoF": "compressibleVoF",
        "incompressibleDriftFlux": "incompressibleDriftFlux",
        "incompressibleMultiphaseVoF": "incompressibleMultiphaseVoF",
        "incompressibleVoF": "incompressibleVoF",
        "isothermalFluid": "isothermalFluid",
        "multiphaseEuler": "multiphaseEuler",
        # Solid modules
        "solid": "solid",
        "solidDisplacement": "solidDisplacement",
        # Film modules
        "isothermalFilm": "isothermalFilm",
        "film": "film",
        # Utility modules
        "functions": "functions",
        "movingMesh": "movingMesh",
    }

    SOLVER_CLASSES: ClassVar[Dict[str, Type[BaseSolver]]] = {}

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

        # --- Gestion dynamique des champs ---
        self.fields_manager = CaseFieldsManager(
            is_solid=is_solid,
            with_gravity=with_gravity,
            is_vof=is_vof,
            energy_activated=energy_activated,
            turbulence_model=turbulence_model,
            with_radiation=with_radiation,
        )

        # Initialize subcomponents avec accès aux champs
        self.system = SystemDirectory(self)
        self.constant = ConstantDirectory(self)
        self.boundary = Boundary(self, fields_manager=self.fields_manager)  # Passer le gestionnaire de champs

        # Generic flags (gardés pour compatibilité)
        self.compressible = compressible
        self.with_gravity = with_gravity
        self.is_vof = is_vof
        self.is_solid = is_solid
        self.energy_activated = energy_activated
        self.transient = transient
        self.turbulence_model = turbulence_model
        self.with_radiation = with_radiation

    @classmethod
    def create(
        cls,
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
    ) -> BaseSolver:
        solver_class: Type[BaseSolver] = cls.SOLVER_CLASSES.get(solver_name, cls)
        return solver_class(
            case_path,
            solver_name,
            compressible=compressible,
            with_gravity=with_gravity,
            is_vof=is_vof,
            is_solid=is_solid,
            energy_activated=energy_activated,
            transient=transient,
            turbulence_model=turbulence_model,
            with_radiation=with_radiation,
        )


    # ---------- Directory and setup ----------
    def ensure_dirs(self) -> None:
        (self.case_path / "system").mkdir(parents=True, exist_ok=True)
        (self.case_path / "constant").mkdir(parents=True, exist_ok=True)
        (self.case_path / "0").mkdir(parents=True, exist_ok=True)

    def setup_case(self) -> None:
        self.ensure_dirs()
        self.update_case_specific_attributes()

    @abstractmethod
    def update_case_specific_attributes(self) -> None:
        """
        Implement in subclass to set solver-specific flags/parameters.
        """
        raise NotImplementedError

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

    def check_solver_module_exists(self) -> bool:
        foam_modules = os.getenv("FOAM_MODULES", "")
        if not foam_modules:
            print("⚠️  $FOAM_MODULES environment variable is not set.")
            return False

        module_path = Path(foam_modules) / self.foamrun_module
        if not module_path.exists():
            print(f"⚠️  Solver module '{self.foamrun_module}' not found in {foam_modules}")
            return False

        return True

    def run_simulation(self, log_filename: str | None = None) -> None:
        if log_filename is None:
            log_filename = f"log.{self.solver_name}"

        if not self.check_solver_module_exists():
            raise RuntimeError(f"Solver module '{self.foamrun_module}' is not available.")

        try:
            self.run_command(["foamRun", "-solver", self.foamrun_module], log_filename)
            print(f"✅ Simulation {self.solver_name} finished successfully. Log: {self.case_path / log_filename}")
        except subprocess.CalledProcessError:
            print(f"❌ Simulation {self.solver_name} failed. See log: {self.case_path / log_filename}")
            raise RuntimeError(f"Simulation failed (see {self.case_path / log_filename})")




