from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from foampilot.solver import Solver

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import fichiers constants
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.constant.physicalProperties import PhysicalPropertiesFile
from foampilot.constant.gravityFile import GravityFile
from foampilot.constant.pRefFile import PRefFile
from foampilot.constant.radiationProperties import RadiationPropertiesFile, FvModelsFile
from foampilot.constant.phasePropertiesFile import PhasePropertiesFile
from foampilot.constant.momentumTransportFile import MomentumTransportFile
from foampilot.constant.phasePhysicalPropertiesFile import PhasePhysicalPropertiesFile

class ConstantDirectory:
    def __init__(self, solver: Solver, *, with_radiation: bool = False):
        """
        Initialize constant directory manager.

        Args:
            solver: Base solver instance
            with_radiation: Enable radiation by default
        """
        self.solver = solver
        self.with_radiation = with_radiation

        # VoF-specific attributes
        self._vof_phases: Optional[list[str]] = None
        self._vof_sigma: float = 0.0728
        self._vof_phase_properties: dict = {}

        # Initialisation des fichiers constants
        self._transportProperties = TransportPropertiesFile(self.solver)
        self._physicalProperties = PhysicalPropertiesFile(self.solver)
        self._gravity = GravityFile(self.solver)
        self._pRef = PRefFile()

        # Radiation files
        self._radiation: Optional[RadiationPropertiesFile] = None
        self._fvmodels: Optional[FvModelsFile] = None
        self._turbulenceProperties: Optional[TurbulencePropertiesFile] = None

        if with_radiation:
            self.enable_radiation()

    # Properties
    @property
    def transportProperties(self):
        return self._transportProperties

    @property
    def turbulenceProperties(self):
        return self._turbulenceProperties

    @property
    def physicalProperties(self):
        return self._physicalProperties

    @property
    def gravity(self):
        return self._gravity

    @property
    def pRef(self):
        return self._pRef

    @property
    def radiation(self):
        return self._radiation

    # VoF configuration
    def configure_vof(self, phases=None, sigma: float = 0.0728, phase_properties: dict | None = None):
        """Configure this ConstantDirectory for a VoF (two-phase) case.

        After calling this, ``write()`` will emit ``phaseProperties``,
        ``physicalProperties.<phase>`` and ``momentumTransport`` instead of
        the single-phase ``transportProperties`` / ``turbulenceProperties`` /
        ``pRef`` files.

        Args:
            phases: Ordered list of phase names, e.g. ``["water", "air"]``.
            sigma: Surface tension coefficient (N/m).
            phase_properties: Dict mapping phase name → {"nu": ..., "rho": ...}.
        """
        self._vof_phases = list(phases) if phases else ["water", "air"]
        self._vof_sigma = float(sigma) if isinstance(sigma, (int, float)) else sigma
        self._vof_phase_properties = phase_properties or {}

    def _write_vof_constants(self, constant_path: Path):
        """Write VoF-specific constant files and remove conflicting single-phase files."""
        is_vof = getattr(self.solver, "is_vof", False) and self._vof_phases is not None

        if not is_vof:
            return

        # --- phaseProperties ---
        phase_props = PhasePropertiesFile(
            parent=self.solver,
            phases=self._vof_phases,
            sigma=self._vof_sigma,
        )
        phase_props.write(constant_path / "phaseProperties")

        # --- physicalProperties.<phase> ---
        for phase in self._vof_phases:
            props = self._vof_phase_properties.get(phase, {})
            nu = props.get("nu", 1e-6)
            rho = props.get("rho", 1000)
            pp_file = PhasePhysicalPropertiesFile(
                parent=self.solver,
                phase=phase,
                nu=nu,
                rho=rho,
                thermo_type=props.get("thermoType"),
                mixture=props.get("mixture"),
            )
            pp_file.write(constant_path / f"physicalProperties.{phase}")

        # --- momentumTransport ---
        simulation_type, _ = self.solver.get_turbulence_configuration()
        mt_file = MomentumTransportFile(
            parent=self.solver,
            simulationType=simulation_type,
        )
        mt_file.write(constant_path / "momentumTransport")

        # --- Remove files that conflict with the two-phase transport model ---
        conflicting_files = ["transportProperties", "turbulenceProperties", "physicalProperties"]
        if not getattr(self.solver, "compressible", False):
            conflicting_files.append("pRef")
        for fname in conflicting_files:
            fpath = constant_path / fname
            if fpath.exists():
                fpath.unlink()

    def import_reference_file(self, source_path: str | Path, filename: str | None = None) -> Path:
        """Import a complete OpenFOAM constant dictionary without lossy parsing."""
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = Path(self.solver.case_path) / "constant" / (filename or source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return target

    def remove_files(self, filenames: list[str]) -> list[Path]:
        """Remove named files from ``constant`` and return removed paths.

        This is useful after importing a complete reference case when
        ``setup_case`` has emitted legacy/default dictionaries that must not
        coexist with the imported OpenFOAM dictionaries.
        """
        constant_path = Path(self.solver.case_path) / "constant"
        removed: list[Path] = []
        for filename in filenames:
            path = constant_path / filename
            if path.is_file():
                path.unlink()
                removed.append(path)
        return removed

    # Radiation management
    def enable_radiation(self, model: str = "P1", **kwargs):
        self.with_radiation = True
        self._radiation = RadiationPropertiesFile(self.solver, model=model, **kwargs)
        self._fvmodels = FvModelsFile(self.solver)
        logger.info(f"Radiation enabled with model: {model}")

    def disable_radiation(self):
        self.with_radiation = False
        self._radiation = None
        self._fvmodels = None
        logger.info("Radiation disabled")

    # Write files
    def write(self):
        constant_path = Path(self.solver.case_path) / "constant"
        constant_path.mkdir(parents=True, exist_ok=True)

        # --- Turbulence properties -----------------------------------------
        # OpenFOAM 13 renamed turbulenceProperties → momentumTransport
        # for compressible solvers (fluid, buoyantSimpleFoam, etc.)
        simulationType, model = self.solver.get_turbulence_configuration()
        is_compressible = getattr(self.solver, "compressible", False)

        use_modular_fluid = getattr(self.solver, "solver_name", "") == "fluid"
        if is_compressible or use_modular_fluid:
            mt_file = MomentumTransportFile(
                parent=self.solver,
                simulationType=simulationType,
                RASModel=model if simulationType == "RAS" else None,
                LESModel=model if simulationType == "LES" else None,
            )
            mt_file.write(constant_path / "momentumTransport")
        else:
            turbulence = TurbulencePropertiesFile(
                parent=self.solver,
                simulationType=simulationType,
                RASModel=model if simulationType == "RAS" else None,
                LESModel=model if simulationType == "LES" else None,
            )
            turbulence.write(constant_path / "turbulenceProperties")


        # Transport / Physical
        if getattr(self.solver, "compressible", False) or use_modular_fluid:
            self._physicalProperties.write(constant_path / "physicalProperties")
            self._pRef.write(constant_path / "pRef")
        else:
            self._transportProperties.write(constant_path / "transportProperties")
            self._pRef.write(constant_path / "pRef")

        # Gravity
        if getattr(self.solver, "with_gravity", False):
            self._gravity.write()
            # Update p → p_rgh if necessary
            if "p" in self.solver.fields_manager.fields and "p_rgh" not in self.solver.fields_manager.fields:
                self.solver.fields_manager.fields["p_rgh"] = self.solver.fields_manager.fields.pop("p")

        # Radiation
        if self.with_radiation:
            if self._radiation is None:
                self.enable_radiation()
            self._radiation.write(constant_path )
            self._fvmodels.write(constant_path )

        # VoF-specific constant files (overwrites single-phase files + cleanup)
        self._write_vof_constants(constant_path)

        logger.info(f"Constant directory written to {constant_path}")
        return self