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

# Core dictionaries module (Internal Use Only - delegates to this in the future)
from foampilot.core.dictionaries import FoamDict, DictionaryWriter, CaseLayout


class ConstantDirectory:
    """Manages the constant directory for an OpenFOAM case.

    Internal Use Only - This class delegates to core/dictionaries module.
    The business logic (VoF configuration, radiation management, turbulence
    model selection) is preserved, while actual file writing is coordinated
    through FoamDict/DictionaryWriter.
    """

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

        # Core dictionaries module (Internal Use Only)
        self._dictionary_writer: Optional[DictionaryWriter] = None
        self._case_layout: Optional[CaseLayout] = None

        if with_radiation:
            self.enable_radiation()

    def _get_dictionary_writer(self) -> DictionaryWriter:
        """Get or create the DictionaryWriter for this case."""
        if self._dictionary_writer is None:
            constant_path = Path(self.solver.case_path) / "constant"
            self._dictionary_writer = DictionaryWriter(constant_path)
        return self._dictionary_writer

    def _get_case_layout(self) -> CaseLayout:
        """Get or create the CaseLayout for this case."""
        if self._case_layout is None:
            self._case_layout = CaseLayout(self.solver.case_path)
        return self._case_layout

    @property
    def writer(self) -> DictionaryWriter:
        """Access the DictionaryWriter for fluent dictionary operations."""
        return self._get_dictionary_writer()

    @property
    def layout(self) -> CaseLayout:
        """Access the CaseLayout for directory management."""
        return self._get_case_layout()

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

    def _create_foam_dict(self, file_instance, name: str) -> FoamDict:
        """Create a FoamDict from a file class instance.

        Args:
            file_instance: OpenFOAMFile subclass instance
            name: Name for the FoamDict (used as filename)

        Returns:
            FoamDict with same header and data as the file instance
        """
        foam_dict = FoamDict(object_name=file_instance.object_name)
        foam_dict._header = file_instance.header.copy()
        foam_dict._data = file_instance.attributes.copy()
        return foam_dict

    def write(self):
        constant_path = Path(self.solver.case_path) / "constant"
        self.layout.ensure()

        writer = self._get_dictionary_writer()
        writer.clear()

        simulationType, model = self.solver.get_turbulence_configuration()
        config = self.solver.config

        if config.is_compressible or config.use_solver_keyword:
            mt_file = MomentumTransportFile(
                parent=self.solver,
                simulationType=simulationType,
                RASModel=model if simulationType == "RAS" else None,
                LESModel=model if simulationType == "LES" else None,
            )
            writer.register("momentumTransport", self._create_foam_dict(mt_file, "momentumTransport"))
        else:
            turbulence = TurbulencePropertiesFile(
                parent=self.solver,
                simulationType=simulationType,
                RASModel=model if simulationType == "RAS" else None,
                LESModel=model if simulationType == "LES" else None,
            )
            writer.register("turbulenceProperties", self._create_foam_dict(turbulence, "turbulenceProperties"))

        # Transport / Physical
        if config.is_compressible or config.use_solver_keyword:
            writer.register("physicalProperties", self._create_foam_dict(self._physicalProperties, "physicalProperties"))
            if config.writes_pRef:
                writer.register("pRef", self._create_foam_dict(self._pRef, "pRef"))
        else:
            writer.register("transportProperties", self._create_foam_dict(self._transportProperties, "transportProperties"))
            if config.writes_pRef:
                writer.register("pRef", self._create_foam_dict(self._pRef, "pRef"))

        if config.requires_gravity:
            writer.register("g", self._create_foam_dict(self._gravity, "g"))
            if "p" in self.solver.fields_manager.fields and "p_rgh" not in self.solver.fields_manager.fields:
                self.solver.fields_manager.fields["p_rgh"] = self.solver.fields_manager.fields.pop("p")

        if self.with_radiation:
            if self._radiation is None:
                self.enable_radiation()
            writer.register("radiationProperties", self._create_foam_dict(self._radiation, "radiationProperties"))
            writer.register("fvModels", self._create_foam_dict(self._fvmodels, "fvModels"))

        if config.is_vof and self._vof_phases is not None:
            phase_props = PhasePropertiesFile(
                parent=self.solver,
                phases=self._vof_phases,
                sigma=self._vof_sigma,
            )
            writer.register("phaseProperties", self._create_foam_dict(phase_props, "phaseProperties"))

            for phase in self._vof_phases:
                props = self._vof_phase_properties.get(phase, {})
                nu = props.get("nu", 1e-6)
                rho = props.get("rho", 1000)
                pp_file = PhasePhysicalPropertiesFile(
                    parent=self.solver, phase=phase, nu=nu, rho=rho
                )
                writer.register(f"physicalProperties.{phase}", self._create_foam_dict(pp_file, f"physicalProperties.{phase}"))

            mt_file = MomentumTransportFile(
                parent=self.solver,
                simulationType=simulationType,
            )
            writer.register("momentumTransport", self._create_foam_dict(mt_file, "momentumTransport"))

            for fname in ("transportProperties", "turbulenceProperties", "pRef"):
                writer.unregister(fname)

        writer.write_all()

        if config.is_vof and self._vof_phases is not None:
            for fname in ("transportProperties", "turbulenceProperties", "pRef"):
                fpath = constant_path / fname
                if fpath.exists():
                    fpath.unlink()

        logger.info(f"Constant directory written to {constant_path}")
        return self