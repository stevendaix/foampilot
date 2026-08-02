import logging

logger = logging.getLogger(__name__)


from pathlib import Path
from typing import Dict, List, Optional, Any

from foampilot.solver.base_solver import BaseSolver
from foampilot.cht.regions import FluidRegion, SolidRegion
from foampilot.cht.interfaces import CoupledInterface


class ChtSolver(BaseSolver):
    """Solver for conjugate heat transfer (CHT) simulations.

    Supports multi-region setups with fluid and solid domains
    coupled through interfacial heat transfer.

    Parameters
    ----------
    case_path : str or Path
        Path to the OpenFOAM case directory.
    solver_name : str
        CHT solver name (``chtMultiRegionFoam`` or
        ``chtMultiRegionSimpleFoam``).
    regions : list of FluidRegion or SolidRegion
        The fluid and solid regions that participate in the CHT
        simulation.
    interfaces : list of CoupledInterface, optional
        The interfaces between fluid and solid regions.
    region_solvers : dict, optional
        Mapping from region name to solver type, e.g.
        ``{"fluid": "fluid", "solid": "solid"}``.
    """

    def __init__(
        self,
        case_path: str | Path,
        solver_name: str = "chtMultiRegionFoam",
        regions: Optional[List[Any]] = None,
        interfaces: Optional[List[CoupledInterface]] = None,
        region_solvers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        if solver_name not in self.SOLVER_MODULES:
            raise ValueError(
                f"Solver '{solver_name}' not supported. "
                f"Available CHT solvers: chtMultiRegionFoam, "
                f"chtMultiRegionSimpleFoam"
            )

        self.regions = regions or []
        self.interfaces = interfaces or []

        # Build default region solvers if not provided
        if region_solvers is None:
            region_solvers = {}
            for region in self.regions:
                if isinstance(region, SolidRegion):
                    region_solvers[region.name] = "solid"
                else:
                    region_solvers[region.name] = "fluid"

        self._region_solvers = region_solvers

        # Default turbulence model for fluid regions
        if "turbulence_model" not in kwargs or kwargs["turbulence_model"] is None:
            kwargs["turbulence_model"] = "kOmegaSST"

        super().__init__(
            case_path=case_path,
            solver_name=solver_name,
            energy_activated=True,
            is_solid=any(
                isinstance(r, SolidRegion) for r in self.regions
            ),
            transient="Simple" not in solver_name,
            **kwargs,
        )

    def setup_case(self) -> None:
        """Set up the full multi-region CHT case.

        Calls the parent ``setup_case`` first (which writes the
        base system / constant files), then writes region-specific
        directories, field files, thermophysical properties, and
        interface definitions.
        """
        super().setup_case()
        self._set_region_solvers_on_controlDict()
        self._write_region_directories()
        self._write_region_fields()
        self._write_region_solid_properties()
        self._write_region_thermophysical_properties()
        self._write_interfaces()

    def _set_region_solvers_on_controlDict(self) -> None:
        """Inject ``regionSolvers`` into the case controlDict so that
        ``chtMultiRegionFoam`` knows which solver module to use for
        each region."""
        if self._region_solvers:
            self.system.controlDict.set_region_solvers(self._region_solvers)

    def _write_region_directories(self) -> None:
        for region in self.regions:
            region_dir = self.case_path / "0" / region.name
            region_dir.mkdir(parents=True, exist_ok=True)

    def _write_region_fields(self) -> None:
        for region in self.regions:
            region_path = self.case_path / "0" / region.name

            # Temperature field — both fluid and solid
            T_file = region_path / "T"
            T_file.write_text(region.get_T_field_content())

            # Velocity field — only for fluid regions
            if isinstance(region, FluidRegion):
                U_file = region_path / "U"
                U_file.write_text(region.get_U_field_content())

    def _write_region_solid_properties(self) -> None:
        for region in self.regions:
            if isinstance(region, SolidRegion):
                const_dir = self.case_path / "constant" / region.name
                const_dir.mkdir(parents=True, exist_ok=True)

                tp_file = const_dir / "thermophysicalProperties"
                tp_file.write_text(region.get_thermophysical_properties())

                tp_file = const_dir / "transportProperties"
                tp_file.write_text(region.get_transport_properties())

    def _write_region_thermophysical_properties(self) -> None:
        """Write ``thermophysicalProperties`` for fluid regions.

        Fluid regions in CHT typically need a compressible thermo
        model (``heRhoThermo``) in the ``constant/<region>/`` directory.
        """
        for region in self.regions:
            if isinstance(region, FluidRegion):
                const_dir = self.case_path / "constant" / region.name
                const_dir.mkdir(parents=True, exist_ok=True)

                tp_file = const_dir / "thermophysicalProperties"
                tp_file.write_text(region.get_thermophysical_properties())

    def _write_interfaces(self) -> None:
        if not self.interfaces:
            return

        interface_dir = self.case_path / "constant" / "regionInterfaces"
        interface_dir.mkdir(parents=True, exist_ok=True)

        for interface in self.interfaces:
            interface_file = interface_dir / f"{interface.name}.dict"
            interface_file.write_text(interface.get_content())

    def update_case_specific_attributes(self) -> None:
        """CHT-specific attributes are handled in ``setup_case``."""
        pass

    def run_simulation(self, nb_proc: int = 1, log_filename: str | None = None):
        """Run the CHT simulation.

        For ``chtMultiRegionFoam`` and ``chtMultiRegionSimpleFoam``,
        the solver is launched directly as a standalone binary rather
        than through ``foamRun -solver``.

        Args:
            nb_proc: Number of processors (1 for serial).
            log_filename: Name of the log file.
        """
        if log_filename is None:
            log_filename = f"log.{self.solver_name}"

        logger.info("Running CHT simulation: %s", self.solver_name)

        if nb_proc >= 2:
            return self._run_parallel(nb_proc, log_filename)

        self.run_command(
            [self.solver_name, "-case", str(self.case_path)],
            log_filename,
        )

    def _run_parallel(self, nb_proc: int, log_filename: str):
        """Run a parallel CHT simulation."""
        log_path = self.case_path / log_filename
        logger.info("Parallel CHT run with %d processors", nb_proc)

        if hasattr(self.system, "ensure_decomposeParDict"):
            self.system.ensure_decomposeParDict(nb_proc)
            self.system.write()

        import subprocess as _sp
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=== decomposePar -allRegions ===\n")
            _sp.run(
                ["decomposePar", "-allRegions", "-case", str(self.case_path)],
                stdout=log_file, stderr=_sp.STDOUT, check=True,
            )
            log_file.write(f"\n=== mpirun {self.solver_name} -parallel ===\n")
            _sp.run(
                ["mpirun", "-np", str(nb_proc), self.solver_name, "-parallel",
                 "-case", str(self.case_path)],
                cwd=str(self.case_path),
                stdout=log_file, stderr=_sp.STDOUT, check=True,
            )
            log_file.write("\n=== reconstructPar -allRegions ===\n")
            _sp.run(
                ["reconstructPar", "-allRegions", "-case", str(self.case_path)],
                stdout=log_file, stderr=_sp.STDOUT, check=True,
            )
        logger.info("Parallel CHT simulation finished (log: %s)", log_path)
