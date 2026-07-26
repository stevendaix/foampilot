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
    """

    def __init__(
        self,
        case_path: str | Path,
        solver_name: str = "chtMultiRegionFoam",
        regions: Optional[List[Any]] = None,
        interfaces: Optional[List[CoupledInterface]] = None,
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

        super().__init__(
            case_path=case_path,
            solver_name=solver_name,
            energy_activated=True,
            is_solid=any(
                isinstance(r, SolidRegion) for r in self.regions
            ),
            **kwargs,
        )

    def setup_case(self) -> None:
        super().setup_case()
        self._write_region_directories()
        self._write_region_fields()
        self._write_region_solid_properties()
        self._write_interfaces()

    def _write_region_directories(self) -> None:
        for region in self.regions:
            region_dir = self.case_path / "0" / region.name
            region_dir.mkdir(parents=True, exist_ok=True)

    def _write_region_fields(self) -> None:
        for region in self.regions:
            region_path = self.case_path / "0" / region.name
            T_file = region_path / "T"
            T_file.write_text(region.get_T_field_content())

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

    def _write_interfaces(self) -> None:
        if not self.interfaces:
            return

        interface_dir = self.case_path / "constant" / "regionInterfaces"
        interface_dir.mkdir(parents=True, exist_ok=True)

        for interface in self.interfaces:
            interface_file = interface_dir / f"{interface.name}.dict"
            interface_file.write_text(interface.get_content())

    def update_case_specific_attributes(self) -> None:
        pass