import logging
import shutil

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
        self.write_region_system_files()
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
                scalar_fields = {
                    "p": ("0", "[1 -1 -2 0 0 0 0]"),
                    "p_rgh": ("0", "[1 -1 -2 0 0 0 0]"),
                    "k": ("0", "[0 2 -2 0 0 0 0]"),
                    "omega": ("0", "[0 0 -1 0 0 0 0]"),
                    "nut": ("0", "[0 2 -1 0 0 0 0]"),
                }
                for field_name, (value, dimensions) in scalar_fields.items():
                    (region_path / field_name).write_text(
                        region.get_scalar_field_content(field_name, value, dimensions)
                    )

    def set_region_momentum_transport(self, region_name: str, simulation_type: str = "laminar") -> None:
        """Write OF13 ``momentumTransport`` for a region."""
        region_dir = self.case_path / "constant" / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        (region_dir / "momentumTransport").write_text(
            "FoamFile\n{\n    format ascii;\n    class dictionary;\n"
            f'    location "constant/{region_name}";\n    object momentumTransport;\n}}\n\n'
            f"simulationType  {simulation_type};\n"
        )

    def set_region_gravity(self, region_name: str, value: str = "(0 0 0)") -> None:
        """Write the OF13 uniformDimensionedVectorField ``g`` for a region."""
        region_dir = self.case_path / "constant" / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        (region_dir / "g").write_text(
            "FoamFile\n{\n    format ascii;\n    class uniformDimensionedVectorField;\n"
            f'    location "constant/{region_name}";\n    object g;\n}}\n\n'
            "dimensions [0 1 -2 0 0 0 0];\n"
            f"value {value};\n"
        )

    def _write_region_solid_properties(self) -> None:
        for region in self.regions:
            if isinstance(region, SolidRegion):
                const_dir = self.case_path / "constant" / region.name
                const_dir.mkdir(parents=True, exist_ok=True)

                if self.solver_name == "chtMultiRegionFoam" and hasattr(region, "get_physical_properties"):
                    tp_file = const_dir / "physicalProperties"
                    tp_file.write_text(region.get_physical_properties())
                else:
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

                if self.solver_name == "chtMultiRegionFoam" and hasattr(region, "get_physical_properties"):
                    tp_file = const_dir / "physicalProperties"
                    tp_file.write_text(region.get_physical_properties())
                else:
                    tp_file = const_dir / "thermophysicalProperties"
                    tp_file.write_text(region.get_thermophysical_properties())

    def write_region_system_files(self) -> None:
        """Create OF13 per-region dictionaries and neutralize mono-region functions."""
        functions_file = self.case_path / "system" / "functions"
        if functions_file.exists():
            content = functions_file.read_text()
            header = content.split("// * * *", 1)[0]
            functions_file.write_text(header + "// Multi-region functions are configured explicitly per case.\n")
        for region in self.regions:
            region_system = self.case_path / "system" / region.name
            region_system.mkdir(parents=True, exist_ok=True)
            for filename in ("fvSchemes", "fvSolution"):
                source = self.case_path / "system" / filename
                if source.exists():
                    target = region_system / filename
                    shutil.copyfile(source, target)
                    if filename == "fvSolution" and isinstance(region, SolidRegion):
                        content = target.read_text()
                        additions = (
                            "    e { solver GAMG; smoother symGaussSeidel; tolerance 1e-6; relTol 0.1; }\n"
                            "    eFinal { $e; relTol 0; }\n"
                        )
                        content = content.replace("solvers\n{\n", "solvers\n{\n" + additions, 1)
                        target.write_text(content)
                    if filename == "fvSolution" and isinstance(region, FluidRegion):
                        content = target.read_text()
                        additions = (
                            "    rho { solver diagonal; }\n"
                            "    rhoFinal { $rho; relTol 0; }\n"
                            "    p_rgh { solver GAMG; smoother symGaussSeidel; tolerance 1e-7; relTol 0.01; }\n"
                            "    p_rghFinal { $p_rgh; relTol 0; }\n"
                            "    h { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.1; }\n"
                            "    hFinal { $h; relTol 0; }\n"
                        )
                        content = content.replace("solvers\n{\n", "solvers\n{\n" + additions, 1)
                        target.write_text(content)
                    if filename == "fvSchemes" and isinstance(region, FluidRegion):
                        content = target.read_text()
                        schemes = (
                            "    div(phi,K) Gauss linear;\n"
                            "    div(phi,h) Gauss upwind;\n"
                            "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n"
                        )
                        content = content.replace("divSchemes\n{\n", "divSchemes\n{\n" + schemes, 1)
                        target.write_text(content)

    def _write_interfaces(self) -> None:
        if not self.interfaces:
            return

        interface_dir = self.case_path / "constant" / "regionInterfaces"
        interface_dir.mkdir(parents=True, exist_ok=True)

        for interface in self.interfaces:
            interface_file = interface_dir / f"{interface.name}.dict"
            interface_file.write_text(interface.get_content())

    def set_region_boundary_conditions(self, region_name: str, field_name: str, conditions: Dict[str, Dict[str, str]]) -> None:
        """Write explicit OpenFOAM boundaryField entries for a region field."""
        import re
        field_path = self.case_path / "0" / region_name / field_name
        if not field_path.exists():
            raise FileNotFoundError(f"Region field not found: {field_path}")
        boundary_path = self.case_path / "constant" / region_name / "polyMesh" / "boundary"
        boundary_text = boundary_path.read_text() if boundary_path.exists() else ""
        patch_blocks = re.findall(r"^\s{4}([A-Za-z0-9_./:+-]+)\s*\n\s*\{(.*?)^\s{4}\}", boundary_text, flags=re.MULTILINE | re.DOTALL)
        patch_types = {name: (re.search(r"\btype\s+([^;]+);", block).group(1).strip() if re.search(r"\btype\s+([^;]+);", block) else "patch") for name, block in patch_blocks}
        patch_names = list(patch_types)
        fallback = conditions.get(".*", {})
        expanded: Dict[str, Dict[str, str]] = {
            patch: ({"type": "empty"} if patch_types[patch] == "empty" else dict(fallback))
            for patch in patch_names
        }
        for patch, entries in conditions.items():
            if patch != ".*":
                expanded[patch] = entries
        lines = ["boundaryField", "{"]
        for patch, entries in expanded.items():
            lines.extend([f"    {patch}", "    {"])
            for key, value in entries.items():
                lines.append(f"        {key} {value};")
            lines.append("    }")
        lines.append("}")
        content = field_path.read_text()
        content = re.sub(r"boundaryField\s*\{.*\n\}", "\n".join(lines), content, count=1, flags=re.DOTALL)
        field_path.write_text(content)

    def set_region_internal_field(self, region_name: str, field_name: str, value: str) -> None:
        """Set a region field internalField without shelling out to foamDictionary."""
        import re
        field_path = self.case_path / "0" / region_name / field_name
        if not field_path.exists():
            raise FileNotFoundError(f"Region field not found: {field_path}")
        content = field_path.read_text()
        content, count = re.subn(
            r"(^\s*internalField\s+)(?:uniform\s+[^;]+|nonuniform\s+[^;]+);",
            rf"\g<1>{value};",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"internalField not found in {field_path}")
        field_path.write_text(content)

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

        executable = "foamMultiRun" if self.solver_name == "chtMultiRegionFoam" else self.solver_name
        self.run_command(
            [executable, "-case", str(self.case_path)],
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
