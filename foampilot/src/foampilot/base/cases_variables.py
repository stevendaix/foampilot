from __future__ import annotations
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
from foampilot.utilities.manageunits import ValueWithUnit
from foampilot.base.openFOAMFile import OpenFOAMFile


class CaseFieldsManager:
    """Dynamically generates OpenFOAM fields based on solver and physical configurations.

    This manager automates the selection of required initial field files (e.g., U, p, T, k)
    by inspecting the physical characteristics of the simulation. It adapts to
    compressibility, gravity effects, multiphase flows (VOF), radiation, and
    various turbulence models.

    For conjugate heat transfer (CHT) cases, pass the ``regions`` parameter to
    generate per-region field sets.  Solid regions will only receive a ``T``
    field (no ``U``), while fluid regions receive ``U``, pressure, and
    turbulence fields as appropriate.

    Attributes:
        compressible (bool): Whether the solver handles compressible flow.
        with_gravity (bool): If True, uses gravity-related fields (e.g., `p_rgh`).
        is_vof (bool): If True, adds Volume of Fluid phase fraction fields.
        is_solid (bool): If True, restricts fields to solid-state physics (e.g., only T).
        energy_activated (bool): Whether heat transfer/energy equations are solved.
        with_radiation (bool): Whether radiation models are enabled.
        turbulence_model (str): The specific turbulence model used (determines fields like k, epsilon, omega).
        fields (Dict[str, Dict[str, Any]]): Dictionary storing the generated field configurations and initial values.
        physical_properties (Dict[str, ValueWithUnit]): Registry for physical constants (reserved for future use).
        turbulence_properties (Dict[str, Any]): Registry for turbulence constants (reserved for future use).
        regions (list, optional): List of ``FluidRegion`` / ``SolidRegion`` objects for CHT multi-region field generation.
        region_fields (Dict[str, Dict[str, Dict[str, Any]]]): Per-region field configurations keyed by region name.
    """

    def __init__(
        self,
        *,
        compressible: bool = False,
        with_gravity: bool = False,
        is_vof: bool = False,
        is_solid: bool = False,
        energy_activated: bool = False,
        with_radiation: bool = False,
        turbulence_model: Optional[str] = None,
        regions: Optional[List[Any]] = None,
        with_moving_mesh: bool = False,
    ):
        """Initializes the CaseFieldsManager and triggers initial field generation.

        Args:
            compressible: Enable compressible flow fields. Defaults to False.
            with_gravity: Account for gravity (switching p to p_rgh in incompressible). Defaults to False.
            is_vof: Enable multiphase (Volume of Fluid) fields. Defaults to False.
            is_solid: Set up for solid-only heat transfer. Defaults to False.
            energy_activated: Enable temperature fields for heat transfer. Defaults to False.
            with_radiation: Enable radiation-specific fields (G, q_r). Defaults to False.
            turbulence_model: Name of the turbulence model (e.g., "kEpsilon", "kOmegaSST").
                Defaults to None which is treated as "kEpsilon" internally.
            regions: Optional list of ``FluidRegion`` or ``SolidRegion`` objects for
                CHT multi-region field generation.  When provided, ``region_fields``
                is populated with per-region field dicts.
            with_moving_mesh: If True, adds ``pointDisplacement`` field for moving mesh.
        """
        self.compressible = compressible
        self.with_gravity = with_gravity
        self.is_vof = is_vof
        self.is_solid = is_solid
        self.energy_activated = energy_activated
        self.with_radiation = with_radiation
        self.turbulence_model = turbulence_model or "kEpsilon"
        self.regions = regions
        self.with_moving_mesh = with_moving_mesh
        self.region_fields: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Storage
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.physical_properties: Dict[str, ValueWithUnit] = {}
        self.turbulence_properties: Dict[str, Any] = {}
        self.custom_initial_values: Dict[str, Any] = {}

        self._generate_fields()

        # Generate per-region fields if regions were provided
        if self.regions:
            self._generate_region_fields()

    def _generate_fields(self) -> None:
        """Internal logic to populate the fields dictionary based on physical flags.

        This method clears the current field list and re-evaluates which OpenFOAM
        files are required (e.g., deciding between 'p' and 'p_rgh' or adding
        turbulence scalars).
        """
        # Reset
        self.fields.clear()

        # --- Base pressure and velocity fields
        pressure_name = "p_rgh" if self.with_gravity and not self.compressible else "p"
        self.fields[pressure_name] = {"value": ValueWithUnit(0, "Pa")}
        if not self.is_solid:
            self.fields["U"] = {"value": ValueWithUnit(0, "m/s")}

        # --- Volume fraction (VOF)
        if self.is_vof:
            self.fields["alpha.water"] = {"value": ValueWithUnit(1.0, "")}
            self.fields["alpha.air"] = {"value": ValueWithUnit(0.0, "")}

        # --- Energy or temperature field
        if self.energy_activated or self.compressible:
            self.fields["T"] = {"value": ValueWithUnit(300, "K")}

        # --- Radiation
        if self.with_radiation:
            self.fields["G"] = {"value": ValueWithUnit(0, "W/m^2")}
            self.fields["q_r"] = {"value": ValueWithUnit(0, "W/m^2")}

        # --- Turbulence model fields
        if self.turbulence_model:
            self._generate_turbulence_fields()

        if self.is_solid:
            self.fields = {"T": {"value": ValueWithUnit(300, "K")}}
            if self.with_moving_mesh:
                self.fields["pointDisplacement"] = {"value": ValueWithUnit(0, "m")}
        else:
            if self.with_moving_mesh:
                self.fields["pointDisplacement"] = {"value": ValueWithUnit(0, "m")}

    def _generate_turbulence_fields(self) -> None:
        """Internal logic to add scalars and vectors required by turbulence models.

        Supported model keywords include 'kepsilon', 'omega', 'spalart', and 'v2'.
        """
        model = self.turbulence_model.lower()

        if model == "laminar":
            return

        if model.startswith("les:") and "keqn" in model:
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "kepsilon" in model:
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}
            self.fields["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "omega" in model:
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["omega"] = {"value": ValueWithUnit(1, "1/s")}
            self.fields["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "spalart" in model:
            self.fields["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
            self.fields["nuTilda"] = {"value": ValueWithUnit(0.05, "m^2/s")}
        elif "v2" in model:
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}
            self.fields["v2"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
        else:
            # Default to k-epsilon if unknown
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}

    # ------------------------------------------------------------------
    # Multi-region field generation
    # ------------------------------------------------------------------

    def _generate_region_fields(self) -> None:
        """Populate ``region_fields`` for each region in ``self.regions``.

        Solid regions get only ``T``; fluid regions get ``U``, pressure,
        temperature (if energy activated), and turbulence fields.
        """
        from foampilot.cht.regions import SolidRegion

        self.region_fields = {}

        for region in self.regions:
            region_name = region.name
            is_solid = isinstance(region, SolidRegion)

            region_turbulence = (
                region.turbulence_model
                if hasattr(region, "turbulence_model")
                else self.turbulence_model
            )

            region_fields: Dict[str, Dict[str, Any]] = {}

            pressure_name = "p_rgh" if self.with_gravity and not self.compressible else "p"
            region_fields[pressure_name] = {"value": ValueWithUnit(0, "Pa")}

            if not is_solid:
                region_fields["U"] = {"value": ValueWithUnit(0, "m/s")}

            if self.is_vof and not is_solid:
                region_fields["alpha.water"] = {"value": ValueWithUnit(1.0, "")}
                region_fields["alpha.air"] = {"value": ValueWithUnit(0.0, "")}

            if self.energy_activated or self.compressible:
                region_fields["T"] = {"value": ValueWithUnit(region.temperature, "K")}

            if self.with_radiation:
                region_fields["G"] = {"value": ValueWithUnit(0, "W/m^2")}
                region_fields["q_r"] = {"value": ValueWithUnit(0, "W/m^2")}

            if region_turbulence and not is_solid:
                self._generate_turbulence_fields_for(region_fields, region_turbulence)

            if not is_solid and self.with_moving_mesh:
                region_fields["pointDisplacement"] = {"value": ValueWithUnit(0, "m")}

            if is_solid:
                region_fields = {"T": {"value": ValueWithUnit(region.temperature, "K")}}
                if self.with_moving_mesh:
                    region_fields["pointDisplacement"] = {"value": ValueWithUnit(0, "m")}

            self.region_fields[region_name] = region_fields

    @staticmethod
    def _generate_turbulence_fields_for(
        fields_dict: Dict[str, Dict[str, Any]], model: str
    ) -> None:
        """Populate ``fields_dict`` with turbulence scalars for the given model."""
        model_lower = model.lower()

        if model_lower == "laminar":
            return

        if "kepsilon" in model_lower:
            fields_dict["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            fields_dict["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}
            fields_dict["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "omega" in model_lower:
            fields_dict["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            fields_dict["omega"] = {"value": ValueWithUnit(1, "1/s")}
            fields_dict["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "spalart" in model_lower:
            fields_dict["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "v2" in model_lower:
            fields_dict["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            fields_dict["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}
            fields_dict["v2"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
        else:
            fields_dict["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            fields_dict["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}

    def get_region_field_names(self, region_name: str) -> List[str]:
        """Return the list of field names generated for a specific region.

        Args:
            region_name: Name of the region.

        Returns:
            List of field names (e.g. ``['T', 'U', 'k', 'omega', 'nut']``).
        """
        return list(self.region_fields.get(region_name, {}).keys())

    def get_region_fields(self, region_name: str) -> Dict[str, Dict[str, Any]]:
        """Return the field configuration dict for a specific region.

        Args:
            region_name: Name of the region.

        Returns:
            Dictionary mapping field names to their config dicts.
        """
        return self.region_fields.get(region_name, {})

    def write_region_boundary_files(
        self,
        region_name: str,
        case_path: Union[str, Path],
        boundaries: Dict[str, Dict[str, Any]],
        internal_field_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        """Write boundary-condition field files for a specific region.

        This method writes each field file into ``0/<region_name>/``
        using :meth:`OpenFOAMFile.write_boundary_file`.

        Args:
            region_name: Name of the region (e.g. ``"fluid"``, ``"solid"``).
            case_path: Path to the OpenFOAM case root directory.
            boundaries: Dict mapping patch names to BC parameter dicts.
            internal_field_overrides: Optional dict mapping field names to
                custom ``internalField`` strings.
        """
        internal_field_overrides = internal_field_overrides or {}
        foam_file = OpenFOAMFile("region_boundary")

        for field_name in self.get_region_field_names(region_name):
            region_0_path = Path(case_path) / "0" / region_name
            region_0_path.mkdir(parents=True, exist_ok=True)

            foam_file.write_boundary_file(
                field=field_name,
                boundaries=boundaries,
                case_path=str(region_0_path),
                internal_field=internal_field_overrides.get(field_name),
            )

    # ------------------------------------------------------------------
    # Public API (backward-compatible)
    # ------------------------------------------------------------------

    def import_reference_field(self, source_path: Union[str, Path], case_path: Union[str, Path], field_name: Optional[str] = None) -> Path:
        """Import a complete OpenFOAM initial field without lossy parsing."""
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        target_dir = Path(case_path) / "0"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (field_name or source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".gz":
            import gzip
            target.write_bytes(gzip.decompress(source.read_bytes()))
        else:
            target.write_bytes(source.read_bytes())
        return target

    def get_field_names(self) -> list[str]:
        """Returns the names of all generated fields."""
        return list(self.fields.keys())

    def set_vof_primary_phase(self, phase: str) -> None:
        """Select the primary VoF phase field, e.g. ``alpha.vapour``.

        This replaces the historical ``alpha.water``/``alpha.air`` defaults
        while preserving the generic VoF field-generation workflow.
        """
        if not self.is_vof:
            raise ValueError("set_vof_primary_phase requires a VoF case")
        if not phase or any(ch.isspace() for ch in phase):
            raise ValueError("phase must be a non-empty OpenFOAM word")
        for field_name in list(self.fields):
            if field_name.startswith("alpha."):
                self.fields.pop(field_name)
        self.fields[f"alpha.{phase}"] = {"value": ValueWithUnit(0.0, "")}

    def register_field(self, name: str, value: Any, unit: str = "") -> None:
        """Register an additional OpenFOAM field required by a reference case.

        The field is handled by the normal system, constant and boundary writers;
        no direct file manipulation is needed in a tutorial runner.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Field name must be a non-empty string")
        if isinstance(value, str) and (value.startswith("uniform ") or value.startswith("nonuniform ")):
            self.fields[name] = {"value": value}
        else:
            self.fields[name] = {"value": ValueWithUnit(value, unit)}
        self.custom_initial_values[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Exports the field configurations to a simplified dictionary format.

        This is primarily used for serialization or for passing data to
        other OpenFOAM dictionary writers.

        Returns:
            Dict[str, Any]: A dictionary where keys are field names and
                values are string representations of their magnitudes and units.
        """
        return {k: str(v["value"]) for k, v in self.fields.items()}

    def __repr__(self) -> str:
        flags = (
            f"compressible={self.compressible}, gravity={self.with_gravity}, vof={self.is_vof}, "
            f"solid={self.is_solid}, energy={self.energy_activated}, radiation={self.with_radiation}, "
            f"model={self.turbulence_model}, "
            f"regions={len(self.regions) if self.regions else 0}"
        )
        return f"<CaseFieldsManager {flags}>"
