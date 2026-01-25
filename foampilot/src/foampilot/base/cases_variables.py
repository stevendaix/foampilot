from __future__ import annotations
from typing import Dict, Optional, Any
from foampilot.utilities.manageunits import ValueWithUnit


class CaseFieldsManager:
    """Dynamically generates OpenFOAM fields based on solver and physical configurations.

    This manager automates the selection of required initial field files (e.g., U, p, T, k) 
    by inspecting the physical characteristics of the simulation. It adapts to 
    compressibility, gravity effects, multiphase flows (VOF), radiation, and 
    various turbulence models.

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
                Defaults to "kEpsilon".
        """
        self.compressible = compressible
        self.with_gravity = with_gravity
        self.is_vof = is_vof
        self.is_solid = is_solid
        self.energy_activated = energy_activated
        self.with_radiation = with_radiation
        self.turbulence_model = turbulence_model or "kEpsilon"

        # Storage
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.physical_properties: Dict[str, ValueWithUnit] = {}
        self.turbulence_properties: Dict[str, Any] = {}

        self._generate_fields()

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

        # --- Solid-specific field
        if self.is_solid:
            self.fields = {"T": {"value": ValueWithUnit(300, "K")}}  # Only temperature in solids

    def _generate_turbulence_fields(self) -> None:
        """Internal logic to add scalars and vectors required by turbulence models.

        Supported model keywords include 'kepsilon', 'omega', 'spalart', and 'v2'.
        """
        model = self.turbulence_model.lower()

        if "kepsilon" in model:
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}
            self.fields["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "omega" in model:
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["omega"] = {"value": ValueWithUnit(1, "1/s")}
            self.fields["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "spalart" in model:
            self.fields["nut"] = {"value": ValueWithUnit(1e-5, "m^2/s")}
        elif "v2" in model:
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}
            self.fields["v2"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
        else:
            # Default to k-epsilon if unknown
            self.fields["k"] = {"value": ValueWithUnit(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": ValueWithUnit(0.1, "m^2/s^3")}

    def get_field_names(self) -> list[str]:
        """Returns the names of all generated fields.

        Returns:
            list[str]: A list of strings representing the field filenames (e.g., ['U', 'p', 'k']).
        """
        return list(self.fields.keys())

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
            f"model={self.turbulence_model}"
        )
        return f"<CaseFieldsManager {flags}>"
