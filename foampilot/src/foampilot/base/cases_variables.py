from __future__ import annotations
from typing import Dict, Optional, Any
from foampilot.utilities.manageunits import Quantity


class CaseFieldsManager:
    """
    Dynamically generate OpenFOAM fields depending on solver configuration.
    Adapts automatically to:
      - compressible / incompressible
      - gravity (p_rgh)
      - VOF
      - radiation
      - temperature / energy
      - turbulence model
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
        self.compressible = compressible
        self.with_gravity = with_gravity
        self.is_vof = is_vof
        self.is_solid = is_solid
        self.energy_activated = energy_activated
        self.with_radiation = with_radiation
        self.turbulence_model = turbulence_model or "kEpsilon"

        # Storage
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.physical_properties: Dict[str, Quantity] = {}
        self.turbulence_properties: Dict[str, Any] = {}

        self._generate_fields()

    # ------------------------------------------------------------------ #
    def _generate_fields(self) -> None:
        """Generate field dictionary depending on physical flags."""

        # Reset
        self.fields.clear()

        # --- Base pressure and velocity fields
        pressure_name = "p_rgh" if self.with_gravity and not self.compressible else "p"
        self.fields[pressure_name] = {"value": Quantity(0, "Pa")}
        if not self.is_solid:
            self.fields["U"] = {"value": Quantity(0, "m/s")}

        # --- Volume fraction (VOF)
        if self.is_vof:
            self.fields["alpha.water"] = {"value": Quantity(1.0, "")}
            self.fields["alpha.air"] = {"value": Quantity(0.0, "")}

        # --- Energy or temperature field
        if self.energy_activated or self.compressible:
            self.fields["T"] = {"value": Quantity(300, "K")}

        # --- Radiation
        if self.with_radiation:
            self.fields["G"] = {"value": Quantity(0, "W/m^2")}
            self.fields["q_r"] = {"value": Quantity(0, "W/m^2")}

        # --- Turbulence model fields
        if self.turbulence_model:
            self._generate_turbulence_fields()

        # --- Solid-specific field
        if self.is_solid:
            self.fields = {"T": {"value": Quantity(300, "K")}}  # Only temperature in solids

    # ------------------------------------------------------------------ #
    def _generate_turbulence_fields(self) -> None:
        """Add fields depending on turbulence model."""
        model = self.turbulence_model.lower()

        if "kepsilon" in model:
            self.fields["k"] = {"value": Quantity(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": Quantity(0.1, "m^2/s^3")}
            self.fields["nut"] = {"value": Quantity(1e-5, "m^2/s")}
        elif "omega" in model:
            self.fields["k"] = {"value": Quantity(0.1, "m^2/s^2")}
            self.fields["omega"] = {"value": Quantity(1, "1/s")}
            self.fields["nut"] = {"value": Quantity(1e-5, "m^2/s")}
        elif "spalart" in model:
            self.fields["nut"] = {"value": Quantity(1e-5, "m^2/s")}
        elif "v2" in model:
            self.fields["k"] = {"value": Quantity(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": Quantity(0.1, "m^2/s^3")}
            self.fields["v2"] = {"value": Quantity(0.1, "m^2/s^2")}
        else:
            # Default to k-epsilon if unknown
            self.fields["k"] = {"value": Quantity(0.1, "m^2/s^2")}
            self.fields["epsilon"] = {"value": Quantity(0.1, "m^2/s^3")}

    # ------------------------------------------------------------------ #
    def get_field_names(self) -> list[str]:
        """Return the list of field names."""
        return list(self.fields.keys())

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Export to a simplified dict for serialization."""
        return {k: str(v["value"]) for k, v in self.fields.items()}

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        flags = (
            f"compressible={self.compressible}, gravity={self.with_gravity}, vof={self.is_vof}, "
            f"solid={self.is_solid}, energy={self.energy_activated}, radiation={self.with_radiation}, "
            f"model={self.turbulence_model}"
        )
        return f"<CaseFieldsManager {flags}>"