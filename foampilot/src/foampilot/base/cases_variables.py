from typing import Dict, Any, Optional, List, Tuple
from foampilot.utilities.manageunits import Quantity

class CaseVariables:
    def __init__(self, solver_flags: Dict[str, Any]):
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.physical_properties: Dict[str, Quantity] = {}
        self.turbulence_properties: Dict[str, Any] = {}
        self.solver_flags = solver_flags

        # Génère les champs en fonction des flags
        self._generate_fields()

    def _generate_fields(self) -> None:
        """Génère les champs nécessaires en fonction des flags du solveur."""
        # --- Champs de base pour un fluide ---
        if not self.solver_flags.get("is_solid", False):
            # Pression : p ou p_rgh selon la gravité
            pressure_field = "p_rgh" if self.solver_flags.get("with_gravity", False) else "p"
            self.add_field(pressure_field, 0, "Pa")

            # Vitesse
            self.add_field("U", (0, 0, 0), "m/s")

            # VoF : alpha.water, alpha.air, etc.
            if self.solver_flags.get("is_vof", False):
                self.add_field("alpha.water", 0, "-")
                self.add_field("alpha.air", 1, "-")

            # Rayonnement : K ou radiation
            if self.solver_flags.get("with_radiation", False):
                self.add_field("K", 0, "1/m")  # Coefficient d'absorption

        # --- Champs pour un solide ---
        else:
            self.add_field("D", (0, 0, 0), "m")  # Déplacement
            self.add_field("T", 300, "K")       # Température

        # --- Énergie : T ou h ---
        if self.solver_flags.get("energy_activated", False):
            if not self.solver_flags.get("is_solid", False):
                self.add_field("T", 300, "K")  # Température pour fluide
            # Pour un solide, T est déjà ajouté

        # --- Turbulence : k-ε ou k-ω ---
        turbulence_model = self.solver_flags.get("turbulence_model", None)
        if turbulence_model == "kEpsilon":
            self.add_field("k", 0.01, "m²/s²")
            self.add_field("epsilon", 0.01, "m²/s³")
            self.add_field("nut", 0, "m²/s")
        elif turbulence_model == "kOmegaSST":
            self.add_field("k", 0.01, "m²/s²")
            self.add_field("omega", 1, "1/s")
            self.add_field("nut", 0, "m²/s")

    def add_field(self, name: str, value: Any, units: Optional[str] = None) -> None:
        if units:
            value = Quantity(value, units) if not isinstance(value, Quantity) else value
        self.fields[name] = {"value": value}

    def get_field_names(self) -> List[str]:
        return list(self.fields.keys())

    def get_field(self, name: str) -> Dict[str, Any]:
        return self.fields.get(name, {})
