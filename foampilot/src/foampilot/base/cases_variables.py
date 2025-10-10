from typing import Dict, Any, Optional, List, Tuple
from foampilot.utilities.manageunits import Quantity


class CaseFieldsManager:
    """
    Gère la création dynamique des champs OpenFOAM en fonction des flags du solveur.
    Exemples :
      - with_gravity=True → utilise p_rgh au lieu de p
      - is_vof=True → ajoute alpha.water, alpha.air
      - energy_activated=True → ajoute T
      - turbulence_model="kEpsilon" → ajoute k, epsilon, nut
    """
    def __init__(
        self,
        is_solid: bool = False,
        with_gravity: bool = False,
        is_vof: bool = False,
        energy_activated: bool = False,
        turbulence_model: Optional[str] = None,
        with_radiation: bool = False,
    ):
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.is_solid = is_solid
        self.with_gravity = with_gravity
        self.is_vof = is_vof
        self.energy_activated = energy_activated
        self.turbulence_model = turbulence_model
        self.with_radiation = with_radiation

        # Génère les champs nécessaires
        self._generate_fields()

    def _generate_fields(self) -> None:
        """Génère les champs en fonction des flags."""
        # --- Champs de base pour un fluide ---
        if not self.is_solid:
            # Pression : p ou p_rgh selon la gravité
            pressure_field = "p_rgh" if self.with_gravity else "p"
            self.add_field(pressure_field, 0, "Pa")

            # Vitesse
            self.add_field("U", (0, 0, 0), "m/s")

            # VoF : alpha.water, alpha.air
            if self.is_vof:
                self.add_field("alpha.water", 0, "-")
                self.add_field("alpha.air", 1, "-")

            # Rayonnement : K
            if self.with_radiation:
                self.add_field("K", 0, "1/m")

        # --- Champs pour un solide ---
        else:
            self.add_field("D", (0, 0, 0), "m")  # Déplacement
            if self.energy_activated:
                self.add_field("T", 300, "K")     # Température

        # --- Énergie : T (si fluide et énergie activée) ---
        if not self.is_solid and self.energy_activated:
            self.add_field("T", 300, "K")

        # --- Turbulence : k-ε ou k-ω ---
        if self.turbulence_model == "kEpsilon":
            self.add_field("k", 0.01, "m²/s²")
            self.add_field("epsilon", 0.01, "m²/s³")
            self.add_field("nut", 0, "m²/s")
        elif self.turbulence_model == "kOmegaSST":
            self.add_field("k", 0.01, "m²/s²")
            self.add_field("omega", 1, "1/s")
            self.add_field("nut", 0, "m²/s")

    def add_field(self, name: str, value: Any, units: Optional[str] = None) -> None:
        """Ajoute un champ avec sa valeur et ses unités."""
        if units:
            value = Quantity(value, units) if not isinstance(value, Quantity) else value
        self.fields[name] = {"value": value}

    def get_field_names(self) -> List[str]:
        """Retourne la liste des noms de champs."""
        return list(self.fields.keys())

    def get_field(self, name: str) -> Dict[str, Any]:
        """Retourne un champ par son nom."""
        return self.fields.get(name, {})
