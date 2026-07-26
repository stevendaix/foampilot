from typing import Dict, Optional, Any, List
from foampilot.base.openFOAMFile import OpenFOAMFile


class FvModelsFile(OpenFOAMFile):
    """
    OpenFOAM fvModels file for source term modeling (porous media,
    fans, heat sources, etc.) as introduced in OpenFOAM v1906+.

    Replaces the older fvOptions system with a more structured approach.
    """

    SUPPORTED_MODELS = [
        "porousZone",
        "fan",
        "heater",
        "source",
        "buoyancy",
        "swirl",
    ]

    def __init__(
        self,
        parent: Optional[Any] = None,
        models: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.parent = parent
        self.models = models or []

        super().__init__(object_name="fvModels")

    def add_porous_zone(
        self,
        name: str,
        patch_names: List[str],
        permeability: Dict[str, float],
        porosity: float = 1.0,
        inertial_density: float = 0.0,
    ) -> None:
        """Add a porous zone model."""
        model = {
            "type": "porousZone",
            "active": True,
            "selectionMode": "patches",
            "patches": patch_names,
            "porousZone": {
                " permeability": permeability,
                "porosity": porosity,
                "inertialDensity": inertial_density,
            },
        }
        self.models.append((name, model))

    def add_fan(
        self,
        name: str,
        patch_names: List[str],
        fan_curve: Dict[str, Any],
        power: float = 0.0,
        origin: Optional[List[float]] = None,
        axis: Optional[List[float]] = None,
    ) -> None:
        """Add a fan model for rotating machinery."""
        model: Dict[str, Any] = {
            "type": "fan",
            "active": True,
            "selectionMode": "patches",
            "patches": patch_names,
            "fan": {
                "fanCurve": fan_curve,
                "power": power,
            },
        }
        if origin is not None:
            model["fan"]["origin"] = origin
        if axis is not None:
            model["fan"]["axis"] = axis
        self.models.append((name, model))

    def add_heat_source(
        self,
        name: str,
        patch_names: List[str],
        heat_source: Dict[str, Any],
    ) -> None:
        """Add a heat source model."""
        model = {
            "type": "source",
            "active": True,
            "selectionMode": "patches",
            "patches": patch_names,
            "source": heat_source,
        }
        self.models.append((name, model))

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, model in self.models:
            result[name] = model
        return result

    def write(self, filepath: str) -> None:
        self.attributes = self.to_dict()
        self.write_file(filepath)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], parent: Optional[Any] = None) -> "FvModelsFile":
        models = []
        for name, model in config.items():
            models.append((name, model))
        return cls(parent=parent, models=models)