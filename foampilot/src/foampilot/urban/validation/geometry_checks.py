from dataclasses import dataclass
from typing import List, Tuple, Optional

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class GeometryValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]


class GeometryValidator:
    def __init__(self, options: Optional[dict] = None):
        self.options = options or {}

    def validate_building(self, building) -> GeometryValidationResult:
        errors = []
        warnings = []

        if not building.footprint.is_valid:
            errors.append(f"Building {building.id}: invalid footprint")

        if building.roof_z <= building.ground_z:
            errors.append(f"Building {building.id}: roof_z <= ground_z")

        if building.height < 0:
            errors.append(f"Building {building.id}: negative height")

        min_area = self.options.get("min_area", ValueWithUnit(1.0, "m^2"))
        min_height = self.options.get("min_height", ValueWithUnit(0.5, "m"))

        if building.area < min_area.get_in("m^2"):
            warnings.append(f"Building {building.id}: area below threshold")

        if building.height < min_height.get_in("m"):
            warnings.append(f"Building {building.id}: height below threshold")

        return GeometryValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_urban_model(self, urban) -> GeometryValidationResult:
        all_errors = []
        all_warnings = []

        for b in urban.buildings():
            result = self.validate_building(b)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return GeometryValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )
