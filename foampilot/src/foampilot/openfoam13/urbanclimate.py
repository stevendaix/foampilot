"""Typed urban climate profiles and the public FoamPilot case API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .physics import PhysicsConfig, check_openfoam13_case


@dataclass(frozen=True)
class UrbanClimateProfile:
    name: str
    description: str
    regions: tuple[str, ...]
    vegetation: bool = False
    ham: bool = False
    radiation: bool = False


PROFILES: dict[str, UrbanClimateProfile] = {
    "streetCanyon_CFD": UrbanClimateProfile("streetCanyon_CFD", "Single-region street canyon CFD", ("air",)),
    "streetCanyon_CFDHAM": UrbanClimateProfile("streetCanyon_CFDHAM", "Street canyon with heat-air-moisture coupling", ("air", "ground", "buildings"), ham=True, radiation=True),
    "streetCanyon_CFDHAM_grass": UrbanClimateProfile("streetCanyon_CFDHAM_grass", "Street canyon HAM case with grass", ("air", "ground", "buildings", "vegetation"), ham=True, radiation=True, vegetation=True),
    "streetCanyon_CFDHAM_veg": UrbanClimateProfile("streetCanyon_CFDHAM_veg", "Street canyon HAM case with vegetation", ("air", "ground", "buildings", "vegetation"), ham=True, radiation=True, vegetation=True),
    "windAroundBuildings_CFDHAM": UrbanClimateProfile("windAroundBuildings_CFDHAM", "Wind around buildings with HAM", ("air", "ground", "buildings"), ham=True, radiation=True),
    "windAroundBuildings_CFDHAM_veg": UrbanClimateProfile("windAroundBuildings_CFDHAM_veg", "Wind around buildings with HAM and vegetation", ("air", "ground", "buildings", "vegetation"), ham=True, radiation=True, vegetation=True),
}


class UrbanClimateCase:
    """Specialized public facade for native multi-region case generation.

    The case is generated from typed profile configuration; no complete case
    tree is copied.
    """

    def __init__(self, profile: UrbanClimateProfile):
        self.profile = profile

    @classmethod
    def from_name(cls, name: str) -> "UrbanClimateCase":
        try:
            return cls(PROFILES[name])
        except KeyError as exc:
            raise ValueError(f"Unknown urbanclimate profile: {name}") from exc

    def _region_specs(self):
        from .urbanclimate_native import RegionSpec
        specs = [RegionSpec("air", "fluid", temperature=300.0, velocity=(1.0, 0.0, 0.0))]
        if self.profile.ham:
            specs.extend((RegionSpec("ground", "solid"), RegionSpec("buildings", "solid")))
        if self.profile.vegetation:
            specs.append(RegionSpec("vegetation", "vegetation"))
        return tuple(specs)

    def write_case(
        self,
        destination: str | Path,
        *,
        overwrite: bool = False,
        physics: PhysicsConfig | None = None,
        urban_properties: Mapping[str, Any] | None = None,
    ) -> Path:
        from .urbanclimate_native import UrbanClimateNativeCaseBuilder
        properties = {
            "profile": self.profile.name,
            "ham": self.profile.ham,
            "radiation": self.profile.radiation,
            "vegetation": self.profile.vegetation,
        }
        if urban_properties:
            properties.update(urban_properties)
        builder = UrbanClimateNativeCaseBuilder(
            destination,
            self._region_specs(),
            profile=self.profile.name,
            ham=self.profile.ham,
            vegetation=self.profile.vegetation,
            radiation=self.profile.radiation,
            physics=physics or PhysicsConfig(urban=properties),
        )
        return builder.write_case(overwrite=overwrite)

    materialize = write_case

    @staticmethod
    def validate(case_path: str | Path) -> list[str]:
        errors = check_openfoam13_case(case_path)
        root = Path(case_path)
        for required in (root / "0", root / "constant", root / "system"):
            if not required.is_dir():
                errors.append(f"missing generated case directory: {required.relative_to(root)}")
        return errors


def materialize_all(output_root: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Generate all six native cases under ``output_root``."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    return [UrbanClimateCase.from_name(name).write_case(output_root / name, overwrite=overwrite) for name in PROFILES]
