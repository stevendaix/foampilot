"""FoamPilot materialization helpers for the urbanMicroclimateFoam cases.

The original tutorials remain versioned as controlled templates.  A case is
never run directly from those templates: :class:`UrbanClimateCase` creates a
new case directory, copies the selected multi-region template, and asks the
OpenFOAM 13 physics layer to write provenance and optional support dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
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
    "streetCanyon_CFD": UrbanClimateProfile(
        "streetCanyon_CFD", "Single-region street canyon CFD", ("air",)
    ),
    "streetCanyon_CFDHAM": UrbanClimateProfile(
        "streetCanyon_CFDHAM", "Street canyon with heat-air-moisture coupling", ("air", "ground", "buildings", "street", "leeward", "windward"), ham=True, radiation=True
    ),
    "streetCanyon_CFDHAM_grass": UrbanClimateProfile(
        "streetCanyon_CFDHAM_grass", "Street canyon HAM case with grass", ("air", "ground", "buildings", "street", "leeward", "windward"), ham=True, radiation=True, vegetation=True
    ),
    "streetCanyon_CFDHAM_veg": UrbanClimateProfile(
        "streetCanyon_CFDHAM_veg", "Street canyon HAM case with vegetation", ("air", "vegetation", "ground", "buildings", "street", "leeward", "windward"), ham=True, radiation=True, vegetation=True
    ),
    "windAroundBuildings_CFDHAM": UrbanClimateProfile(
        "windAroundBuildings_CFDHAM", "Wind around buildings with HAM", ("air", "ground", "buildings"), ham=True, radiation=True
    ),
    "windAroundBuildings_CFDHAM_veg": UrbanClimateProfile(
        "windAroundBuildings_CFDHAM_veg", "Wind around buildings with HAM and vegetation", ("air", "vegetation", "ground", "buildings"), ham=True, radiation=True, vegetation=True
    ),
}


class UrbanClimateCase:
    """Create and validate a named urbanMicroclimateFoam case."""

    def __init__(self, profile: UrbanClimateProfile, template_root: str | Path):
        self.profile = profile
        self.template_root = Path(template_root)

    @classmethod
    def from_name(cls, name: str, template_root: str | Path) -> "UrbanClimateCase":
        try:
            profile = PROFILES[name]
        except KeyError as exc:
            raise ValueError(f"Unknown urbanclimate profile: {name}") from exc
        return cls(profile, template_root)

    def materialize(
        self,
        destination: str | Path,
        *,
        overwrite: bool = False,
        physics: PhysicsConfig | None = None,
        urban_properties: Mapping[str, Any] | None = None,
    ) -> Path:
        """Materialize a case and write FoamPilot-owned support files.

        The template is copied into a new destination. Existing destinations
        are protected unless ``overwrite=True``. The generated manifest records
        the selected profile, source template and OpenFOAM version.
        """
        source = self.template_root / self.profile.name
        destination = Path(destination)
        if not source.is_dir():
            raise FileNotFoundError(f"Urban climate template not found: {source}")
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing case: {destination}")
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

        profile_properties = {
            "profile": self.profile.name,
            "ham": self.profile.ham,
            "radiation": self.profile.radiation,
            "vegetation": self.profile.vegetation,
        }
        if physics is None:
            cfg = PhysicsConfig(urban=dict(urban_properties or profile_properties))
        else:
            cfg = physics
            if urban_properties:
                cfg.urban.update(urban_properties)
            for key, value in profile_properties.items():
                cfg.urban.setdefault(key, value)
        cfg.write_support_files(destination)
        manifest = destination / "foampilotUrbanClimate.json"
        manifest.write_text(json.dumps({
            "profile": self.profile.name,
            "description": self.profile.description,
            "regions": self.profile.regions,
            "vegetation": self.profile.vegetation,
            "ham": self.profile.ham,
            "radiation": self.profile.radiation,
            "template": str(source),
            "openfoam": {"vendor": cfg.openfoam_vendor, "version": cfg.openfoam_version},
        }, indent=2) + "\n", encoding="utf-8")
        return destination

    def write_case(
        self,
        destination: str | Path,
        *,
        overwrite: bool = False,
        physics: PhysicsConfig | None = None,
        urban_properties: Mapping[str, Any] | None = None,
    ) -> Path:
        """FoamPilot-compatible alias used by example ``run.py`` entry points."""
        return self.materialize(
            destination,
            overwrite=overwrite,
            physics=physics,
            urban_properties=urban_properties,
        )

    @staticmethod
    def validate(case_path: str | Path) -> list[str]:
        """Run Foampilot preflight checks and verify all expected case roots."""
        errors = check_openfoam13_case(case_path)
        root = Path(case_path)
        for path in (root / "0", root / "constant", root / "system"):
            if not path.is_dir():
                errors.append(f"missing generated case directory: {path.relative_to(root)}")
        return errors


def materialize_all(
    template_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Materialize all six profiles under ``output_root``."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in PROFILES:
        paths.append(UrbanClimateCase.from_name(name, template_root).materialize(
            output_root / name, overwrite=overwrite
        ))
    return paths
