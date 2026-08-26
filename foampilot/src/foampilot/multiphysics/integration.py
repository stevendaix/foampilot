"""OpenFOAM 13 multiphysics integration profiles for Foampilot.

The profiles deliberately keep external research codes out of the Python package.
Foampilot generates a reproducible manifest and case dictionary; compilation is
performed explicitly against a user-selected checkout and never silently guessed.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class PhysicsModule:
    """Metadata and integration contract for one external physics module."""

    name: str
    source_url: str
    source_ref: str
    upstream_openfoam: str
    role: str
    build_command: Tuple[str, ...]
    required_fields: Tuple[str, ...]
    compatible_with: Tuple[str, ...]
    notes: str


MODULES: Mapping[str, PhysicsModule] = {
    "sedifoam": PhysicsModule(
        name="sediFoam",
        source_url="https://github.com/xiaoh/sediFoam.git",
        source_ref="master",
        upstream_openfoam="legacy OpenFOAM releases; not OF13",
        role="Eulerian–DEM sediment transport through LAMMPS",
        build_command=("./Allwmake.sh",),
        required_fields=("U", "p", "alpha", "phi"),
        compatible_with=("libacoustics",),
        notes="Requires a separately installed LAMMPS interface; porting is source/API work, not a Python wrapper.",
    ),
    "openhfdib_dem": PhysicsModule(
        name="openHFDIB-DEM",
        source_url="https://github.com/techMathGroup/openHFDIB-DEM.git",
        source_ref="master",
        upstream_openfoam="OpenFOAM v8",
        role="immersed-boundary CFD–DEM for arbitrary particle shapes",
        build_command=("./compileAll.sh",),
        required_fields=("U", "p", "voidfraction", "particleVelocity"),
        compatible_with=("libacoustics",),
        notes="Foundation v13 API port required; both DEM backends cannot be active in one case.",
    ),
    "libacoustics": PhysicsModule(
        name="libAcoustics",
        source_url="https://github.com/unicfdlab/libAcoustics.git",
        source_ref="v2512",
        upstream_openfoam="OpenFOAM+ v2512; no Foundation OF13 branch upstream",
        role="acoustic source extraction and FW-H post-processing",
        build_command=("wmake", "libso"),
        required_fields=("U", "p", "phi"),
        compatible_with=("sedifoam", "openhfdib_dem"),
        notes="The v2512 branch is a reference API only; Foundation OF13 compatibility must be maintained in an adapter.",
    ),
}


class MultiphysicsConfigurationError(ValueError):
    """Raised when a requested multiphysics combination is unsafe or incomplete."""


@dataclass(frozen=True)
class MultiphysicsConfiguration:
    """Validated Foampilot contract for an OpenFOAM 13 case."""

    modules: Tuple[str, ...]
    openfoam_version: str = "13"
    case_name: str = "foampilotMultiphysics"

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(item.lower() for item in self.modules))
        object.__setattr__(self, "modules", normalized)
        self.validate()

    def validate(self) -> None:
        if self.openfoam_version != "13":
            raise MultiphysicsConfigurationError("Cette intégration cible exclusivement OpenFOAM Foundation 13.")
        if not self.modules:
            raise MultiphysicsConfigurationError("Au moins un module physique doit être sélectionné.")
        unknown = sorted(set(self.modules) - set(MODULES))
        if unknown:
            raise MultiphysicsConfigurationError(f"Modules inconnus: {', '.join(unknown)}")
        dem = {"sedifoam", "openhfdib_dem"}.intersection(self.modules)
        if len(dem) > 1:
            raise MultiphysicsConfigurationError(
                "sediFoam et openHFDIB-DEM sont deux backends DEM alternatifs; "
                "sélectionnez-en un seul par cas."
            )
        for module in self.modules:
            profile = MODULES[module]
            incompatible = set(self.modules) - {module} - set(profile.compatible_with)
            if incompatible:
                raise MultiphysicsConfigurationError(
                    f"{module} n'est pas compatible avec: {', '.join(sorted(incompatible))}"
                )

    @property
    def required_fields(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(field for name in self.modules for field in MODULES[name].required_fields))

    def manifest(self) -> Dict[str, object]:
        return {
            "format": "foampilot-multiphysics-v1",
            "openfoam": {"distribution": "Foundation", "version": self.openfoam_version},
            "case": self.case_name,
            "modules": [asdict(MODULES[name]) for name in self.modules],
            "requiredFields": list(self.required_fields),
            "portingPolicy": "external-source-adapter; no silent fallback to another OpenFOAM family",
        }

    def openfoam_dictionary(self) -> str:
        lines = [
            "FoamFile",
            "{",
            "    version 2.0;",
            "    format ascii;",
            "    class dictionary;",
            "    object foampilotMultiphysics;",
            "}",
            "",
            "openfoamDistribution Foundation;",
            "openfoamVersion 13;",
            f"caseName {self.case_name};",
            "modules (" + " ".join(self.modules) + ");",
            "requiredFields (" + " ".join(self.required_fields) + ");",
            "",
            "// Generated by Foampilot; external module source remains explicit.",
        ]
        return "\n".join(lines) + "\n"

    def write_case_assets(self, case_path: os.PathLike[str] | str) -> Tuple[Path, Path]:
        """Write the JSON audit manifest and OpenFOAM dictionary for a case."""
        root = Path(case_path)
        system = root / "system"
        system.mkdir(parents=True, exist_ok=True)
        manifest_path = system / "foampilotMultiphysics.json"
        dictionary_path = system / "foampilotMultiphysics"
        manifest_path.write_text(json.dumps(self.manifest(), indent=2) + "\n", encoding="utf-8")
        dictionary_path.write_text(self.openfoam_dictionary(), encoding="utf-8")
        return manifest_path, dictionary_path


def check_openfoam13() -> str:
    """Return the sourced OpenFOAM version, failing clearly when unavailable."""
    env = os.environ.copy()
    bashrc = Path("/opt/openfoam13/etc/bashrc")
    if bashrc.exists():
        command = f". {bashrc} >/dev/null 2>&1 && foamVersion 2>&1"
        result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, env=env, check=False)
    else:
        result = subprocess.run(["foamVersion"], text=True, capture_output=True, env=env, check=False)
    version = result.stdout.strip()
    if result.returncode != 0 or version != "OpenFOAM-13":
        raise RuntimeError(f"OpenFOAM 13 indisponible (sortie={version!r}, code={result.returncode}).")
    return version


def build_plan(config: MultiphysicsConfiguration, source_root: os.PathLike[str] | str) -> List[Dict[str, object]]:
    """Create an auditable, non-executing build plan for selected modules."""
    root = Path(source_root)
    return [
        {
            "module": name,
            "source": str(root / name),
            "sourceRef": MODULES[name].source_ref,
            "command": list(MODULES[name].build_command),
            "openfoam": "13",
            "execute": False,
        }
        for name in config.modules
    ]
