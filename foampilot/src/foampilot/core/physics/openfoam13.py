"""OpenFOAM 13 physics integrations used by FoamPilot.

The module deliberately keeps third-party C++ libraries optional.  FoamPilot
writes only portable OpenFOAM dictionaries by default and refuses to enable a
library whose declared vendor/version is not compatible with the case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import re


SUPPORTED_MODULES = {
    "boundaryConditions": "optional runtime boundary-condition library",
    "MachineLearningTurbulenceModels": "optional turbulence library",
    "urbanMicroclimateFoam-tutorials": "case templates and urban defaults",
    "adaptive-mesh-refinement": "dynamic refinement workflow",
    "PythonFOAM": "optional Python/C++ co-simulation examples",
}


@dataclass(frozen=True)
class ExternalModule:
    name: str
    repository: str
    revision: str
    license: str
    openfoam_families: tuple[str, ...]
    enabled: bool = False


PORTED_COMPONENTS = {
    "ZYturbulentInlet": {
        "source": "boundaryConditions",
        "path": "third_party/openfoam13/ported/boundaryConditions/ZYturbulentInlet",
        "status": "compiled-foundation13",
    },
    "turbulentInletTable": {
        "source": "boundaryConditions",
        "path": "third_party/openfoam13/ported/boundaryConditions/turbulentInletTable",
        "status": "compiled-foundation13",
    },
    "calculateNut": {
        "source": "MachineLearningTurbulenceModels",
        "path": "third_party/openfoam13/ported/MachineLearningTurbulenceModels/calculateNut",
        "status": "compiled-foundation13",
    },
    "calculateGamma": {
        "source": "MachineLearningTurbulenceModels",
        "path": "third_party/openfoam13/ported/MachineLearningTurbulenceModels/calculateGamma",
        "status": "compiled-foundation13",
    },
    "calculateRFV": {
        "source": "MachineLearningTurbulenceModels",
        "path": "third_party/openfoam13/ported/MachineLearningTurbulenceModels/calculateRFV",
        "status": "compiled-foundation13",
    },
    "calculateRFVperp": {
        "source": "MachineLearningTurbulenceModels",
        "path": "third_party/openfoam13/ported/MachineLearningTurbulenceModels/calculateRFVperp",
        "status": "compiled-foundation13",
    },
    "calculateRperp": {
        "source": "MachineLearningTurbulenceModels",
        "path": "third_party/openfoam13/ported/MachineLearningTurbulenceModels/calculateRperp",
        "status": "compiled-foundation13",
    },
}


DEFAULT_MODULES = (
    ExternalModule("boundaryConditions", "ZhangYanTJU/boundaryConditions", "dd3c819e4a715ee64f1782b71b56889ac487f352", "GPL-3.0", ("ESI", "Foundation")),
    ExternalModule("MachineLearningTurbulenceModels", "mthsmcd/MachineLearningTurbulenceModels", "955df440c5bd38c5cbded3f5c889a655ac57750a", "AGPL-3.0", ("ESI",)),
    ExternalModule("urbanMicroclimateFoam-tutorials", "OpenFOAM-BuildingPhysics/urbanMicroclimateFoam-tutorials", "1e6d81393103d2cf3d0b5658ca484d986906a591", "GPL-3.0", ("Foundation", "ESI")),
    ExternalModule("adaptive-mesh-refinement", "airshaper/adaptive-mesh-refinement", "f24babf10b4c98b90fae14b8c6e340b7ca48fa84", "GPL-3.0", ("ESI",)),
    ExternalModule("PythonFOAM", "argonne-lcf/PythonFOAM", "6fde8698071b56f913c5e92c647f22ab38b26466", "Argonne", ("Foundation", "ESI")),
)


@dataclass
class PhysicsConfig:
    """Declarative configuration for optional OpenFOAM 13 capabilities."""

    openfoam_vendor: str = "Foundation"
    openfoam_version: int = 13
    modules: tuple[ExternalModule, ...] = DEFAULT_MODULES
    boundary_conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    turbulence: dict[str, Any] = field(default_factory=dict)
    urban: dict[str, Any] = field(default_factory=dict)
    adaptive_mesh: dict[str, Any] = field(default_factory=dict)
    pythonfoam: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.openfoam_version != 13:
            errors.append("OpenFOAM 13 is required by this integration layer")
        if self.openfoam_vendor not in {"Foundation", "ESI"}:
            errors.append("openfoam_vendor must be 'Foundation' or 'ESI'")
        enabled = {m.name: m for m in self.modules if m.enabled}
        if self.turbulence and "MachineLearningTurbulenceModels" not in enabled:
            errors.append("ML turbulence requires the MachineLearningTurbulenceModels module enabled")
        if self.openfoam_vendor not in {m for mod in enabled.values() for m in mod.openfoam_families} and enabled:
            errors.append("enabled modules contain no library compatible with the selected OpenFOAM vendor")
        for patch, spec in self.boundary_conditions.items():
            if not patch or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", patch):
                errors.append(f"invalid boundary patch name: {patch!r}")
            if "type" not in spec:
                errors.append(f"boundary condition for {patch!r} must define type")
        if self.adaptive_mesh:
            field_name = self.adaptive_mesh.get("sourceField", "curl(U)")
            if field_name not in {"curl(U)", "grad(p)", "grad(T)"} and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(field_name)):
                errors.append("adaptive_mesh.field must be a safe field expression")
            if float(self.adaptive_mesh.get("lowerRefinementLevel", 0)) < 0:
                errors.append("adaptive_mesh.lowerRefinementLevel must be non-negative")
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("Invalid OpenFOAM 13 physics configuration: " + "; ".join(errors))

    def manifest(self) -> dict[str, Any]:
        return {
            "openfoam": {"vendor": self.openfoam_vendor, "version": self.openfoam_version},
            "modules": [m.__dict__ for m in self.modules],
            "portedComponents": PORTED_COMPONENTS,
            "capabilities": {
                "boundaryConditions": bool(self.boundary_conditions),
                "machineLearningTurbulence": bool(self.turbulence),
                "urbanMicroclimate": bool(self.urban),
                "adaptiveMeshRefinement": bool(self.adaptive_mesh),
                "pythonFOAM": bool(self.pythonfoam),
            },
        }

    def write_support_files(self, case_path: str | Path) -> list[Path]:
        """Write portable support dictionaries; never overwrite user fields."""
        self.require_valid()
        root = Path(case_path)
        system = root / "system"
        constant = root / "constant"
        system.mkdir(parents=True, exist_ok=True)
        constant.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        manifest = root / "foampilotPhysics.json"
        manifest.write_text(json.dumps(self.manifest(), indent=2) + "\n", encoding="utf-8")
        written.append(manifest)
        if self.adaptive_mesh:
            p = system / "dynamicMeshDict"
            if not p.exists():
                p.write_text(_dynamic_mesh_dict(self.adaptive_mesh), encoding="utf-8")
                written.append(p)
        if self.boundary_conditions:
            p = constant / "foampilotBoundaryConditions.json"
            if not p.exists():
                p.write_text(json.dumps(self.boundary_conditions, indent=2) + "\n", encoding="utf-8")
                written.append(p)
        if self.urban:
            p = constant / "foampilotUrbanProperties"
            if not p.exists():
                p.write_text(_foam_dict("foampilotUrbanProperties", self.urban), encoding="utf-8")
                written.append(p)
        if self.turbulence:
            p = constant / "foampilotMLTurbulenceProperties"
            if not p.exists():
                p.write_text(_foam_dict("foampilotMLTurbulenceProperties", self.turbulence), encoding="utf-8")
                written.append(p)
        if self.pythonfoam:
            p = system / "foampilotPythonFOAMProperties"
            if not p.exists():
                p.write_text(_foam_dict("foampilotPythonFOAMProperties", self.pythonfoam), encoding="utf-8")
                written.append(p)
        return written


def _safe_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.15g}"
    if isinstance(value, (tuple, list)):
        return "(" + " ".join(_safe_scalar(v) for v in value) + ")"
    text = str(value)
    if "\n" in text or "{" in text or "}" in text:
        raise ValueError("multiline or block values are not accepted in portable physics dictionaries")
    return text


def _foam_dict(object_name: str, values: Mapping[str, Any]) -> str:
    lines = ["FoamFile", "{", "    version 2.0;", "    format ascii;", "    class dictionary;", f"    object {object_name};", "}", ""]
    for key, value in values.items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
            raise ValueError(f"unsafe OpenFOAM dictionary key: {key!r}")
        lines.append(f"{key} {_safe_scalar(value)};")
    return "\n".join(lines) + "\n"


def _dynamic_mesh_dict(cfg: Mapping[str, Any]) -> str:
    field = str(cfg.get("refinementField", "refVal"))
    max_level = int(cfg.get("maxRefinementLevel", 2))
    lower = float(cfg.get("lowerRefinementLevel", 0.1))
    upper = float(cfg.get("upperRefinementLevel", 1.0))
    if max_level < 0 or lower < 0 or upper < lower:
        raise ValueError("invalid adaptive mesh refinement thresholds")
    return """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object dynamicMeshDict;
}

dynamicFvMesh dynamicRefineFvMesh;

refineInterval 1;
field %s;
lowerRefinementLevel %.15g;
upperRefinementLevel %.15g;
unrefineLevel %.15g;
nBufferLayers 1;
maxRefinement %d;
maxCells 2000000;
correctFluxes (phi none);
dumpLevel true;
""" % (field, lower, upper, lower, max_level)


def check_openfoam13_case(case_path: str | Path) -> list[str]:
    """Static preflight checks for generated cases, including explicit ``nu``."""
    root = Path(case_path)
    errors: list[str] = []
    for required in ("system/controlDict", "constant", "0"):
        if not (root / required).exists():
            errors.append(f"missing required case path: {required}")
    transport = root / "constant/transportProperties"
    if transport.exists() and not re.search(r"\bnu\s+[^;]+;", transport.read_text(encoding="utf-8", errors="replace")):
        errors.append("constant/transportProperties must explicitly define nu")
    if (root / "system/dynamicMeshDict").exists():
        text = (root / "system/dynamicMeshDict").read_text(encoding="utf-8")
        for key in ("dynamicFvMesh", "refineInterval", "field", "maxRefinement"):
            if not re.search(rf"\b{key}\s+", text):
                errors.append(f"dynamicMeshDict missing {key}")
    return errors


def module_catalog() -> tuple[ExternalModule, ...]:
    """Return the pinned external-module catalog used for reproducible manifests."""
    return DEFAULT_MODULES
