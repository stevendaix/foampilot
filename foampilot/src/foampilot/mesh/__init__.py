"""Mesh utilities with optional backends loaded on demand."""
from __future__ import annotations

from .BlockMeshFile import BlockMesher
from .ops import (
    create_case_structure,
    restore_initial_fields,
    write_dynamic_mesh_dict,
    write_mesh_motion,
    write_rotating_zone,
)
from .marine_motion import FOUNDATION13_JOINTS, write_six_dof_dynamic_mesh_dict
from .marine_mrf import MarineMRFZone, write_marine_mrf
from .marine_overset import (
    DonorStencil,
    OversetZone,
    build_donor_stencil,
    build_donor_stencils,
    build_zone_id,
    inverse_distance_interpolate,
    validate_zones,
    write_donor_stencils,
    write_intermesh_stencils,
    write_marine_overset_constraint,
    write_zone_id_field,
)

_LAZY_MODULES = {
    "GmshMesher": ("foampilot.mesh.gmsh_mesher", "GmshMesher"),
    "SnappyMesher": ("foampilot.mesh.snappymesh", "SnappyMesher"),
    "DirectOpenFOAMExporter": ("foampilot.mesh.direct_openfoam_exporter", "DirectOpenFOAMExporter"),
    "GmshQualityAnalyzer": ("foampilot.mesh.quality", "GmshQualityAnalyzer"),
    "QualityThresholds": ("foampilot.mesh.quality", "QualityThresholds"),
    "QualityReport": ("foampilot.mesh.quality", "QualityReport"),
    "ElementQuality": ("foampilot.mesh.quality", "ElementQuality"),
    "CheckMeshParser": ("foampilot.mesh.quality", "CheckMeshParser"),
    "QualityGate": ("foampilot.mesh.quality", "QualityGate"),
    "OpenFOAMQualityAnalyzer": ("foampilot.mesh.quality", "OpenFOAMQualityAnalyzer"),
    "AdaptiveMeshImprover": ("foampilot.mesh.adaptation", "AdaptiveMeshImprover"),
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        import importlib
        module_name, attribute = _LAZY_MODULES[name]
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value
        return value
    if name in {
        "create_rectangle_geo", "create_channel_with_obstacle_geo", "create_step_geo",
        "create_cylinder_in_channel_geo", "create_car_channel_geo", "create_thermal_room_geo",
        "create_buildings_geo", "create_motorcycle_geo",
    }:
        import importlib
        value = getattr(importlib.import_module("foampilot.mesh.geo_generator"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BlockMesher", "GmshMesher", "SnappyMesher", "DirectOpenFOAMExporter",
    "GmshQualityAnalyzer", "QualityThresholds", "QualityReport", "ElementQuality",
    "CheckMeshParser", "QualityGate", "OpenFOAMQualityAnalyzer", "AdaptiveMeshImprover",
    "create_case_structure", "restore_initial_fields", "write_dynamic_mesh_dict",
    "write_mesh_motion", "write_rotating_zone", "write_six_dof_dynamic_mesh_dict",
    "FOUNDATION13_JOINTS", "MarineMRFZone", "write_marine_mrf", "OversetZone",
    "DonorStencil", "validate_zones", "build_zone_id", "write_zone_id_field",
    "build_donor_stencil", "build_donor_stencils", "write_donor_stencils",
    "write_marine_overset_constraint", "write_intermesh_stencils", "inverse_distance_interpolate",
]
