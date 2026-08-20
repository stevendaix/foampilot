"""Lazy public exports for meshing backends."""

_LAZY = {
    "BlockMesher": ("foampilot.mesh.BlockMeshFile", "BlockMesher"),
    "GmshMesher": ("foampilot.mesh.gmsh_mesher", "GmshMesher"),
    "SnappyMesher": ("foampilot.mesh.snappymesh", "SnappyMesher"),
    "DirectOpenFOAMExporter": ("foampilot.mesh.direct_openfoam_exporter", "DirectOpenFOAMExporter"),
    "AdaptiveMeshImprover": ("foampilot.mesh.adaptation", "AdaptiveMeshImprover"),
    "GmshQualityAnalyzer": ("foampilot.mesh.quality", "GmshQualityAnalyzer"),
    "QualityThresholds": ("foampilot.mesh.quality", "QualityThresholds"),
    "QualityReport": ("foampilot.mesh.quality", "QualityReport"),
    "ElementQuality": ("foampilot.mesh.quality", "ElementQuality"),
    "CheckMeshParser": ("foampilot.mesh.quality", "CheckMeshParser"),
    "QualityGate": ("foampilot.mesh.quality", "QualityGate"),
    "OpenFOAMQualityAnalyzer": ("foampilot.mesh.quality", "OpenFOAMQualityAnalyzer"),
}

_GEO = {
    name: ("foampilot.mesh.geo_generator", name)
    for name in (
        "create_rectangle_geo", "create_channel_with_obstacle_geo", "create_step_geo",
        "create_cylinder_in_channel_geo", "create_car_channel_geo", "create_thermal_room_geo",
        "create_buildings_geo", "create_motorcycle_geo",
    )
}
_LAZY.update(_GEO)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    import importlib
    module_name, attribute = _LAZY[name]
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY)
