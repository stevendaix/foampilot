"""Core geometry capabilities.

Generic capabilities for CAD, surfaces, topology, and VMTK processing.
"""
from foampilot.core.geometry.cad import BSplineFitter, OCCBuilder
from foampilot.core.geometry.topology import (
    BoundaryRole,
    OpenProfile,
    SurfaceTopologyAnalyzer,
    OpenProfileClassifier,
    TopologyCenterlineExtractor,
    Section,
    TopologySectionExtractor,
)
from foampilot.core.geometry.topology.vmtk import (
    vmtkBaseScript,
    vmtkCenterlines,
    vmtkCenterlineSections,
    vmtkBranchSections,
    vmtkDistanceToCenterlines,
    vmtkMeshGenerator,
    vmtkMeshQuality,
    vmtkMeshWriter,
    vmtkSurfaceReader,
    vmtkSurfaceWriter,
    vmtkSurfaceToNumpy,
    vmtkSurfaceRemesher,
)
from foampilot.core.geometry.surfaces import (
    SurfaceReader,
    SurfaceWriter,
    SurfaceMerger,
    SurfaceSplitter,
    SurfaceCleaner,
    MeshRepairer,
    remove_duplicates,
    fix_normals,
    NormalCalculator,
    fix_face_orientations,
)

__all__ = [
    "BSplineFitter",
    "OCCBuilder",
    "BoundaryRole",
    "OpenProfile",
    "SurfaceTopologyAnalyzer",
    "OpenProfileClassifier",
    "TopologyCenterlineExtractor",
    "Section",
    "TopologySectionExtractor",
    "vmtkBaseScript",
    "vmtkCenterlines",
    "vmtkCenterlineSections",
    "vmtkBranchSections",
    "vmtkDistanceToCenterlines",
    "vmtkMeshGenerator",
    "vmtkMeshQuality",
    "vmtkMeshWriter",
    "vmtkSurfaceReader",
    "vmtkSurfaceWriter",
    "vmtkSurfaceToNumpy",
    "vmtkSurfaceRemesher",
    "SurfaceReader",
    "SurfaceWriter",
    "SurfaceMerger",
    "SurfaceSplitter",
    "SurfaceCleaner",
    "MeshRepairer",
    "remove_duplicates",
    "fix_normals",
    "NormalCalculator",
    "fix_face_orientations",
]