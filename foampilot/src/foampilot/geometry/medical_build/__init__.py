"""Validated, serializable medical geometry analysis and reconstruction contracts."""
from .analysis_data import BranchRecord, GeometryAnalysisData, SectionRecord
from .global_blockmesh import BlockRecord, GlobalBlockMesh
from .models import BoundaryCondition, MedicalBuildConfig, ReconstructionSpec
from .reconstruction import Build123dReconstruction, SectionLoftInput, normalize_sections
from .vascular_graph import GraphValidation, VascularGraph, build_vascular_graph
from .deformation import LocalDeformationSpec, apply_local_deformation, deformation_report
from .snappy_export import MedicalSnappyExporter, SnappyExportConfig

__all__ = [
    "BoundaryCondition", "MedicalBuildConfig", "ReconstructionSpec",
    "SectionRecord", "BranchRecord", "GeometryAnalysisData",
    "BlockRecord", "GlobalBlockMesh",
    "Build123dReconstruction", "SectionLoftInput", "normalize_sections",
    "GraphValidation", "VascularGraph", "build_vascular_graph",
    "LocalDeformationSpec", "apply_local_deformation", "deformation_report",
    "MedicalSnappyExporter", "SnappyExportConfig",
]
