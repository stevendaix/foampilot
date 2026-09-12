"""Solver registry for OpenFOAM solver metadata."""

from __future__ import annotations


SOLVER_MODULES = {
    "fluid": "fluid",
    "incompressibleFluid": "incompressibleFluid",
    "multicomponentFluid": "multicomponentFluid",
    "compressibleVoF": "compressibleVoF",
    "incompressibleVoF": "incompressibleVoF",
    "solidDisplacement": "solidDisplacement",
    "functions": "functions",
    "movingMesh": "movingMesh",
    "icoFoam": "icoFoam",
    "simpleFoam": "simpleFoam",
    "pimpleFoam": "pimpleFoam",
    "pimpleDyMFoam": "pimpleDyMFoam",
    "rhoCentralFoam": "rhoCentralFoam",
    "sonicFoam": "sonicFoam",
    "reactingFoam": "reactingFoam",
    "scalarTransportFoam": "scalarTransportFoam",
    "chtMultiRegionFoam": "chtMultiRegionFoam",
    "chtMultiRegionSimpleFoam": "chtMultiRegionSimpleFoam",
    "compressibleSinglePhasePorosityFoam": "compressibleSinglePhasePorosityFoam",
    "porousSimpleFoam": "porousSimpleFoam",
    "overInterDyMFoam": "overInterDyMFoam",
    "rhoSimpleFoam": "rhoSimpleFoam",
}

LEGACY_SOLVERS = {
    "overInterDyMFoam",
    "rhoSimpleFoam",
    "simpleFoam",
    "pimpleFoam",
    "marineFoam",
}


class SolverRegistry:
    """Registry for OpenFOAM solver metadata."""

    @staticmethod
    def get_module(solver_name: str) -> str:
        return SOLVER_MODULES.get(solver_name, solver_name)

    @staticmethod
    def is_legacy(solver_name: str) -> bool:
        return solver_name in LEGACY_SOLVERS

    @staticmethod
    def get_solver_module(solver_name: str) -> str | None:
        return SOLVER_MODULES.get(solver_name)
