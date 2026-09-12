from foampilot.openfoam.solvers.configs.base_config import SolverConfig, config_from_flags
from foampilot.openfoam.solvers.configs.incompressible import IncompressibleConfig
from foampilot.openfoam.solvers.configs.compressible import CompressibleConfig
from foampilot.openfoam.solvers.configs.multiphase import VoFConfig
from foampilot.openfoam.solvers.configs.cht import CHTConfig
from foampilot.openfoam.solvers.configs.solid import SolidConfig

__all__ = [
    "SolverConfig",
    "config_from_flags",
    "IncompressibleConfig",
    "CompressibleConfig",
    "VoFConfig",
    "CHTConfig",
    "SolidConfig",
]
