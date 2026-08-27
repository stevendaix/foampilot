"""Native fluid–structure interaction helpers for OpenFOAM.

The native workflow targets rigid-body FSI: OpenFOAM computes the fluid forces,
updates a six-degree-of-freedom body, and moves the surrounding mesh. No
external coupling executable is required. Deformable solid mechanics remains a
separate capability and is deliberately not hidden behind this API.
"""

from .native import (
    FSIConfigurationError,
    NativeRigidFSI,
    RigidBody,
    write_native_rigid_fsi,
)

__all__ = [
    "FSIConfigurationError",
    "NativeRigidFSI",
    "RigidBody",
    "write_native_rigid_fsi",
]
