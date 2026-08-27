"""MRF rotor-zone configuration for marine propeller cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from foampilot.mesh.ops import write_rotating_zone


@dataclass(frozen=True)
class MarineMRFZone:
    """Physical description of a steady rotating propeller zone."""

    cell_zone: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    omega: float
    non_rotating_patches: tuple[str, ...]

    def validate(self) -> None:
        if not self.cell_zone.strip():
            raise ValueError("cell_zone must not be empty")
        if len(self.origin) != 3 or len(self.axis) != 3:
            raise ValueError("origin and axis must be 3-vectors")
        if not any(abs(value) > 0 for value in self.axis):
            raise ValueError("axis must be a non-zero 3-vector")
        if self.omega == 0:
            raise ValueError("omega must be non-zero for an active MRF zone")
        if not self.non_rotating_patches:
            raise ValueError("non_rotating_patches must not be empty")


def write_marine_mrf(case_path: str | Path, zone: MarineMRFZone) -> Path:
    """Validate and write an OpenFOAM ``constant/MRFProperties`` file."""
    zone.validate()
    return write_rotating_zone(
        case_path,
        cell_zone=zone.cell_zone,
        origin=zone.origin,
        axis=zone.axis,
        omega=zone.omega,
        non_rotating_patches=zone.non_rotating_patches,
    )
