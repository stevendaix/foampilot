"""Foundation 13 actuation-disk source generation for propellers."""

from __future__ import annotations

from math import pi
from pathlib import Path

from dataclasses import dataclass


@dataclass(frozen=True)
class ActuationDiskSource:
    """Configuration for the native Foundation 13 ``actuationDisk`` fvModel."""

    cell_zone: str
    disk_dir: tuple[float, float, float]
    cp: float
    ct: float
    disk_area: float
    upstream_point: tuple[float, float, float]
    phase_name: str | None = None
    velocity_field: str = "U"

    def validate(self) -> None:
        if not self.cell_zone.strip():
            raise ValueError("cell_zone must not be empty")
        if len(self.disk_dir) != 3 or not any(abs(v) > 0 for v in self.disk_dir):
            raise ValueError("disk_dir must be a non-zero 3-vector")
        if self.cp < 0 or self.ct <= 0:
            raise ValueError("cp must be non-negative and ct strictly positive")
        if self.disk_area <= 0:
            raise ValueError("disk_area must be strictly positive")
        if len(self.upstream_point) != 3:
            raise ValueError("upstream_point must be a 3-vector")
        if not self.velocity_field.strip():
            raise ValueError("velocity_field must not be empty")


def actuation_disk_from_propeller(
    *,
    cell_zone: str,
    diameter: float,
    disk_dir: tuple[float, float, float],
    cp: float,
    ct: float,
    upstream_point: tuple[float, float, float],
    phase_name: str | None = None,
) -> ActuationDiskSource:
    """Build a native actuation-disk source from propeller geometry."""
    if diameter <= 0:
        raise ValueError("diameter must be strictly positive")
    return ActuationDiskSource(
        cell_zone=cell_zone,
        disk_dir=disk_dir,
        cp=cp,
        ct=ct,
        disk_area=pi * diameter**2 / 4.0,
        upstream_point=upstream_point,
        phase_name=phase_name,
    )


def write_actuation_disk(case_path: str | Path, source: ActuationDiskSource) -> Path:
    """Write a Foundation 13 ``constant/fvModels`` actuation disk."""
    source.validate()
    root = Path(case_path)
    constant = root / "constant"
    constant.mkdir(parents=True, exist_ok=True)
    path = constant / "fvModels"
    phase_line = f"    phase      {source.phase_name};\n" if source.phase_name else ""
    content = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object fvModels;
}}

propellerActuationDisk
{{
    type            actuationDisk;
    cellZone        {source.cell_zone};
{phase_line}    U               {source.velocity_field};
    diskDir         ({' '.join(str(v) for v in source.disk_dir)});
    Cp              {source.cp};
    Ct              {source.ct};
    diskArea        {source.disk_area};
    upstreamPoint   ({' '.join(str(v) for v in source.upstream_point)});
}}
"""
    path.write_text(content, encoding="utf-8")
    return path
