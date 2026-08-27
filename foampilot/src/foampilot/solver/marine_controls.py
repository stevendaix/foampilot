"""Validated propulsion and rudder commands for marine cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PropellerCommand:
    """Propeller command expressed as RPM and optional thrust target."""

    rpm: float
    diameter: float
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    thrust_target: float | None = None

    def validate(self) -> None:
        if self.rpm < 0:
            raise ValueError("propeller rpm must be non-negative")
        if self.diameter <= 0:
            raise ValueError("propeller diameter must be strictly positive")
        if len(self.axis) != 3 or not any(abs(v) > 0 for v in self.axis):
            raise ValueError("propeller axis must be a non-zero 3-vector")
        if self.thrust_target is not None and self.thrust_target < 0:
            raise ValueError("thrust_target must be non-negative")


@dataclass(frozen=True)
class RudderCommand:
    """Rudder command expressed as an angle in degrees."""

    angle_deg: float
    max_angle_deg: float = 35.0
    rate_limit_deg_s: float | None = None

    def validate(self) -> None:
        if self.max_angle_deg <= 0:
            raise ValueError("max_angle_deg must be strictly positive")
        if abs(self.angle_deg) > self.max_angle_deg:
            raise ValueError("rudder angle exceeds max_angle_deg")
        if self.rate_limit_deg_s is not None and self.rate_limit_deg_s <= 0:
            raise ValueError("rate_limit_deg_s must be strictly positive")


def write_marine_controls(
    case_path: str | Path,
    *,
    propeller: PropellerCommand,
    rudder: RudderCommand,
) -> Path:
    """Write a stable, solver-neutral ``constant/marineControls`` dictionary."""
    propeller.validate()
    rudder.validate()
    root = Path(case_path)
    constant = root / "constant"
    constant.mkdir(parents=True, exist_ok=True)
    path = constant / "marineControls"
    thrust = "none" if propeller.thrust_target is None else str(propeller.thrust_target)
    rate = "none" if rudder.rate_limit_deg_s is None else str(rudder.rate_limit_deg_s)
    content = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object marineControls;
}}

propeller
{{
    rpm {propeller.rpm};
    diameter {propeller.diameter};
    axis ({' '.join(str(v) for v in propeller.axis)});
    thrustTarget {thrust};
}}

rudder
{{
    angleDeg {rudder.angle_deg};
    maxAngleDeg {rudder.max_angle_deg};
    rateLimitDegS {rate};
}}
"""
    path.write_text(content, encoding="utf-8")
    return path
