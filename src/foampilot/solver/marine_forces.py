"""Analytical marine force models used to build Foundation 13 fvModels."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path


@dataclass(frozen=True)
class PropellerForceModel:
    """Open-water propeller thrust and torque model.

    ``kt`` and ``kq`` are non-dimensional coefficients. ``rpm`` is converted
    to revolutions per second before evaluating the standard open-water laws.
    """

    rho: float
    diameter: float
    rpm: float
    kt: float
    kq: float
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def validate(self) -> None:
        if self.rho <= 0 or self.diameter <= 0:
            raise ValueError("rho and diameter must be strictly positive")
        if self.rpm < 0:
            raise ValueError("rpm must be non-negative")
        if self.kt < 0 or self.kq < 0:
            raise ValueError("kt and kq must be non-negative")
        if len(self.axis) != 3 or not any(abs(v) > 0 for v in self.axis):
            raise ValueError("axis must be a non-zero 3-vector")

    @property
    def revolutions_per_second(self) -> float:
        self.validate()
        return self.rpm / 60.0

    @property
    def thrust(self) -> float:
        n = self.revolutions_per_second
        return self.kt * self.rho * n**2 * self.diameter**4

    @property
    def torque(self) -> float:
        n = self.revolutions_per_second
        return self.kq * self.rho * n**2 * self.diameter**5


@dataclass(frozen=True)
class RudderForceModel:
    """Quasi-steady rudder side-force model."""

    rho: float
    area: float
    lift_coefficient: float
    inflow_speed: float
    angle_deg: float
    moment_arm: float

    def validate(self) -> None:
        if self.rho <= 0 or self.area <= 0:
            raise ValueError("rho and area must be strictly positive")
        if self.inflow_speed < 0:
            raise ValueError("inflow_speed must be non-negative")
        if self.moment_arm < 0:
            raise ValueError("moment_arm must be non-negative")

    @property
    def side_force(self) -> float:
        self.validate()
        return 0.5 * self.rho * self.inflow_speed**2 * self.area * self.lift_coefficient

    @property
    def yaw_moment(self) -> float:
        return self.side_force * self.moment_arm


def write_force_model(case_path: str | Path, *, propeller: PropellerForceModel, rudder: RudderForceModel) -> Path:
    """Write computed reference loads to ``constant/marineForces``.

    The file is intentionally a solver-neutral input for the next C++
    ``fvModel`` increment; it is not silently presented as an OpenFOAM force
    source until that runtime model is implemented and compiled.
    """
    propeller.validate()
    rudder.validate()
    root = Path(case_path)
    constant = root / "constant"
    constant.mkdir(parents=True, exist_ok=True)
    path = constant / "marineForces"
    axis = " ".join(str(v) for v in propeller.axis)
    content = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object marineForces;
}}

propeller
{{
    thrust {propeller.thrust};
    torque {propeller.torque};
    axis ({axis});
}}

rudder
{{
    sideForce {rudder.side_force};
    yawMoment {rudder.yaw_moment};
}}
"""
    path.write_text(content, encoding="utf-8")
    return path
