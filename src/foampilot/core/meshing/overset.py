"""Small, solver-independent overset prototype used for Foundation 13 porting.

This module deliberately contains only deterministic geometry and interpolation
primitives. It is not an OpenFOAM runtime implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, isfinite
from typing import Iterable, Sequence

Point = tuple[float, ...]
Value = float | tuple[float, ...]


@dataclass(frozen=True)
class OversetZone:
    """Axis-aligned mesh-zone description with a consecutive integer id."""

    name: str
    zone_id: int
    lower: Point
    upper: Point

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("zone name must not be empty")
        if self.zone_id < 0:
            raise ValueError("zone_id must be non-negative")
        if len(self.lower) != len(self.upper) or not self.lower:
            raise ValueError("zone bounds must have the same non-zero dimension")
        if any(not isfinite(x) for x in (*self.lower, *self.upper)):
            raise ValueError("zone bounds must be finite")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("each lower bound must be smaller than its upper bound")

    def contains(self, point: Sequence[float]) -> bool:
        if len(point) != len(self.lower):
            raise ValueError("point dimension does not match zone dimension")
        return all(lo <= x <= hi for x, lo, hi in zip(point, self.lower, self.upper))


def validate_zones(zones: Iterable[OversetZone]) -> tuple[OversetZone, ...]:
    """Validate that zone ids are unique and consecutive starting at zero."""

    result = tuple(zones)
    ids = sorted(zone.zone_id for zone in result)
    if ids != list(range(len(ids))):
        raise ValueError("zone ids must be consecutive and start at zero")
    names = [zone.name for zone in result]
    if len(set(names)) != len(names):
        raise ValueError("zone names must be unique")
    dimensions = {len(zone.lower) for zone in result}
    if len(dimensions) > 1:
        raise ValueError("all zones must have the same dimension")
    return result


def build_zone_id(points: Iterable[Sequence[float]], zones: Iterable[OversetZone]) -> list[int]:
    """Assign the highest-priority containing zone id to every point.

    Zones are considered in descending zone id order, so the moving mesh can
    override the background mesh in their overlap.
    """

    checked = validate_zones(zones)
    ordered = tuple(sorted(checked, key=lambda zone: zone.zone_id, reverse=True))
    result: list[int] = []
    for point in points:
        containing = [zone.zone_id for zone in ordered if zone.contains(point)]
        if not containing:
            raise ValueError(f"point {tuple(point)!r} is outside all overset zones")
        result.append(containing[0])
    return result


def write_zone_id_field(
    case_path: str,
    points: Iterable[Sequence[float]],
    zones: Iterable[OversetZone],
    time_name: str = "0",
    field_name: str = "zoneID",
) -> str:
    """Write a Foundation-style scalar zone field and return its path."""

    if not field_name or any(char.isspace() for char in field_name):
        raise ValueError("field_name must be a non-empty token")
    zone_ids = build_zone_id(points, zones)
    if not zone_ids:
        raise ValueError("at least one point is required")
    from pathlib import Path

    output = Path(case_path) / time_name / field_name
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "\\n".join(str(zone_id) for zone_id in zone_ids)
    output.write_text(
        "FoamFile\\n"
        "{\\n"
        "    version 2.0;\\n"
        "    format ascii;\\n"
        "    class volScalarField;\\n"
        f"    location \\\"{time_name}\\\";\\n"
        f"    object {field_name};\\n"
        "}\\n"
        "dimensions      [0 0 0 0 0 0 0];\\n"
        f"internalField   nonuniform List<scalar>\\n{len(zone_ids)}\\n(\\n{body}\\n);\\n"
        "boundaryField\\n{ }\\n"
    )
    return str(output)


@dataclass(frozen=True)
class DonorStencil:
    """Interpolation stencil for one acceptor point."""

    donor_indices: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.donor_indices or len(self.donor_indices) != len(self.weights):
            raise ValueError("a stencil needs aligned, non-empty donors and weights")
        if any(index < 0 for index in self.donor_indices):
            raise ValueError("donor indices must be non-negative")
        if any(not isfinite(weight) or weight < 0.0 for weight in self.weights):
            raise ValueError("donor weights must be finite and non-negative")
        if abs(sum(self.weights) - 1.0) > 1e-12:
            raise ValueError("donor weights must sum to one")


def build_donor_stencil(
    target: Sequence[float],
    donor_points: Sequence[Sequence[float]],
    n_donors: int = 4,
    max_distance: float | None = None,
) -> DonorStencil:
    """Build an inverse-distance stencil for one acceptor."""

    if n_donors < 1:
        raise ValueError("n_donors must be positive")
    if not donor_points:
        raise ValueError("at least one donor point is required")
    if max_distance is not None and (not isfinite(max_distance) or max_distance <= 0.0):
        raise ValueError("max_distance must be finite and positive")
    dimension = len(target)
    if dimension == 0 or any(len(point) != dimension for point in donor_points):
        raise ValueError("all points must have the same non-zero dimension")
    ranked = sorted(
        ((index, dist(tuple(target), tuple(point))) for index, point in enumerate(donor_points)),
        key=lambda item: item[1],
    )
    if max_distance is not None:
        ranked = [item for item in ranked if item[1] <= max_distance]
    if len(ranked) < n_donors:
        raise ValueError("not enough donors within the requested distance")
    ranked = ranked[:n_donors]
    if ranked[0][1] == 0.0:
        return DonorStencil((ranked[0][0],), (1.0,))
    inverse = [1.0 / distance for _, distance in ranked]
    total = sum(inverse)
    if not isfinite(total) or total <= 0.0:
        raise ValueError("invalid donor weights")
    weights = tuple(value / total for value in inverse)
    return DonorStencil(
        tuple(index for index, _ in ranked),
        weights,
    )


def build_donor_stencils(
    target_points: Iterable[Sequence[float]],
    donor_points: Sequence[Sequence[float]],
    n_donors: int = 4,
    max_distance: float | None = None,
) -> tuple[DonorStencil, ...]:
    """Build donor stencils for all acceptor points."""

    return tuple(
        build_donor_stencil(target, donor_points, n_donors, max_distance)
        for target in target_points
    )


def write_donor_stencils(
    case_path: str,
    acceptor_points: Iterable[Sequence[float]],
    donor_points: Sequence[Sequence[float]],
    n_donors: int = 4,
    max_distance: float | None = None,
    time_name: str = "0",
) -> str:
    """Write a solver-neutral donor map for inspection and later C++ use."""

    stencils = build_donor_stencils(
        acceptor_points, donor_points, n_donors, max_distance
    )
    from pathlib import Path

    output = Path(case_path) / "constant" / "marineOversetStencils"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "FoamFile",
        "{",
        "    version 2.0;",
        "    format ascii;",
        "    class dictionary;",
        "    object marineOversetStencils;",
        "}",
        "",
        f"timeName {time_name};",
        f"nAcceptors {len(stencils)};",
        "acceptors",
        "(",
    ]
    for acceptor_index, stencil in enumerate(stencils):
        donors = " ".join(str(index) for index in stencil.donor_indices)
        weights = " ".join(f"{weight:.16g}" for weight in stencil.weights)
        lines.extend(
            [
                "    {",
                f"        index {acceptor_index};",
                f"        donorIndices ({donors});",
                f"        weights ({weights});",
                "    }",
            ]
        )
    lines.extend([");", ""])
    output.write_text("\n".join(lines))
    return str(output)


def write_marine_overset_constraint(
    case_path: str,
    fields: Sequence[str] = ("U", "p", "p_rgh", "alpha.water"),
    library: str = "libmarineOversetProbe.so",
) -> str:
    """Write a Foundation 13 ``system/fvConstraints`` overset entry."""

    if not fields or any(
        not field or any(char.isspace() for char in field) for field in fields
    ):
        raise ValueError("fields must contain non-empty OpenFOAM field names")
    if not library or any(char.isspace() for char in library):
        raise ValueError("library must be a non-empty token")

    from pathlib import Path

    output = Path(case_path) / "system" / "fvConstraints"
    output.parent.mkdir(parents=True, exist_ok=True)
    field_tokens = " ".join(fields)
    output.write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    version 2.0;",
                "    format ascii;",
                "    class dictionary;",
                "    object fvConstraints;",
                "}",
                "marineOverset",
                "{",
                "    type marineOversetConstraint;",
                f"    libs (\"{library}\");",
                f"    fields ({field_tokens});",
                "}",
                "",
            ]
        )
    )
    return str(output)


def write_intermesh_stencils(
    case_path: str,
    stencils: Sequence[DonorStencil],
    acceptor_indices: Sequence[int],
    donor_region: str = "background",
    acceptor_region: str = "hull",
) -> str:
    """Write a Foundation-readable cross-region stencil contract."""

    if len(stencils) != len(acceptor_indices) or not stencils:
        raise ValueError("stencils and acceptor_indices must be non-empty and aligned")
    if not donor_region or not acceptor_region:
        raise ValueError("region names must be non-empty")
    from pathlib import Path

    output = Path(case_path) / "constant" / "marineInterMeshStencils"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "FoamFile",
        "{",
        "    version 2.0;",
        "    format ascii;",
        "    class dictionary;",
        "    object marineInterMeshStencils;",
        "}",
        f"donorRegion {donor_region};",
        f"acceptorRegion {acceptor_region};",
        "acceptors",
        "(",
    ]
    for index, stencil in zip(acceptor_indices, stencils):
        donors = " ".join(str(value) for value in stencil.donor_indices)
        weights = " ".join(f"{value:.16g}" for value in stencil.weights)
        lines.extend(
            [
                "    {",
                f"        index {index};",
                f"        donorIndices ({donors});",
                f"        weights ({weights});",
                "    }",
            ]
        )
    lines.extend([ ");", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return str(output)


def inverse_distance_interpolate(
    target: Sequence[float],
    donor_points: Sequence[Sequence[float]],
    donor_values: Sequence[Value],
    n_donors: int = 4,
) -> Value:
    """Interpolate a scalar or vector value using inverse-distance weights."""

    if n_donors < 1:
        raise ValueError("n_donors must be positive")
    if len(donor_points) != len(donor_values) or not donor_points:
        raise ValueError("donor points and values must be non-empty and aligned")
    dimension = len(target)
    if dimension == 0 or any(len(point) != dimension for point in donor_points):
        raise ValueError("all points must have the same non-zero dimension")
    ranked = sorted(
        zip(donor_points, donor_values), key=lambda item: dist(tuple(target), tuple(item[0]))
    )[:n_donors]
    distances = [dist(tuple(target), tuple(point)) for point, _ in ranked]
    if distances[0] == 0.0:
        return ranked[0][1]
    weights = [1.0 / distance for distance in distances]
    weight_sum = sum(weights)
    if not isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("invalid interpolation weights")
    weights = [weight / weight_sum for weight in weights]
    first = ranked[0][1]
    if isinstance(first, tuple):
        size = len(first)
        if any(not isinstance(value, tuple) or len(value) != size for _, value in ranked):
            raise ValueError("donor values must have consistent scalar/vector type")
        return tuple(
            sum(weight * value[index] for weight, (_, value) in zip(weights, ranked))
            for index in range(size)
        )
    if any(isinstance(value, tuple) for _, value in ranked):
        raise ValueError("donor values must have consistent scalar/vector type")
    return sum(weight * float(value) for weight, (_, value) in zip(weights, ranked))
