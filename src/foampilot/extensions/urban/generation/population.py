"""Population de régions compatible avec le contrat UrbGEN PopulateRegion.

Cette étape ne crée pas de bâtiments : elle produit les points candidats transmis
au générateur de masses. Les géométries sont Shapely et les angles sont en radians,
comme dans le composant Grasshopper original.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, floor, pi, sin, sqrt
from random import Random
from typing import Iterable, Optional

from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


@dataclass(frozen=True)
class PopulateRegionConfig:
    """Paramètres de population correspondant aux entrées du composant original."""

    count: int
    mode: int = 0
    jitter: float = 0.0
    angle: float = 0.0
    seed: int = 0
    min_dist: Optional[float] = None
    max_attempts_factor: int = 500


@dataclass(frozen=True)
class PopulateRegionResult:
    """Résultat déterministe de la population d’une région."""

    points: tuple[Point, ...]
    region: Polygon
    usable_area: float
    spacing: Optional[float]
    mode: int
    seed: int

    @property
    def count(self) -> int:
        return len(self.points)


def _as_polygon(region: Polygon | Iterable[Polygon], holes: Optional[Iterable[Polygon]]) -> Polygon:
    if isinstance(region, Polygon):
        base = region
    else:
        base = unary_union(list(region))
    if base.is_empty or base.geom_type != "Polygon" or not base.is_valid:
        raise ValueError("region must be one valid closed planar Polygon")
    hole_geoms = [h for h in (holes or []) if h is not None and not h.is_empty]
    if hole_geoms:
        base = base.difference(unary_union(hole_geoms))
    if base.is_empty or base.geom_type != "Polygon":
        raise ValueError("region minus holes must remain one non-empty Polygon")
    return base


def _rotate_xy(point: Point, angle: float, origin: tuple[float, float]) -> Point:
    if abs(angle) < 1e-15:
        return point
    ox, oy = origin
    dx, dy = point.x - ox, point.y - oy
    c, s = cos(angle), sin(angle)
    return Point(ox + c * dx - s * dy, oy + s * dx + c * dy)


def _inside(region: Polygon, point: Point) -> bool:
    return region.covers(point)


def _grid_candidates(region: Polygon, spacing: float, mode: int, jitter: float, angle: float, rng: Random) -> list[Point]:
    minx, miny, maxx, maxy = region.bounds
    cx, cy = region.centroid.x, region.centroid.y
    points: list[Point] = []
    row = 0
    y = miny + spacing * 0.5
    # A finite guard prevents pathological inputs from causing an unbounded loop.
    while y <= maxy + spacing * 0.5 and row < 100000:
        x_offset = spacing * 0.5
        if mode == 3 and row % 2:
            x_offset += spacing * 0.5
        x = minx + x_offset
        while x <= maxx + spacing * 0.5:
            jx = jy = 0.0
            if mode == 2:
                amplitude = max(0.0, min(0.49, jitter)) * spacing
                jx = rng.uniform(-amplitude, amplitude)
                jy = rng.uniform(-amplitude, amplitude)
            p = _rotate_xy(Point(x + jx, y + jy), angle, (cx, cy))
            if _inside(region, p):
                points.append(p)
            x += spacing
        y += spacing * (sqrt(3.0) / 2.0 if mode == 3 else 1.0)
        row += 1
    return points


def _grid_for_count(region: Polygon, count: int, mode: int, jitter: float, angle: float, rng: Random) -> tuple[list[Point], float]:
    area = region.area
    spacing = sqrt(max(area, 1e-12) / max(count, 1))
    best: tuple[list[Point], float, int] | None = None
    # The point count is monotone only approximately for concave regions; retain
    # the closest candidate across a bounded binary search.
    lo, hi = spacing * 0.15, spacing * 3.5
    for _ in range(18):
        mid = (lo + hi) / 2.0
        candidate = _grid_candidates(region, mid, mode, jitter, angle, rng)
        score = abs(len(candidate) - count)
        if best is None or score < best[2]:
            best = (candidate, mid, score)
        if len(candidate) > count:
            lo = mid
        else:
            hi = mid
    points, spacing, _ = best or ([], spacing, 0)
    # Preserve row/generation ordering and make the target cardinality exact when
    # there are enough candidates. This mirrors the component's approximate Count
    # contract while avoiding accidental overpopulation downstream.
    if len(points) > count:
        points = points[:count]
    return points, spacing


def _random_population(region: Polygon, count: int, seed: int, min_dist: Optional[float], attempts_factor: int) -> list[Point]:
    rng = Random(seed)
    minx, miny, maxx, maxy = region.bounds
    points: list[Point] = []
    distance = max(0.0, float(min_dist or 0.0))
    attempts = max(1000, max(1, count) * max(10, attempts_factor))
    for _ in range(attempts):
        if len(points) >= count:
            break
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if not _inside(region, p):
            continue
        if distance and any(p.distance(q) < distance for q in points):
            continue
        points.append(p)
    return points


def populate_region(region: Polygon, config: PopulateRegionConfig, *, holes: Optional[Iterable[Polygon]] = None) -> PopulateRegionResult:
    """Generate candidate points using UrbGEN's documented population modes.

    ``angle`` is in radians. ``jitter`` is interpreted as a fraction of grid
    spacing and is clamped to ``[0, 0.49]``. Random mode honours ``min_dist``;
    grid modes solve a spacing estimate from usable area and target count.
    """
    if config.count < 0:
        raise ValueError("count must be non-negative")
    mode = int(config.mode)
    if mode not in range(4):
        raise ValueError("mode must be 0 (random), 1 (regular), 2 (jittered), or 3 (staggered)")
    usable = _as_polygon(region, holes)
    if config.count == 0:
        return PopulateRegionResult((), usable, usable.area, None, mode, int(config.seed))
    rng = Random(int(config.seed))
    if mode == 0:
        points = _random_population(usable, config.count, int(config.seed), config.min_dist, config.max_attempts_factor)
        spacing = None
    else:
        points, spacing = _grid_for_count(usable, config.count, mode, config.jitter, config.angle, rng)
    return PopulateRegionResult(tuple(points), usable, usable.area, spacing, mode, int(config.seed))


# Alias matching the Grasshopper component's readable name.
populate_region_points = populate_region
