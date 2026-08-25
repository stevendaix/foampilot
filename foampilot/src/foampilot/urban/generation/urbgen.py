"""UrbGEN-compatible stochastic urban massing for foampilot.

This module mirrors the public UrbGEN generator contract: site/setback,
centroid population, typology grammar, tower growth toward BCR, podium
expansion, FAR-derived floors, height regulation, rotation and post-placement
rules. It returns native ``UrbanModel`` objects for Gmsh/build123d adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, sqrt, atan2, cos, sin, degrees, radians
from random import Random
from typing import Iterable, Optional

from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split, unary_union

from foampilot.urban.model.urban_model import Building, CFDLOD, RoofType, UrbanModel


@dataclass(frozen=True)
class UrbGENConfig:
    bcr: float = 0.35
    upper_bcr: Optional[float] = None
    far: float = 2.0
    setback: float = 5.0
    min_width: float = 8.0
    tower_size_mode: int = 1
    min_footprint_per_tower: float = 20.0
    max_footprint_per_tower: float = 500.0
    max_length_width_ratio: float = 6.0
    min_tower_distance: float = 8.0
    tower_bcr_priority: float = 0.70
    tower_grow_step: float = 1.0
    tower_grow_iterations: int = 100
    seed: int = 0
    tower_typology_mode: int = 6
    arm_length_ratio: float = 1.0
    podium_floors: int = 2
    podium_min_offset: float = 2.0
    podium_max_offset: float = 12.0
    floor_height: float = 3.0
    global_rotation_mode: int = 0
    uniform_rotation_deg: float = 0.0
    courtyard_count: int = 1
    courtyard_break_count: int = 4
    courtyard_break_width: float = 8.0
    courtyard_zone_gap: float = 6.0
    courtyard_split_angle: float = 0.0
    courtyard_break_shift: float = 0.0
    courtyard_layout_mode: int = 0
    height_variation: float = 0.20
    enforce_height_regulation: bool = False
    height_regulation_mode: int = 0
    max_building_height: float = 1000.0
    min_building_height: float = 3.0
    move_to_boundary: bool = False
    move_all_to_setback: bool = False
    align_towers_to_edge: bool = False
    edge_align_both_orientations: bool = True
    move_tower_to_podium_edge: bool = False
    floor_height_override: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.bcr <= 1.0 or (self.upper_bcr is not None and self.upper_bcr < self.bcr):
            raise ValueError("bcr/upper_bcr must satisfy 0 < bcr <= upper_bcr <= 1")
        if self.far <= 0 or self.min_width < 2 or self.min_tower_distance < 0:
            raise ValueError("far must be positive, min_width >= 2 and distance non-negative")
        if self.max_length_width_ratio < 2 or self.tower_bcr_priority < 0 or self.tower_bcr_priority > 1:
            raise ValueError("invalid tower sizing or BCR priority")
        if self.tower_typology_mode not in range(8):
            raise ValueError("tower_typology_mode must be between 0 and 7")


@dataclass
class UrbGENResult:
    model: UrbanModel
    site: Polygon
    buildable_site: Polygon
    tower_footprints: list[Polygon]
    podium_footprints: list[Polygon]
    tower_angles: list[float]
    tower_typologies: list[int]
    actual_bcr: float
    actual_far: float
    total_gfa: float
    tower_gfa: float
    podium_gfa: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def target_bcr(self) -> float:
        return self.diagnostics["target_bcr"]

    @property
    def target_far(self) -> float:
        return self.diagnostics["target_far"]

    @property
    def bcr_error(self) -> float:
        return self.actual_bcr - self.target_bcr

    @property
    def far_error(self) -> float:
        return self.actual_far - self.target_far


def _rect(width: float, length: float) -> Polygon:
    w, l = width / 2.0, length / 2.0
    return Polygon([(-l, -w), (l, -w), (l, w), (-l, w)])


def _grammar(width: float, length: float, typology: int, arm_ratio: float) -> Polygon:
    base = _rect(width, length)
    if typology == 0:
        return base
    arm = max(width * 0.3, width * arm_ratio)
    if typology == 1:
        return unary_union([base, translate(_rect(arm, length), xoff=-length / 2 + arm / 2)])
    if typology == 2:
        return unary_union([base, translate(_rect(length, arm), yoff=length / 2 - arm / 2)])
    if typology == 3:
        return unary_union([base, translate(_rect(arm, length), xoff=-length / 2 + arm / 2), translate(_rect(arm, length), xoff=length / 2 - arm / 2)])
    if typology == 4:
        return base.difference(translate(_rect(length * 0.65, width * 0.55), xoff=length * 0.2))
    return unary_union([base, _rect(width * arm_ratio, length)])


def _lattice(region: Polygon, spacing: float) -> Iterable[Point]:
    minx, miny, maxx, maxy = region.bounds
    y = miny + spacing / 2
    while y <= maxy:
        x = minx + spacing / 2
        while x <= maxx:
            p = Point(x, y)
            if region.covers(p):
                yield p
            x += spacing
        y += spacing


def _angle(config: UrbGENConfig, rng: Random) -> float:
    if config.global_rotation_mode == 1:
        return max(0.0, min(180.0, config.uniform_rotation_deg))
    if config.global_rotation_mode in (2, 3):
        return float(rng.choice((0, 45, 90, 135, 180)))
    return float(rng.choice((0, 45, 90, 135, 180)))


def _largest_polygon(geometry):
    if geometry.is_empty:
        return None
    if geometry.geom_type == "Polygon":
        return geometry
    parts = [g for g in getattr(geometry, "geoms", ()) if g.geom_type == "Polygon"]
    return max(parts, key=lambda g: g.area) if parts else None


def _split_site_into_zones(region: Polygon, count: int, angle_deg: float, gap: float) -> list[Polygon]:
    if count <= 1:
        return [region]
    minx, miny, maxx, maxy = region.bounds
    diagonal = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5 * 3.0
    angle = radians(angle_deg + (0.0 if (maxx - minx) >= (maxy - miny) else 90.0))
    ux, uy = cos(angle), sin(angle)
    vx, vy = -uy, ux
    center = region.centroid
    zones = []
    span = diagonal / count
    for i in range(count):
        lo = (i - count / 2.0) * span + gap * (i - count / 2.0)
        hi = lo + span - gap
        p0 = (center.x + ux * lo - vx * diagonal, center.y + uy * lo - vy * diagonal)
        p1 = (center.x + ux * lo + vx * diagonal, center.y + uy * lo + vy * diagonal)
        p2 = (center.x + ux * hi + vx * diagonal, center.y + uy * hi + vy * diagonal)
        p3 = (center.x + ux * hi - vx * diagonal, center.y + uy * hi - vy * diagonal)
        piece = _largest_polygon(region.intersection(Polygon([p0, p1, p2, p3])))
        if piece is not None and piece.area > 1.0:
            zones.append(piece)
    return zones or [region]


def _courtyard_ring_segments(zone: Polygon, width: float, breaks: int, break_width: float, coverage: float, shift: float, seed: int) -> list[Polygon]:
    inner = zone.buffer(-width)
    if inner.is_empty:
        return []
    ring = zone.difference(inner)
    perimeter = zone.exterior
    gaps = []
    n = max(1, breaks)
    # Breaks are placed deterministically along the perimeter, with a seed-dependent phase.
    phase = ((seed * 131 + int(shift * 100)) % 1000) / 1000.0
    for i in range(n):
        point = perimeter.interpolate(((i + phase) / n) * perimeter.length)
        gaps.append(point.buffer(width + max(0.3, break_width * max(0.35, 1.0 - coverage))))
    pieces = ring.difference(unary_union(gaps))
    result = []
    for part in getattr(pieces, "geoms", (pieces,)):
        if part.geom_type == "Polygon" and part.area >= width * width * 0.3:
            result.append(part)
    return result


def _build_courtyard_layout(region: Polygon, config: UrbGENConfig, target_area: float) -> list[Polygon]:
    """Build Courtyard segments by zones and grow their perimeter coverage to target BCR."""
    zones = _split_site_into_zones(region, max(1, config.courtyard_count), config.courtyard_split_angle, config.courtyard_zone_gap)
    best = []
    coverage = {0: 0.55, 1: 0.70, 2: 0.85, 3: Random(config.seed + 7301).uniform(0.50, 0.90)}.get(config.tower_size_mode, 0.70)
    for _ in range(max(1, config.tower_grow_iterations)):
        segments = []
        for zone_index, zone in enumerate(zones):
            breaks = config.courtyard_break_count
            if config.tower_size_mode == 0:
                breaks += 2
            elif config.tower_size_mode == 2:
                breaks = max(1, breaks - 1)
            elif config.tower_size_mode == 3:
                breaks = max(1, breaks + Random(config.seed * 131 + zone_index * 17 + 6001).randint(-2, 2))
            segments.extend(_courtyard_ring_segments(zone, config.min_width, breaks, config.courtyard_break_width, coverage, config.courtyard_break_shift, config.seed + zone_index))
        total = sum(p.area for p in segments)
        if total > sum(p.area for p in best):
            best = segments
        if total >= target_area * 0.985 or coverage >= 0.98:
            break
        coverage = min(0.98, coverage + 0.03)
    return best


def _courtyard_blocks(region: Polygon, config: UrbGENConfig) -> list[Polygon]:
    """Create perimeter blocks with explicit gaps, matching Courtyard mode."""
    band = max(2.0, config.min_width)
    ring = region.difference(region.buffer(-band))
    if ring.is_empty:
        return []
    perimeter = ring.boundary
    length = max(band * 2.0, min(band * 5.0, perimeter.length / max(1, config.courtyard_break_count + 1)))
    blocks = []
    n = max(1, int(perimeter.length / max(length + config.courtyard_break_width, 1.0)))
    for i in range(n):
        distance = (i + 0.5) * perimeter.length / n
        point = perimeter.interpolate(distance)
        delta = 0.5
        p0 = perimeter.interpolate(max(0.0, distance - delta))
        p1 = perimeter.interpolate(min(perimeter.length, distance + delta))
        angle = __import__("math").degrees(__import__("math").atan2(p1.y - p0.y, p1.x - p0.x))
        block = translate(rotate(_rect(band, length), angle, origin=(0, 0)), xoff=point.x, yoff=point.y)
        if region.covers(block):
            blocks.append(block)
    return blocks


def _move_to_boundary(shape: Polygon, region: Polygon, radial: bool) -> Polygon:
    center = Point(region.centroid.x, region.centroid.y)
    direction = Point(shape.centroid.x - center.x, shape.centroid.y - center.y) if radial else Point(region.exterior.interpolate(region.exterior.project(shape.centroid)).x - shape.centroid.x, region.exterior.interpolate(region.exterior.project(shape.centroid)).y - shape.centroid.y)
    norm = max((direction.x * direction.x + direction.y * direction.y) ** 0.5, 1e-9)
    step = 1.0
    moved = shape
    for _ in range(10000):
        candidate = translate(moved, xoff=direction.x / norm * step, yoff=direction.y / norm * step)
        if not region.covers(candidate):
            break
        moved = candidate
    return moved


def _move_tower_to_podium_edge(shape: Polygon, podium: Polygon, region: Polygon) -> Polygon:
    """Slide a tower toward the nearest podium/site edge without leaving it."""
    target = region.exterior.interpolate(region.exterior.project(shape.centroid))
    dx, dy = target.x - shape.centroid.x, target.y - shape.centroid.y
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    moved = shape
    for _ in range(10000):
        candidate = translate(moved, xoff=dx / norm, yoff=dy / norm)
        if not podium.covers(candidate):
            break
        moved = candidate
    return moved


def _align_to_edge(shape: Polygon, region: Polygon, angle: float) -> Polygon:
    nearest = region.exterior.interpolate(region.exterior.project(shape.centroid))
    tangent = 0.0
    best = 1e100
    coords = list(region.exterior.coords)
    for a, b in zip(coords, coords[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = ((shape.centroid.x - (a[0] + b[0]) / 2) ** 2 + (shape.centroid.y - (a[1] + b[1]) / 2) ** 2)
        if d < best:
            best, tangent = d, __import__("math").degrees(__import__("math").atan2(dy, dx))
    return rotate(shape, tangent - angle, origin="centroid")


def _grow_towers_to_bcr(towers, angles, codes, lengths, width, buildable, config, target_area):
    if not towers or codes[0] == 7:
        return towers, lengths
    current = sum(p.area for p in towers)
    rng = Random(config.seed + 10000)
    for _ in range(max(0, config.tower_grow_iterations)):
        if current >= target_area * 0.985:
            break
        improved = False
        order = list(range(len(towers)))
        rng.shuffle(order)
        for i in order:
            if current >= target_area * 0.985:
                break
            new_length = lengths[i] + config.tower_grow_step
            candidate = translate(rotate(_grammar(width, new_length, codes[i], max(0.3, config.arm_length_ratio)), angles[i], origin=(0, 0)), xoff=towers[i].centroid.x, yoff=towers[i].centroid.y)
            if candidate.area > config.max_footprint_per_tower or not buildable.covers(candidate):
                continue
            others = [p for j, p in enumerate(towers) if j != i]
            if any(not candidate.buffer(config.min_tower_distance).disjoint(p) for p in others):
                continue
            current += candidate.area - towers[i].area
            towers[i] = candidate
            lengths[i] = new_length
            improved = True
        if not improved:
            break
    return towers, lengths


def _trim_towers_to_bcr(towers, angles, codes, lengths, site, config):
    target = site.area * config.bcr
    if not towers:
        return towers, angles, codes, lengths
    center = site.centroid
    order = sorted(range(len(towers)), key=lambda i: towers[i].centroid.distance(center), reverse=True)
    keep = list(range(len(towers)))
    while sum(towers[i].area for i in keep) > target * 1.015 and len(keep) > 1:
        keep.remove(order[len(order) - len(keep)])
    keep.sort()
    return ([towers[i] for i in keep], [angles[i] for i in keep], [codes[i] for i in keep], [lengths[i] for i in keep])


def _height_distribution(count, base_floors, variation, seed, min_floors):
    if count <= 0:
        return []
    rng = Random(seed + 8888)
    if variation <= 0.01:
        return [max(min_floors, int(base_floors))] * count
    spread = max(1, int(round(abs(variation) * max(1, base_floors) / 2.0)))
    values = [max(min_floors, int(round(base_floors + rng.uniform(-spread, spread)))) for _ in range(count)]
    if count >= 3 and max(values) == min(values):
        values[0] = max(min_floors, values[0] - spread)
        values[-1] += spread
    return values


def _find_podium_offset(towers, buildable, config, target_area):
    if config.podium_floors <= 0 or not towers:
        return 0.0, []
    lo, hi = max(0.0, config.podium_min_offset), max(config.podium_min_offset + 0.5, config.podium_max_offset)
    best_offset, best_podium = 0.0, []
    for _ in range(12):
        mid = (lo + hi) / 2.0
        union = unary_union([p.buffer(mid) for p in towers])
        clipped = union.intersection(buildable)
        parts = [clipped] if clipped.geom_type == "Polygon" else [p for p in clipped.geoms if p.geom_type == "Polygon"]
        area = sum(p.area for p in parts)
        if area >= target_area:
            best_offset, best_podium = mid, parts
            hi = mid
        else:
            lo = mid
    if not best_podium:
        mid = min(config.podium_max_offset, max(config.podium_min_offset, 0.35 * sqrt(buildable.area)))
        clipped = unary_union([p.buffer(mid) for p in towers]).intersection(buildable)
        best_podium = [clipped] if clipped.geom_type == "Polygon" else [p for p in clipped.geoms if p.geom_type == "Polygon"]
        best_offset = mid
    return best_offset, best_podium


def generate_urbgen(site: Polygon, config: UrbGENConfig = UrbGENConfig(), *, crs: Optional[str] = None, centroids: Optional[Iterable[Point]] = None) -> UrbGENResult:
    """Generate a deterministic random UrbGEN neighbourhood from one site."""
    if site.is_empty or not site.is_valid or site.area <= 0:
        raise ValueError("site must be a valid, non-empty polygon")
    buildable = site.buffer(-max(0.0, config.setback))
    if buildable.is_empty:
        buildable = site
    if buildable.geom_type != "Polygon":
        buildable = max(buildable.geoms, key=lambda g: g.area)
    rng = Random(config.seed % 10000)
    typology = config.tower_typology_mode
    courtyard_mode = typology == 7
    courtyard_towers = []
    if courtyard_mode:
        courtyard_towers = _build_courtyard_layout(buildable, config, site.area * config.bcr * config.tower_bcr_priority)
        seeds = [p.centroid for p in courtyard_towers]
        typology = 7
    target_tower_area = site.area * config.bcr * config.tower_bcr_priority
    width = max(2.0, config.min_width)
    mode_factor = {0: 0.75, 1: 1.0, 2: 1.25, 3: rng.uniform(0.75, 1.25)}.get(config.tower_size_mode, 1.0)
    length = width * min(config.max_length_width_ratio, 2.0 * mode_factor)
    spacing = max(width + config.min_tower_distance, width * 1.5)
    seeds = list(centroids) if centroids is not None else (seeds if courtyard_mode else list(_lattice(buildable, spacing)))
    rng.shuffle(seeds)
    towers: list[Polygon] = []
    angles: list[float] = []
    codes: list[int] = []
    lengths: list[float] = []
    covered = 0.0
    for seed in seeds:
        if covered >= target_tower_area or len(towers) >= config.tower_grow_iterations:
            break
        code = rng.randrange(6) if config.tower_typology_mode == 6 else typology
        angle = _angle(config, rng)
        if courtyard_mode:
            shape = courtyard_towers[len(towers)]
            angle = 0.0
        else:
            shape = rotate(_grammar(width, length, code, max(0.3, config.arm_length_ratio)), angle, origin=(0, 0))
            shape = translate(shape, xoff=seed.x, yoff=seed.y)
        for _ in range(max(1, config.tower_grow_iterations)):
            if shape.area >= config.min_footprint_per_tower and shape.area <= config.max_footprint_per_tower and buildable.covers(shape) and all(shape.buffer(config.min_tower_distance).disjoint(other) for other in towers):
                break
            if courtyard_mode:
                break
            if shape.area > config.max_footprint_per_tower or not buildable.covers(shape):
                shape = rotate(_grammar(width, max(width, length - config.tower_grow_step), code, config.arm_length_ratio), angle, origin=(0, 0))
                shape = translate(shape, xoff=seed.x, yoff=seed.y)
                break
            length += config.tower_grow_step
            shape = rotate(_grammar(width, length, code, config.arm_length_ratio), angle, origin=(0, 0))
            shape = translate(shape, xoff=seed.x, yoff=seed.y)
        if shape.area < config.min_footprint_per_tower or not buildable.covers(shape) or any(not shape.buffer(config.min_tower_distance).disjoint(other) for other in towers):
            continue
        if not courtyard_mode and config.move_to_boundary:
            shape = _move_to_boundary(shape, buildable, radial=True)
        if not courtyard_mode and config.move_all_to_setback:
            shape = _move_to_boundary(shape, buildable, radial=False)
        if not courtyard_mode and config.align_towers_to_edge:
            shape = _align_to_edge(shape, buildable, angle)
        if not buildable.covers(shape):
            continue
        towers.append(shape)
        angles.append(angle)
        codes.append(code)
        lengths.append(length)
        covered += shape.area
    if not towers:
        raise ValueError("no UrbGEN tower fits the buildable site")

    towers, lengths = _grow_towers_to_bcr(towers, angles, codes, lengths, width, buildable, config, target_tower_area)
    towers, angles, codes, lengths = _trim_towers_to_bcr(towers, angles, codes, lengths, site, config)
    union = unary_union(towers)
    podium: list[Polygon] = []
    actual_offset = 0.0
    if config.podium_floors > 0:
        actual_offset, podium = _find_podium_offset(towers, buildable, config, site.area * config.bcr)
    if config.move_tower_to_podium_edge and podium:
        podium_union = unary_union(podium)
        towers = [_move_tower_to_podium_edge(t, podium_union, buildable) for t in towers]
    tower_floor_area = sum(p.area for p in towers)
    podium_area = sum(p.area for p in podium)
    floor_h = config.floor_height_override or config.floor_height
    target_gfa = site.area * config.far
    tower_floors = max(1, ceil(max(0.0, target_gfa - podium_area * config.podium_floors) / tower_floor_area))
    floors = _height_distribution(len(towers), tower_floors, config.height_variation, config.seed, max(1, ceil(config.min_building_height / floor_h)))
    heights = []
    model = UrbanModel(crs=crs)
    for i, footprint in enumerate(towers):
        h = floors[i] * floor_h
        if config.enforce_height_regulation:
            if config.height_regulation_mode == 1:
                h = min(config.max_building_height, h)
            elif config.height_regulation_mode == 2:
                h = max(config.min_building_height, min(config.max_building_height, h))
            else:
                h = max(config.min_building_height, min(config.max_building_height, h))
        h = max(config.min_building_height, h)
        heights.append(h)
        model.add_building(Building(f"urbgen-tower-{i:04d}", footprint, 0.0, h, RoofType.FLAT, CFDLOD.LOD1, "urbgen", 1.0, {"typology": codes[i], "typology_name": ("I", "L", "T", "H", "C", "Plus", "Random", "Courtyard")[codes[i]], "angle_deg": angles[i], "floors": round(h / floor_h)}))
    for i, footprint in enumerate(podium):
        model.add_building(Building(f"urbgen-podium-{i:04d}", footprint, 0.0, config.podium_floors * floor_h, RoofType.FLAT, CFDLOD.LOD1, "urbgen-podium", 1.0, {"podium_offset": actual_offset, "floors": config.podium_floors}))
    gfa = sum(b.area * max(1, round(b.height / floor_h)) for b in model.buildings())
    return UrbGENResult(model, site, buildable, towers, podium, angles, codes, sum(b.area for b in model.buildings()) / site.area, gfa / site.area, gfa, tower_floor_area * tower_floors, podium_area * config.podium_floors, {"target_bcr": config.bcr, "target_far": config.far, "tower_count": len(towers), "podium_count": len(podium), "actual_podium_offset": actual_offset, "heights": heights, "seed": config.seed})
