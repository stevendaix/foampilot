import gmsh
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def create_rectangle_geo(
    bounds: Tuple[float, float, float, float, float, float],
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.05,
    depth: float = 0.01,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, xmax - xmin, ymax - ymin, depth
        )
    )
    for patch, direction in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, direction)
    geo += (
        '\nPhysical Volume("FLUID") = {1};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_channel_with_obstacle_geo(
    bounds: Tuple[float, float, float, float, float, float],
    obstacle_center: Tuple[float, float, float],
    obstacle_radius: float,
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.05,
    obstacle_height: float = 0.1,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    cx, cy, cz = obstacle_center
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Channel\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx, dy, dz
        ) +
        '// Cylindrical obstacle\n'
        'Cylinder(2) = {{{}, {}, {}, {}, {}, {}, {}, 2*Pi}};\n\n'.format(
            cx, cy, cz, cx, cy, cz + obstacle_height, obstacle_radius
        ) +
        '// Boolean difference\n'
        'BooleanDifference(3) = {{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }};\n\n'
    )
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {3};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_step_geo(
    bounds: Tuple[float, float, float, float, float, float],
    step_height: float,
    step_position: float,
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.02,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    dx_inlet = step_position - xmin
    dx_outlet = xmax - step_position
    total_h = ymax - ymin
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Lower channel (before and after step)\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx_inlet + dx_outlet, step_height, zmax - zmin
        ) +
        '// Upper channel (only after step)\n'
        'Box(2) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            step_position, ymin + step_height, zmin, dx_outlet, total_h - step_height, zmax - zmin
        ) +
        '// Union\n'
        'BooleanUnion(3) = {{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }};\n\n'
    )
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {3};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_cylinder_in_channel_geo(
    channel_dims: Tuple[float, float, float, float, float, float],
    cylinder_pos: Tuple[float, float, float],
    cylinder_radius: float,
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.05,
    channel_depth: float = 0.1,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = channel_dims
    cx, cy, cz = cylinder_pos
    dx = xmax - xmin
    dy = ymax - ymin
    dz = channel_depth
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Channel\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx, dy, dz
        ) +
        '// Cylinder\n'
        'Cylinder(2) = {{{}, {}, {}, {}, {}, {}, {}, 2*Pi}};\n\n'.format(
            cx, cy, cz, cx, cy, cz + channel_depth, cylinder_radius
        ) +
        '// Boolean difference\n'
        'BooleanDifference(3) = {{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }};\n\n'
    )
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {3};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_car_channel_geo(
    channel_dims: Tuple[float, float, float, float, float, float],
    obstacle_center: Tuple[float, float, float],
    obstacle_size: Tuple[float, float, float],
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.1,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = channel_dims
    cx, cy, cz = obstacle_center
    sx, sy, sz = obstacle_size
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Channel\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx, dy, dz
        ) +
        '// Car obstacle\n'
        'Box(2) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            cx - sx / 2, cy - sy / 2, cz - sz / 2, sx, sy, sz
        ) +
        '// Boolean difference\n'
        'BooleanDifference(3) = {{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }};\n\n'
    )
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {3};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_step_geo_v2(
    bounds: Tuple[float, float, float, float, float, float],
    step_height: float,
    step_position: float,
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.02,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    dx_inlet = step_position - xmin
    dx_outlet = xmax - step_position
    total_h = ymax - ymin
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Lower channel (before and after step)\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx_inlet + dx_outlet, step_height, zmax - zmin
        ) +
        '// Upper channel (only after step)\n'
        'Box(2) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            step_position, ymin + step_height, zmin, dx_outlet, total_h - step_height, zmax - zmin
        ) +
        '// Union\n'
        'BooleanUnion(3) = {{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }};\n\n'
    )
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {3};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_thermal_room_geo(
    room_dims: Tuple[float, float, float, float, float, float],
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.1,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = room_dims
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Room\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx, dy, dz
        )
    )
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {1};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_buildings_geo(
    domain_dims: Tuple[float, float, float, float, float, float],
    buildings: List[Dict],
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 5.0,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = domain_dims
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Fluid domain\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx, dy, dz
        )
    )
    for i, b in enumerate(buildings, start=2):
        bx, by, bz = b["center"]
        bsx, bsy, bsz = b["size"]
        geo += '// Building {}\n'.format(i - 1)
        geo += 'Box({}) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            i, bx - bsx / 2, by - bsy / 2, bz - bsz / 2, bsx, bsy, bsz
        )
    geo += '// Subtract all buildings from fluid\n'
    for i in range(2, 2 + len(buildings)):
        geo += 'BooleanDifference(1000) = {{ Volume{{1}}; Delete; }}{{ Volume{{{}}}; Delete; }};\n'.format(i)
        if i < 2 + len(buildings) - 1:
            geo += 'BooleanDifference(1) = {{ Volume{{1000}}; Delete; }}{{ Volume{{{}}}; Delete; }};\n'.format(i + 1)
        else:
            geo += 'BooleanDifference(1) = {{ Volume{{1000}}; Delete; }}{{ Volume{{{}}}; Delete; }};\n'.format(i + 1)
    geo += '\n'
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {1};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path


def create_motorcycle_geo(
    domain_dims: Tuple[float, float, float, float, float, float],
    road_size: Tuple[float, float, float],
    body_size: Tuple[float, float, float],
    body_center: Tuple[float, float, float],
    wheels: List[Dict],
    patch_names: Dict[str, str],
    filename: str,
    lc: float = 0.2,
) -> Path:
    xmin, ymin, zmin, xmax, ymax, zmax = domain_dims
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    rx, ry, rz = road_size
    bx, by, bz = body_size
    bcx, bcy, bcz = body_center
    geo = (
        'SetFactory("OpenCASCADE");\n\n'
        'lc = {};\n\n'.format(lc) +
        '// Fluid domain\n'
        'Box(1) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            xmin, ymin, zmin, dx, dy, dz
        ) +
        '// Road\n'
        'Box(2) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            -rx / 2, -ry / 2, -rz, rx, ry, rz
        ) +
        '// Motorcycle body\n'
        'Box(3) = {{{}, {}, {}, {}, {}, {} }};\n\n'.format(
            bcx - bx / 2, bcy - by / 2, bcz - bz / 2, bx, by, bz
        )
    )
    for i, w in enumerate(wheels, start=4):
        wx, wy, wz = w["center"]
        wr = w["radius"]
        geo += '// Wheel {}\n'.format(i - 3)
        geo += 'Cylinder({}) = {{{}, {}, {}, {}, {}, {}, {}, 2*Pi}};\n\n'.format(
            i, wx, wy, wz, wx, wy, wz + 0.2, wr
        )
    geo += '// Union of all obstacles\n'
    geo += 'BooleanFragments(100) = {{ Volume{{2}}; Delete; }}{{ Volume{{3}}; Delete; }};\n'
    for i in range(4, 4 + len(wheels)):
        geo += 'BooleanFragments(100) = {{ Volume{{100}}; Delete; }}{{ Volume{{{}}}; Delete; }};\n'.format(i)
    geo += '\n// Subtract obstacles from fluid\n'
    geo += 'BooleanDifference(200) = {{ Volume{{1}}; Delete; }}{{ Volume{{100}}; Delete; }};\n\n'
    for patch, face_tag in patch_names.items():
        geo += 'Physical Surface("{}") = {{{}}};\n'.format(patch, face_tag)
    geo += (
        '\nPhysical Volume("FLUID") = {200};\n\n'
        'Mesh 3;\n'
    )
    path = Path(filename)
    path.write_text(geo)
    return path
