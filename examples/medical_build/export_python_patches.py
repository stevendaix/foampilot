from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def cluster(points, tolerance):
    groups = []
    for index, point in enumerate(points):
        for group in groups:
            if np.linalg.norm(point - group["center"]) <= tolerance:
                group["indices"].append(index)
                group["center"] = np.mean([points[i] for i in group["indices"]], axis=0)
                break
        else:
            groups.append({"center": np.asarray(point, float), "indices": [index]})
    return groups


def cap_mesh(section, reverse=False):
    points = np.asarray(section.get("phase_locked_points") or section["points"], float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-8:
        points = points[:-1]
    center = points.mean(axis=0)
    vertices = np.vstack([points, center])
    center_id = len(points)
    faces = []
    for i in range(len(points)):
        face = (center_id, i, (i + 1) % len(points))
        faces.append(face if not reverse else (center_id, (i + 1) % len(points), i))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-tolerance", type=float, default=1.0)
    parser.add_argument("--cap-thickness", type=float, default=0.9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.sections.read_text())
    endpoints = []
    for branch in data["branches"]:
        if branch["sections"]:
            endpoints.extend([
                {"branch": branch["branch_id"], "side": "first", "section": branch["sections"][0]},
                {"branch": branch["branch_id"], "side": "last", "section": branch["sections"][-1]},
            ])

    centers = [np.asarray(endpoint["section"]["center"], float) for endpoint in endpoints]
    groups = cluster(centers, args.endpoint_tolerance)
    terminal_groups = []
    for group in groups:
        sides = {endpoints[index]["side"] for index in group["indices"]}
        if len(group["indices"]) == 1 or len(sides) == 1:
            terminal_groups.append(group)
    if len(terminal_groups) < 2:
        raise RuntimeError(f"Could not identify terminal endpoints: {len(terminal_groups)}")

    terminals = [endpoints[group["indices"][0]] for group in terminal_groups]
    terminals.sort(
        key=lambda endpoint: float(
            np.pi
            * max(
                np.linalg.norm(np.asarray(point) - np.asarray(endpoint["section"]["center"]))
                for point in (endpoint["section"].get("phase_locked_points") or endpoint["section"]["points"])
            ) ** 2
        ),
        reverse=True,
    )

    mesh = trimesh.load_mesh(args.surface, process=False)
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    remove = np.zeros(len(mesh.faces), dtype=bool)
    patch_report = []

    for index, endpoint in enumerate(terminals):
        section = endpoint["section"]
        center = np.asarray(section["center"], float)
        tangent = np.asarray(section["tangent"], float)
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        contour = np.asarray(section.get("phase_locked_points") or section["points"], float)
        radius = float(np.max(np.linalg.norm(contour - center, axis=1)))
        axial = np.abs((face_centers - center) @ tangent)
        alignment = np.abs(face_normals @ tangent)
        radial = np.linalg.norm(
            (face_centers - center) - np.outer((face_centers - center) @ tangent, tangent), axis=1
        )
        mask = (axial <= args.cap_thickness) & (alignment >= 0.45) & (radial <= radius * 1.25)
        remove |= mask
        name = "inlet" if index == 0 else f"outlet_{index - 1}"
        cap = cap_mesh(section, reverse=bool(np.dot(np.mean(face_normals[mask], axis=0) if np.any(mask) else tangent, tangent) > 0))
        cap.export(args.output / f"{name}.stl")
        patch_report.append({
            "name": name,
            "branch": endpoint["branch"],
            "side": endpoint["side"],
            "center": center.tolist(),
            "radius": radius,
            "removed_wall_faces": int(mask.sum()),
            "cap_faces": int(len(cap.faces)),
        })

    wall = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[~remove], process=True)
    wall.export(args.output / "wall.stl")
    report = {
        "input_surface": str(args.surface),
        "sections": str(args.sections),
        "terminal_count": len(terminals),
        "patches": patch_report,
        "wall_faces": int(len(wall.faces)),
        "wall_watertight": bool(wall.is_watertight),
        "wall_components": len(wall.split(only_watertight=False)),
    }
    (args.output / "python_patch_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
