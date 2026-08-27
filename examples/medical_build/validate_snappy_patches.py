from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patch_dir", type=Path)
    parser.add_argument("--closed-reference", type=Path, required=True)
    parser.add_argument("--location", nargs=3, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = {"patches": {}, "location": list(map(float, args.location))}
    for patch_path in sorted(args.patch_dir.glob("*.stl")):
        mesh = trimesh.load_mesh(patch_path, process=False)
        output["patches"][patch_path.name] = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "components": len(mesh.split(only_watertight=False)),
            "boundary_edges": (
                int(len(mesh.edges_boundary))
                if hasattr(mesh, "edges_boundary")
                else None
            ),
            "area": float(mesh.area),
            "volume": float(abs(mesh.volume)),
        }

    import vtk

    points = vtk.vtkPoints()
    points.InsertNextPoint(*map(float, args.location))
    point_data = vtk.vtkPolyData()
    point_data.SetPoints(points)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(args.closed_reference))
    reader.Update()
    selector = vtk.vtkSelectEnclosedPoints()
    selector.SetInputData(point_data)
    selector.SetSurfaceData(reader.GetOutput())
    selector.CheckSurfaceOn()
    selector.Update()
    output["location_inside_closed_reference"] = bool(selector.IsInside(0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
