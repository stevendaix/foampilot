from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import vtk


def read_surface(path: Path):
    reader = vtk.vtkSTLReader() if path.suffix.lower() == ".stl" else vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(reader.GetOutput())
    return output


def write_stl(poly, path: Path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.Write()


def boundary_loops(surface):
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(surface)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(edges.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    boundary = stripper.GetOutput()

    result = []
    ids = vtk.vtkIdList()
    lines = boundary.GetLines()
    lines.InitTraversal()
    while lines.GetNextCell(ids):
        n = ids.GetNumberOfIds()
        if n < 3:
            continue
        polyline = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(n + 1)
        for i in range(n):
            points.InsertNextPoint(boundary.GetPoint(ids.GetId(i)))
            line.GetPointIds().SetId(i, i)
        points.InsertNextPoint(boundary.GetPoint(ids.GetId(0)))
        line.GetPointIds().SetId(n, n)
        polyline.SetPoints(points)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        polyline.SetLines(cells)

        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputData(polyline)
        triangulator.Update()
        cap = vtk.vtkPolyData()
        cap.DeepCopy(triangulator.GetOutput())
        if cap.GetNumberOfCells() == 0:
            continue
        mass = vtk.vtkMassProperties()
        mass.SetInputData(cap)
        mass.Update()
        center = np.mean([cap.GetPoint(i) for i in range(cap.GetNumberOfPoints())], axis=0)
        result.append({"center": center, "poly": cap, "boundary_points": n, "area": float(mass.GetSurfaceArea())})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    surface = read_surface(args.surface)
    data = json.loads(args.analysis.read_text())
    endpoints = []
    for branch in data["branches"]:
        endpoints.append((int(branch["source_cap_id"]), np.asarray(branch["points"][0], float), "source", int(branch["branch_id"])))
        endpoints.append((int(branch["target_cap_id"]), np.asarray(branch["points"][-1], float), "target", int(branch["branch_id"])))

    found = sorted(boundary_loops(surface), key=lambda item: item["area"], reverse=True)
    manifest = {"surface": str(args.surface), "analysis": str(args.analysis), "n_boundary_loops": len(found), "patches": {}}
    for index, loop in enumerate(found):
        cap_id, endpoint, role, branch_id = min(endpoints, key=lambda item: np.linalg.norm(item[1] - loop["center"]))
        name = "inlet" if index == 0 else f"outlet_{index - 1}"
        write_stl(loop["poly"], args.output / f"{name}.stl")
        manifest["patches"][name] = {
            "cap_id": cap_id,
            "role": role,
            "branch_id": branch_id,
            "center": loop["center"].tolist(),
            "area": loop["area"],
            "cells": loop["poly"].GetNumberOfCells(),
            "boundary_points": loop["boundary_points"],
        }

    write_stl(surface, args.output / "wall.stl")
    manifest["patches"]["wall"] = {"cells": surface.GetNumberOfCells(), "points": surface.GetNumberOfPoints(), "open_surface": True}
    (args.output / "patch_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
