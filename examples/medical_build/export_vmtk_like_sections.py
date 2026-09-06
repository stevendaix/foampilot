from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def read_polydata(path: Path):
    import vtk
    suffix = path.suffix.lower()
    if suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif suffix == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported VTK surface/centerline format: {path}")
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Empty VTK dataset: {path}")
    return output


def section_from_plane(surface, center, tangent):
    import vtk

    plane = vtk.vtkPlane()
    plane.SetOrigin(*map(float, center))
    plane.SetNormal(*map(float, tangent))

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)
    cutter.GenerateTrianglesOff()
    cutter.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    poly = stripper.GetOutput()

    contours = []
    for cell_id in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_id)
        points = np.asarray(
            [poly.GetPoint(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())],
            dtype=float,
        )
        if len(points) >= 3:
            contours.append(points)

    if not contours:
        return None

    contour = max(contours, key=len)
    closed = bool(np.linalg.norm(contour[0] - contour[-1]) < 1.0e-5)
    if closed:
        contour = contour[:-1]

    # VMTK computes geometric properties on the actual cut contour. For the
    # export, retain both the raw points and reproducible scalar diagnostics.
    rel = contour - np.asarray(center, dtype=float)
    radial = np.linalg.norm(rel, axis=1)
    return {
        "points": contour.tolist(),
        "closed": closed,
        "point_count": int(len(contour)),
        "radius_min": float(radial.min()),
        "radius_median": float(np.median(radial)),
        "radius_max": float(radial.max()),
        "radius_std": float(radial.std()),
    }


def write_sections_vtp(report: dict, path: Path) -> None:
    import vtk

    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()
    area = vtk.vtkDoubleArray(); area.SetName("CenterlineSectionArea")
    min_size = vtk.vtkDoubleArray(); min_size.SetName("CenterlineSectionMinSize")
    max_size = vtk.vtkDoubleArray(); max_size.SetName("CenterlineSectionMaxSize")
    shape = vtk.vtkDoubleArray(); shape.SetName("CenterlineSectionShape")
    closed = vtk.vtkIntArray(); closed.SetName("CenterlineSectionClosed")
    branch_ids = vtk.vtkIntArray(); branch_ids.SetName("BranchId")
    point_ids = vtk.vtkIntArray(); point_ids.SetName("CenterlinePointId")

    for branch in report["branches"]:
        for section in branch["sections"]:
            contour = np.asarray(section["points"], dtype=float)
            ids = vtk.vtkIdList()
            for point in contour:
                ids.InsertNextId(points.InsertNextPoint(*map(float, point)))
            polygons.InsertNextCell(ids)
            radial = np.asarray(section["points"], dtype=float) - np.asarray(section["center"], dtype=float)
            radii = np.linalg.norm(radial, axis=1)
            area.InsertNextValue(float(section.get("area", 0.0)))
            min_size.InsertNextValue(float(2.0 * radii.min()))
            max_size.InsertNextValue(float(2.0 * radii.max()))
            shape.InsertNextValue(float(radii.min() / max(radii.max(), 1.0e-12)))
            closed.InsertNextValue(int(section["closed"]))
            branch_ids.InsertNextValue(int(branch["branch_id"]))
            point_ids.InsertNextValue(int(section["centerline_point_id"]))

    output = vtk.vtkPolyData()
    output.SetPoints(points)
    output.SetPolys(polygons)
    for array in [area, min_size, max_size, shape, closed, branch_ids, point_ids]:
        output.GetCellData().AddArray(array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(output)
    writer.Write()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sections using VMTK-like plane/surface intersections")
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--centerlines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-vtp", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()
    surface = read_polydata(args.surface)
    centerlines = read_polydata(args.centerlines)
    branches = []

    for cell_id in range(centerlines.GetNumberOfCells()):
        cell = centerlines.GetCell(cell_id)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        sections = []
        stride = max(1, int(args.stride))
        for k in range(0, len(ids), stride):
            point_id = ids[k]
            center = np.asarray(centerlines.GetPoint(point_id), dtype=float)
            prev = np.asarray(centerlines.GetPoint(ids[max(0, k - 1)]), dtype=float)
            nxt = np.asarray(centerlines.GetPoint(ids[min(len(ids) - 1, k + 1)]), dtype=float)
            tangent = nxt - prev
            norm = float(np.linalg.norm(tangent))
            if norm <= 1.0e-12:
                continue
            tangent /= norm
            section = section_from_plane(surface, center, tangent)
            if section is not None:
                section.update({
                    "centerline_point_id": int(point_id),
                    "center": center.tolist(),
                    "tangent": tangent.tolist(),
                    "local_index": int(k),
                })
                sections.append(section)
        branches.append({"branch_id": int(cell_id), "centerline_point_ids": ids, "sections": sections})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "vtkCutter + vtkStripper, VMTK PolyDataCenterlineSections equivalent",
        "surface": str(args.surface),
        "centerlines": str(args.centerlines),
        "surface_points": int(surface.GetNumberOfPoints()),
        "surface_cells": int(surface.GetNumberOfCells()),
        "centerline_points": int(centerlines.GetNumberOfPoints()),
        "branch_count": int(centerlines.GetNumberOfCells()),
        "stride": int(args.stride),
        "branches": branches,
        "elapsed_seconds": round(time.perf_counter() - t0, 6),
    }
    args.output.write_text(json.dumps(report, indent=2))
    if args.output_vtp is not None:
        args.output_vtp.parent.mkdir(parents=True, exist_ok=True)
        write_sections_vtp(report, args.output_vtp)
    print(json.dumps({
        "output": str(args.output),
        "branches": len(branches),
        "sections": sum(len(b["sections"]) for b in branches),
        "closed_sections": sum(sum(int(s["closed"]) for s in b["sections"]) for b in branches),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
