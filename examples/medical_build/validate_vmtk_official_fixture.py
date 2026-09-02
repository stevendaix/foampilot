from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np


def read_dataset(path: Path):
    import vtk
    if path.suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif path.suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif path.suffix == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError(path)
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    input_dir = args.root / "input" if (args.root / "input").is_dir() else args.root
    surface = read_dataset(input_dir / "aorta-surface.vtp")
    centerline = read_dataset(input_dir / "aorta-centerline.vtp")
    branches = read_dataset(input_dir / "aorta-centerline-branches.vtp")
    references = [
        "aorta-surface.stl",
        "aorta-surface-open-ends.stl",
        "aorta-surface-branch-split.vtp",
        "aorta-surface-connectivity-reference.stl",
        "aorta-surface-segment-2.stl",
        "aorta-centerline-attribute-branches.vtp",
        "aorta-centerline-referencesystem.vtp",
        "aorta-mesh.vtu",
    ]
    report = {
        "surface": {
            "points": surface.GetNumberOfPoints(),
            "cells": surface.GetNumberOfCells(),
            "bounds": list(surface.GetBounds()),
        },
        "centerline": {
            "points": centerline.GetNumberOfPoints(),
            "cells": centerline.GetNumberOfCells(),
        },
        "branches": {
            "points": branches.GetNumberOfPoints(),
            "cells": branches.GetNumberOfCells(),
        },
        "references": {name: (input_dir / name).exists() for name in references},
    }
    import vtk
    intersections = []
    stride = max(1, branches.GetNumberOfPoints() // 12)
    for point_id in range(0, branches.GetNumberOfPoints(), stride):
        center = np.asarray(branches.GetPoint(point_id), dtype=float)
        plane = vtk.vtkPlane()
        plane.SetOrigin(*center)
        plane.SetNormal(1.0, 0.0, 0.0)
        cutter = vtk.vtkCutter()
        cutter.SetInputData(surface)
        cutter.SetCutFunction(plane)
        cutter.Update()
        cut = cutter.GetOutput()
        intersections.append({
            "point_id": point_id,
            "points": cut.GetNumberOfPoints(),
            "cells": cut.GetNumberOfCells(),
        })
    report["sample_plane_intersections"] = intersections
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
