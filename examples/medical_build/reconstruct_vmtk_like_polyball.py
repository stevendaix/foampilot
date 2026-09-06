from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    import vtk
    from vtk.util import numpy_support

    ap = argparse.ArgumentParser(description="Pure Python/VTK PolyBall-like reconstruction")
    ap.add_argument("--centerlines", type=Path, required=True)
    ap.add_argument("--output-vtp", type=Path, required=True)
    ap.add_argument("--output-stl", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=0.75)
    args = ap.parse_args()

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(args.centerlines))
    reader.Update()
    cl = reader.GetOutput()
    radius_array = cl.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if radius_array is None:
        raise RuntimeError("Centerlines require MaximumInscribedSphereRadius")

    points = np.asarray([cl.GetPoint(i) for i in range(cl.GetNumberOfPoints())], dtype=np.float32)
    radii = np.asarray([radius_array.GetValue(i) for i in range(cl.GetNumberOfPoints())], dtype=np.float32)
    bounds = np.asarray(cl.GetBounds(), dtype=float).reshape(3, 2)
    margin = float(radii.max() + 2.0 * args.spacing)
    origin = bounds[:, 0] - margin
    upper = bounds[:, 1] + margin
    dims = np.ceil((upper - origin) / args.spacing).astype(int) + 1

    image = vtk.vtkImageData()
    image.SetOrigin(*origin)
    image.SetSpacing(args.spacing, args.spacing, args.spacing)
    image.SetDimensions(*map(int, dims))
    values = np.empty(int(np.prod(dims)), dtype=np.float32)
    # Chunked evaluation avoids a large temporary (voxels x centerline points).
    nx, ny, nz = map(int, dims)
    xx = origin[0] + np.arange(nx, dtype=np.float32) * args.spacing
    yy = origin[1] + np.arange(ny, dtype=np.float32) * args.spacing
    zz = origin[2] + np.arange(nz, dtype=np.float32) * args.spacing
    plane = np.stack(np.meshgrid(xx, yy, indexing="ij"), axis=-1).reshape(-1, 2)
    for k, z in enumerate(zz):
        xyz = np.column_stack([plane, np.full(len(plane), z, dtype=np.float32)])
        field = np.full(len(xyz), np.inf, dtype=np.float32)
        for start in range(0, len(points), 128):
            p = points[start:start + 128]
            r = radii[start:start + 128]
            d2 = ((xyz[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
            field = np.minimum(field, np.sqrt(d2).min(axis=1) - r[np.argmin(d2, axis=1)])
        values[k * len(plane):(k + 1) * len(plane)] = field

    vtk_values = numpy_support.numpy_to_vtk(values, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_values.SetName("PolyBallImplicitFunction")
    image.GetPointData().SetScalars(vtk_values)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, 0.0)
    mc.ComputeNormalsOn()
    mc.Update()

    args.output_vtp.parent.mkdir(parents=True, exist_ok=True)
    vtp_writer = vtk.vtkXMLPolyDataWriter()
    vtp_writer.SetFileName(str(args.output_vtp))
    vtp_writer.SetInputData(mc.GetOutput())
    vtp_writer.Write()
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(str(args.output_stl))
    stl_writer.SetInputData(mc.GetOutput())
    stl_writer.Write()
    print({"points": mc.GetOutput().GetNumberOfPoints(), "cells": mc.GetOutput().GetNumberOfCells(), "dimensions": dims.tolist(), "spacing": args.spacing})


if __name__ == "__main__":
    main()
