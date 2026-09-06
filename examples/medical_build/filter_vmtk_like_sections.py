from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "foampilot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foampilot.geometry.medical_build.section_filtering import (  # noqa: E402
    SectionFilterConfig,
    contour_metrics,
    continuity_rejection,
)


def write_vtp(report: dict, path: Path) -> None:
    import vtk

    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    status = vtk.vtkIntArray(); status.SetName("SectionStatusCode")
    branch_ids = vtk.vtkIntArray(); branch_ids.SetName("BranchId")
    station_ids = vtk.vtkIntArray(); station_ids.SetName("CenterlinePointId")
    radius = vtk.vtkDoubleArray(); radius.SetName("RadiusMedian")
    shape = vtk.vtkDoubleArray(); shape.SetName("ShapeIndex")
    area = vtk.vtkDoubleArray(); area.SetName("SectionArea")
    code = {"VALID": 1, "JUNCTION": 2, "REJECTED": 3}
    for branch in report["branches"]:
        for section in branch["sections"]:
            p = np.asarray(section["points"], dtype=float)
            ids = vtk.vtkIdList()
            for point in p:
                ids.InsertNextId(points.InsertNextPoint(*map(float, point)))
            polys.InsertNextCell(ids)
            status.InsertNextValue(code.get(section["status"], 0))
            branch_ids.InsertNextValue(int(branch["branch_id"]))
            station_ids.InsertNextValue(int(section["centerline_point_id"]))
            radius.InsertNextValue(float(section["radius_median"]))
            shape.InsertNextValue(float(section["shape"]))
            area.InsertNextValue(float(section["area"]))
    output = vtk.vtkPolyData(); output.SetPoints(points); output.SetPolys(polys)
    for array in (status, branch_ids, station_ids, radius, shape, area):
        output.GetCellData().AddArray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(path)); writer.SetInputData(output); writer.Write()


def make_image(report: dict, surface_path: Path, centerlines_path: Path, path: Path) -> None:
    import pyvista as pv

    surface = pv.read(surface_path)
    centerlines = pv.read(centerlines_path)
    contour_mesh = pv.read(path.with_suffix(".vtp"))
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    plotter.set_background("white")
    plotter.add_mesh(surface, color="lightgray", opacity=0.18, show_edges=False)
    plotter.add_mesh(centerlines, color="black", line_width=3)
    colors = {1: "green", 2: "orange", 3: "red"}
    for value, color in colors.items():
        ids = np.flatnonzero(contour_mesh.cell_data["SectionStatusCode"] == value)
        if len(ids) == 0:
            continue
        subset = contour_mesh.extract_cells(ids)
        plotter.add_mesh(subset, color=color, line_width=2, show_edges=True, label={1: "VALID", 2: "JUNCTION", 3: "REJECTED"}[value])
    plotter.add_legend(bcolor="white", face="rectangle")
    plotter.add_text("VMTK-like sections — green valid / orange junction / red rejected", font_size=13, color="black")
    plotter.camera_position = "iso"
    plotter.show(screenshot=str(path.with_suffix(".png")), auto_close=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--surface", type=Path, required=True)
    ap.add_argument("--centerlines", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-radius-ratio", type=float, default=1.8)
    ap.add_argument("--max-area-ratio", type=float, default=2.5)
    args = ap.parse_args()
    config = SectionFilterConfig(max_radius_ratio=args.max_radius_ratio, max_area_ratio=args.max_area_ratio)
    report = json.loads(args.input.read_text())
    counts = {"VALID": 0, "JUNCTION": 0, "REJECTED": 0}
    for branch in report["branches"]:
        sections = branch["sections"]
        candidates = [contour_metrics(np.asarray(s["points"], dtype=float), s["center"], s["tangent"]) for s in sections]
        for candidate, section in zip(candidates, sections):
            # The raw exporter removes the duplicated closing vertex. Preserve
            # VTK connectivity explicitly instead of inferring closure from
            # the first/last coordinate alone.
            candidate.closed = bool(section.get("closed", candidate.closed))
        radii = [c.radius_median for c in candidates if c.closed and c.radius_median > 0]
        expected = float(np.median(radii)) if radii else 1.0
        for i, (section, candidate) in enumerate(zip(sections, candidates)):
            previous = candidates[i - 1] if i > 0 else None
            following = candidates[i + 1] if i + 1 < len(candidates) else None
            if not candidate.closed:
                status, reason = "REJECTED", "OPEN_CONTOUR"
            elif candidate.shape < config.min_shape:
                status, reason = "REJECTED", "BAD_SHAPE"
            else:
                rejected, reason = continuity_rejection(candidate, previous, following, config)
                status = "REJECTED" if rejected else "VALID"
                if not rejected and abs(candidate.radius_median - expected) / max(expected, 1e-12) > 0.8:
                    status, reason = "JUNCTION", "GLOBAL_RADIUS_DEVIATION"
            section.update({
                "closed": bool(candidate.closed),
                "area": candidate.area,
                "perimeter": candidate.perimeter,
                "radius_min": candidate.radius_min,
                "radius_median": candidate.radius_median,
                "radius_max": candidate.radius_max,
                "shape": candidate.shape,
                "status": status,
                "rejection_reason": reason,
            })
            counts[status] += 1
    report["filter"] = {"config": vars(config), "counts": counts, "method": "local contour metrics + longitudinal continuity"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    vtp = args.output.with_suffix(".vtp")
    write_vtp(report, vtp)
    make_image(report, args.surface, args.centerlines, vtp)
    print(json.dumps({"json": str(args.output), "vtp": str(vtp), "png": str(vtp.with_suffix('.png')), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
