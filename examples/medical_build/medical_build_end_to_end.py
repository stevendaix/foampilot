"""End-to-end medical_build example.

The script consumes a contract JSON produced by the analysis phase and writes
portable artifacts. Heavy CAD and OpenFOAM steps are optional so the example
also runs in a minimal Python environment.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np


def load_contract(path: Path) -> dict:
    data=json.loads(path.read_text())
    if not data.get("branches"):
        raise ValueError("The contract must contain at least one branch")
    return data


def _pad_sections(sections: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a lossless rectangular representation and the original lengths.

    Medical contours are not required to have the same number of samples at
    every station.  NaN padding keeps the NPZ format compatible with numeric
    NumPy consumers; ``section_lengths`` identifies the valid prefix of every
    row and prevents padded values from being interpreted as geometry.
    """
    values = [np.asarray(section[key], dtype=float) for section in sections]
    lengths = np.asarray([len(value) for value in values], dtype=np.int64)
    if not values:
        return np.empty((0, 0), dtype=float), lengths
    width = int(lengths.max())
    padded = np.full((len(values), width, *values[0].shape[1:]), np.nan, dtype=float)
    for index, value in enumerate(values):
        padded[index, : len(value), ...] = value
    return padded, lengths


def export_branch_npz(branch: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sections = branch.get("sections", [])
    section_points = [
        section.get("phase_locked_points") or section.get("points")
        for section in sections
    ]
    points, section_lengths = _pad_sections(
        [{"value": value} for value in section_points], "value"
    )
    section_abscissas, abscissa_lengths = _pad_sections(
        [{"value": section["abscissa"]} for section in sections], "value"
    )
    if not np.array_equal(section_lengths, abscissa_lengths):
        raise ValueError("section points and abscissas must have matching lengths")
    np.savez_compressed(
        out / f"branch_{int(branch['branch_id']):02d}.npz",
        points=np.asarray(branch["points"], dtype=float),
        abscissas=np.asarray(branch["abscissas"], dtype=float),
        tangents=np.asarray(branch["tangents"], dtype=float),
        section_centers=np.asarray([s["center"] for s in sections], dtype=float),
        section_points=points,
        section_abscissas=section_abscissas,
        section_lengths=section_lengths,
    )


def export_vtp(data: dict, path: Path) -> str:
    try:
        import vtk
    except ImportError:
        return "vtk unavailable"
    points=vtk.vtkPoints(); lines=vtk.vtkCellArray()
    for branch in data["branches"]:
        ids=[]
        for p in branch["points"]: ids.append(points.InsertNextPoint(*map(float,p)))
        line=vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(len(ids))
        for i,idx in enumerate(ids): line.GetPointIds().SetId(i,idx)
        lines.InsertNextCell(line)
    poly=vtk.vtkPolyData(); poly.SetPoints(points); poly.SetLines(lines)
    writer=vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(path)); writer.SetInputData(poly); writer.Write()
    return "written"


def write_vtk_legacy(data: dict, path: Path) -> None:
    lines=[]; pts=[]; offset=0
    for branch in data["branches"]:
        p=branch["points"]; pts.extend(p); lines.append((offset, len(p))); offset += len(p)
    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\nmedical_build centerlines\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(pts)} float\n")
        for p in pts: f.write("%.9g %.9g %.9g\n"%tuple(p))
        f.write(f"LINES {len(lines)} {sum(n+1 for _,n in lines)}\n")
        for start,n in lines: f.write(str(n)+" "+" ".join(str(start+i) for i in range(n))+"\n")


def write_manifest(data: dict, out: Path, status: dict) -> None:
    manifest={"schema":"foampilot.medical_build.example.v1","source_branches":len(data["branches"]),"outputs":status}
    (out/"export_manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"export_report.md").write_text("# medical_build end-to-end export\n\n"+"\n".join(f"| {k} | {v} |" for k,v in status.items()))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("contract", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cad", action="store_true", help="attempt optional Build123d branch exports")
    ap.add_argument(
        "--openfoam",
        action="store_true",
        help="record that an external OpenFOAM case is required (does not run OpenFOAM)",
    )
    args=ap.parse_args(); start=time.perf_counter(); data=load_contract(args.contract); out=args.output; out.mkdir(parents=True,exist_ok=True); status={}
    (out/"analysis_contract.json").write_text(json.dumps(data,indent=2)); status["analysis_contract.json"]="written"
    for branch in data["branches"]: export_branch_npz(branch,out/"npz")
    status["npz"]="written"
    status["centerlines.vtp"]=export_vtp(data,out/"centerlines.vtp")
    write_vtk_legacy(data,out/"centerlines.vtk"); status["centerlines.vtk"]="written"
    # The full section VTP is intentionally generated by the section backend,
    # because arbitrary contours can have different point counts.
    status["sections.vtp"]="requires section exporter"
    if args.cad:
        try:
            import build123d  # noqa: F401
            status["cad"]="Build123d available; use Build123dReconstruction with section records"
        except ImportError: status["cad"]="skipped: build123d unavailable"
    else: status["cad"]="skipped: pass --cad"
    status["openfoam"] = (
        "not executed: use the dedicated OpenFOAM case runner"
        if args.openfoam
        else "skipped: pass --openfoam"
    )
    status["elapsed_s"]=round(time.perf_counter()-start,6); write_manifest(data,out,status)
    print(json.dumps(status,indent=2))

if __name__ == "__main__": main()
