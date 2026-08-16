#!/usr/bin/env python3
"""Plot the OpenFOAM buildings geometry as seen by the solver."""

import argparse
from pathlib import Path

import pyvista as pv
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing


def plot_openfoam_geometry(case_dir: Path, output_path: Path):
    """Generate a PNG of the OpenFOAM buildings geometry using PyVista."""
    foam_post = FoamPostProcessing(case_path=case_dir)

    vtk_dir = case_dir / "VTK"
    if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
        print("Running foamToVTK...")
        foam_post.foamToVTK(fields=["U", "p"])

    structure = foam_post.get_structure()
    boundaries = structure.get("boundaries", {})
    build_mesh = boundaries.get("buildings")

    if build_mesh is None or build_mesh.n_cells == 0:
        print("No buildings mesh found in OpenFOAM case.")
        return

    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")

    pl.add_mesh(
        build_mesh,
        color="#4CAF50",
        opacity=1.0,
        label="Buildings",
        show_edges=True,
        edge_color="#1B5E20",
        line_width=1.0,
    )

    pl.add_legend()
    pl.camera_position = "xy"
    pl.screenshot(str(output_path))
    pl.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot OpenFOAM buildings geometry")
    parser.add_argument("case", type=Path, help="OpenFOAM case directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    output_path = args.output or (args.case / "openfoam_geometry.png")
    plot_openfoam_geometry(args.case, output_path)


if __name__ == "__main__":
    main()
