#!/usr/bin/env python3
"""Exemple minimal de lecture directe d'un cas OpenFOAM avec PyVista.

Ce script montre comment utiliser ``OpenFOAMDirectReader`` et
``CHTDirectReader`` pour charger un cas OpenFOAM **sans** passer par
``foamToVTK``.  Les maillages et champs sont lus directement depuis
``constant/polyMesh`` et les répertoires de temps, puis convertis en
objets PyVista pour la visualisation.

Cas mono-région : ``planarPoiseuille``
Cas CHT        : ``simple_heated_duct``
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.postprocess import (
    CHTDirectReader,
    OpenFOAMDirectReader,
    read_openfoam,
    read_cht_openfoam,
)

FOAMPILOT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# 1. Mono-région : lecture directe du maillage + champ vitesse
# ---------------------------------------------------------------------------
def example_single_region():
    case_path = FOAMPILOT_ROOT / "planarPoiseuille"
    print("=== Mono-région :", case_path, "===")

    reader = OpenFOAMDirectReader(case_path)
    mesh = reader.to_pyvista(fields=["U"], time_step="1")

    print(f"Points: {mesh.n_points}, Cells: {mesh.n_cells}")
    print(f"Patches: {list(reader.boundary_patches.keys())}")
    print(f"Time steps: {reader.get_time_steps()}")

    if "U" in mesh.cell_data:
        U = mesh.cell_data["U"]
        print(f"U shape: {U.shape}")
        print(f"|U| range: {np.linalg.norm(U, axis=1).min():.4f} -> "
              f"{np.linalg.norm(U, axis=1).max():.4f} m/s")

    mesh.plot(scalars="U", cmap="viridis", show_edges=True, off_screen=True)


# ---------------------------------------------------------------------------
# 2. CHT : lecture multi-régions + visualisation combinée
# ---------------------------------------------------------------------------
def example_cht():
    case_path = Path(__file__).resolve().parent
    print("\n=== CHT :", case_path, "===")

    reader = CHTDirectReader(case_path)
    print("Regions:", reader.region_names)
    print("Types:", reader.regions)

    mb = reader.get_all_meshes(fields=["T"], time_step="0.1")
    print(f"MultiBlock: {mb.n_blocks} blocks")

    for name in mb.keys():
        block = mb[name]
        print(f"  {name}: {block.n_points} points, {block.n_cells} cells")
        if "T" in block.cell_data:
            T = block.cell_data["T"]
            print(f"    T range: {T.min():.2f} K -> {T.max():.2f} K")

    pl = pv.Plotter(off_screen=True)
    for name in mb.keys():
        pl.add_mesh(
            mb[name],
            scalars="T",
            lighting=False,
            scalar_bar_args={"title": f"T ({name})"},
            cmap="coolwarm",
            opacity=0.8,
        )
    pl.screenshot("cht_temperature_direct.png")
    pl.clear()
    print("Screenshot: cht_temperature_direct.png")

    temps = reader.get_interface_temperatures("fluid_to_solid", time_step="0.1")
    print("Interface temps:", temps)


# ---------------------------------------------------------------------------
# 3. Fonctions de convenance
# ---------------------------------------------------------------------------
def example_convenience():
    case_path = Path(__file__).resolve().parent
    print("\n=== Fonctions de convenance ===")

    mesh = read_openfoam(
        FOAMPILOT_ROOT / "planarPoiseuille",
        fields=["U"],
        time_step="1",
    )
    print(f"read_openfoam: {mesh.n_points} points, {mesh.n_cells} cells")

    mb = read_cht_openfoam(case_path, fields=["T"], time_step="0.1")
    print(f"read_cht_openfoam: {mb.n_blocks} blocks")


if __name__ == "__main__":
    example_single_region()
    example_cht()
    example_convenience()
