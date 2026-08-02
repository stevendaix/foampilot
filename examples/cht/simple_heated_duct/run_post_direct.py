#!/usr/bin/env python3
"""Lecture directe d'un cas OpenFOAM avec résultats depuis PyVista.

Ce script démontre l'utilisation de ``OpenFOAMDirectReader`` et
``CHTDirectReader`` pour charger un cas OpenFOAM **sans** passer par
``foamToVTK``.  Les maillages et champs sont lus directement depuis
``constant/polyMesh`` et les répertoires de temps, puis convertis en
objets PyVista pour la visualisation et l'analyse.

Cas mono-région : ``planarPoiseuille`` (OpenFOAM 13).
Cas CHT        : ``simple_heated_duct`` (OpenFOAM 13).
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "foampilot" / "src"))

from foampilot.postprocess import (
    CHTDirectReader,
    OpenFOAMDirectReader,
    read_cht_openfoam,
    read_openfoam,
)

FOAMPILOT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SINGLE_REGION_CASE = FOAMPILOT_ROOT / "planarPoiseuille"
CHT_CASE = Path(__file__).resolve().parent
OUTPUT_DIR = CHT_CASE / "postProcessing_direct"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_single_region():
    """Exemple de lecture mono-région."""
    print("=" * 60)
    print("1. LECTURE MONO-RÉGION (OpenFOAMDirectReader)")
    print("=" * 60)

    case_path = SINGLE_REGION_CASE
    reader = OpenFOAMDirectReader(case_path)
    print(f"Points: {reader.points.shape[0]}")
    print(f"Patches de boundary: {list(reader.boundary_patches.keys())}")
    print(f"Time steps disponibles: {reader.get_time_steps()}")
    print(f"Dernier time: {reader.get_latest_time()}")

    mesh = reader.to_pyvista(fields=["U"], time_step="1")
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    if "U" in mesh.cell_data:
        U = mesh.cell_data["U"]
        print(f"U (cell) shape: {U.shape}")
    elif "U" in mesh.point_data:
        U = mesh.point_data["U"]
        print(f"U (point) shape: {U.shape}")
    else:
        U = None

    if U is not None:
        U_mag = np.linalg.norm(U, axis=1)
        print(f"|U| range: {U_mag.min():.4f} - {U_mag.max():.4f} m/s")

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(
        mesh,
        scalars="U" if "U" in mesh.cell_data or "U" in mesh.point_data else None,
        lighting=False,
        scalar_bar_args={"title": "Velocity magnitude (m/s)"},
        cmap="viridis",
        show_edges=True,
    )
    pl.screenshot(str(OUTPUT_DIR / "single_region_velocity.png"), window_size=(1200, 400))
    pl.clear()
    print(f"Image sauvegardée : {OUTPUT_DIR / 'single_region_velocity.png'}")


def demo_cht():
    """Exemple de lecture CHT (multi-régions)."""
    print("\n" + "=" * 60)
    print("2. LECTURE CHT (CHTDirectReader)")
    print("=" * 60)

    reader = CHTDirectReader(CHT_CASE)
    print(f"Régions détectées: {reader.region_names}")
    print(f"Types: {reader.regions}")

    mb = reader.get_all_meshes(fields=["T"], time_step="0.1")
    print(f"MultiBlock: {mb.n_blocks} blocs")

    for name in mb.keys():
        block = mb[name]
        print(f"\n  Région '{name}': {block.n_points} pts, {block.n_cells} cells")
        if "T" in block.cell_data:
            T = block.cell_data["T"]
            print(f"    T range: {T.min():.2f} K - {T.max():.2f} K")
        if "T" in block.point_data:
            T = block.point_data["T"]
            print(f"    T range: {T.min():.2f} K - {T.max():.2f} K")

    pl = pv.Plotter(off_screen=True)
    for name in mb.keys():
        block = mb[name]
        pl.add_mesh(
            block,
            scalars="T",
            lighting=False,
            scalar_bar_args={"title": f"T ({name})"},
            cmap="coolwarm",
            opacity=0.8,
            show_edges=False,
        )
    pl.screenshot(str(OUTPUT_DIR / "cht_temperature.png"), window_size=(1200, 400))
    pl.clear()
    print(f"Image sauvegardée : {OUTPUT_DIR / 'cht_temperature.png'}")


def demo_vector_field():
    """Exemple de lecture d'un champ vectoriel (U)."""
    print("\n" + "=" * 60)
    print("3. LECTURE CHAMP VECTORIEL (U)")
    print("=" * 60)

    reader = CHTDirectReader(CHT_CASE)
    fluid_mesh = reader.get_mesh(region="fluid", fields=["U"], time_step="0.1")

    if "U" in fluid_mesh.cell_data:
        U = fluid_mesh.cell_data["U"]
    elif "U" in fluid_mesh.point_data:
        U = fluid_mesh.point_data["U"]
    else:
        U = None

    if U is not None:
        U_mag = np.linalg.norm(U, axis=1)
        print(f"U range: {U_mag.min():.4f} - {U_mag.max():.4f} m/s")

        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(
            fluid_mesh,
            scalars=U_mag,
            lighting=False,
            scalar_bar_args={"title": "|U| (m/s)"},
            cmap="viridis",
            show_edges=False,
        )
        pl.screenshot(str(OUTPUT_DIR / "velocity_magnitude.png"), window_size=(1200, 400))
        pl.clear()
        print(f"Image sauvegardée : {OUTPUT_DIR / 'velocity_magnitude.png'}")


def demo_interface_temperatures():
    """Exemple d'extraction des températures d'interface."""
    print("\n" + "=" * 60)
    print("4. TEMPÉRATURES D'INTERFACE")
    print("=" * 60)

    reader = CHTDirectReader(CHT_CASE)
    temps = reader.get_interface_temperatures(interface_name="fluid_to_solid", time_step="0.1")
    for k, v in temps.items():
        print(f"  {k}: {v:.2f} K")


def main():
    print("Cas mono-région:", SINGLE_REGION_CASE)
    print("Cas CHT:", CHT_CASE)
    print("Méthode: lecture directe (sans foamToVTK)")
    print()

    demo_single_region()
    demo_cht()
    demo_vector_field()
    demo_interface_temperatures()

    print("\n" + "=" * 60)
    print("TOUS LES RÉSULTATS ONT ÉTÉ SAUVEGARDÉS DANS:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
