#!/usr/bin/env python3
"""
Visualisation post-traitement pour vérifier que les conditions aux limites
CSV sont bien appliquées.

Génère des slices/contours pour chaque cas.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "foampilot" / "src"))

from foampilot.postprocess import FoamPostProcessing
import pyvista as pv

pv.set_plot_theme("document")
pv.global_theme.background = "white"
pv.global_theme.color = "black"


def visualize_case(case_path: Path, field: str, title: str, cmap: str = "viridis"):
    """Create slice visualization for a case."""
    print(f"\n{'='*60}")
    print(f"Visualisation : {case_path.name} / {field}")
    print(f"{'='*60}")

    post = FoamPostProcessing(str(case_path))

    # Convert to VTK if needed
    vtk_dir = case_path / "VTK"
    if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
        print("  Conversion foamToVTK...")
        post.foamToVTK()
    else:
        print("  VTK déjà présents")

    time_steps = post.list_time_steps()
    if not time_steps:
        print("  ERREUR: aucun pas de temps")
        return False

    # Use last time step
    ts = time_steps[-1]
    print(f"  Pas de temps : {ts}")

    structure = post.get_structure(time_step=ts)
    cell_mesh = structure["cell"]

    if field not in cell_mesh.point_data:
        print(f"  ERREUR: champ '{field}' non trouvé")
        print(f"  Champs disponibles: {list(cell_mesh.point_data.keys())}")
        return False

    data = cell_mesh.point_data[field]
    print(f"  Shape: {data.shape}")
    print(f"  Min: {data.min():.4f}, Max: {data.max():.4f}")

    # Create output directory
    viz_dir = case_path / "visualisations"
    viz_dir.mkdir(exist_ok=True)

    # Slice at z=0.5 (middle of channel)
    bounds = cell_mesh.bounds
    z_mid = (bounds[4] + bounds[5]) / 2.0
    print(f"  Slice at z={z_mid:.2f}")

    try:
        slice_mesh = cell_mesh.slice(normal="z", origin=(0, 0, z_mid))
        print(f"  Slice: {slice_mesh.n_points} points, {slice_mesh.n_cells} cells")

        if slice_mesh.n_points > 0:
            pl = pv.Plotter(off_screen=True, window_size=(1200, 800))
            pl.add_mesh(
                slice_mesh,
                scalars=field,
                show_scalar_bar=True,
                cmap=cmap,
            )
            pl.camera_position = "xy"
            output_file = viz_dir / f"slice_{field}_t{ts}.png"
            pl.screenshot(str(output_file))
            print(f"  Slice sauvegardé: {output_file}")
            pl.close()
        else:
            print("  Slice vide")
    except Exception as e:
        print(f"  ERREUR slice: {e}")

    # Contour plot
    try:
        pl2 = pv.Plotter(off_screen=True, window_size=(1200, 800))
        pl2.add_mesh(
            cell_mesh,
            scalars=field,
            show_scalar_bar=True,
            cmap=cmap,
        )
        pl2.camera_position = "xy"
        output_file2 = viz_dir / f"contour_{field}_t{ts}.png"
        pl2.screenshot(str(output_file2))
        print(f"  Contour sauvegardé: {output_file2}")
        pl2.close()
    except Exception as e:
        print(f"  ERREUR contour: {e}")

    # For vector fields, also plot magnitude
    if field == "U" and data.ndim == 2:
        try:
            mag = np.linalg.norm(data, axis=1)
            cell_mesh.point_data["U_magnitude"] = mag

            pl3 = pv.Plotter(off_screen=True, window_size=(1200, 800))
            pl3.add_mesh(
                cell_mesh,
                scalars="U_magnitude",
                show_scalar_bar=True,
                cmap="viridis",
            )
            pl3.camera_position = "xy"
            output_file3 = viz_dir / f"contour_U_magnitude_t{ts}.png"
            pl3.screenshot(str(output_file3))
            print(f"  Magnitude sauvegardé: {output_file3}")
            pl3.close()
        except Exception as e:
            print(f"  ERREUR magnitude: {e}")

    return True


def main():
    base = Path(__file__).resolve().parent
    examples_dir = base

    cases = [
        ("case_uniform_scalar_steady", "T", "Température - Uniforme Stationnaire", "coolwarm"),
        ("case_uniform_scalar", "T", "Température - Uniforme Transitoire", "coolwarm"),
        ("case_uniform_vector", "U", "Vitesse - Uniforme Transitoire", "viridis"),
        ("case_spatial", "T", "Température - Spatial Transitoire", "coolwarm"),
        ("case_spatial_steady", "T", "Température - Spatial Stationnaire", "coolwarm"),
    ]

    results = {}
    for case_name, field, title, cmap in cases:
        case_path = examples_dir / case_name
        if case_path.exists():
            try:
                ok = visualize_case(case_path, field, title, cmap)
                results[case_name] = "OK" if ok else "FAIL"
            except Exception as e:
                print(f"ERREUR sur {case_name}: {e}")
                results[case_name] = f"ERROR: {e}"
        else:
            print(f"Cas non trouvé: {case_path}")
            results[case_name] = "MISSING"

    print(f"\n{'='*60}")
    print("Résumé")
    print(f"{'='*60}")
    for case_name, status in results.items():
        print(f"  {case_name}: {status}")

    all_ok = all(s == "OK" for s in results.values())
    if all_ok:
        print("\nToutes les visualisations ont été générées !")
    else:
        print("\nCertaines visualisations ont échoué.")
        sys.exit(1)


if __name__ == "__main__":
    main()
