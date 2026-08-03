#!/usr/bin/env python3
"""
Post-traitement PyVista pour les exemples CSV foampilot.

Ce script vérifie que les champs CSV (uniformes et spatiaux) sont correctement
lus par OpenFOAM et visualisables avec PyVista.

Usage :
    cd examples/csv_example
    python post_csv.py [--case-dir case_uniform_scalar]
"""

import argparse
from pathlib import Path
import sys

# Add foampilot src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "foampilot" / "src"))

from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description="Post-traitement PyVista pour exemples CSV")
    parser.add_argument("--case-dir", default="case_uniform_scalar", help="Répertoire du cas")
    parser.add_argument("--field", default="T", help="Champ à visualiser")
    parser.add_argument("--time-step", type=int, default=-1, help="Pas de temps (-1 = dernier)")
    args = parser.parse_args()

    case_path = Path(__file__).resolve().parent / args.case_dir
    if not case_path.exists():
        print(f"Erreur : le cas {case_path} n'existe pas")
        sys.exit(1)

    print(f"Post-traitement du cas : {case_path}")
    print(f"Champ : {args.field}")

    post = FoamPostProcessing(str(case_path))

    # Convertir en VTK si nécessaire
    vtk_dir = case_path / "VTK"
    if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
        print("Conversion foamToVTK en cours...")
        post.foamToVTK()
    else:
        print("Fichiers VTK déjà présents")

    # Lister les pas de temps
    time_steps = post.list_time_steps()
    if not time_steps:
        print("Aucun pas de temps trouvé dans VTK")
        sys.exit(1)

    print(f"Pas de temps disponibles : {time_steps}")

    # Sélectionner le pas de temps
    if args.time_step == -1:
        time_step = time_steps[-1]
    else:
        time_step = args.time_step

    print(f"Pas de temps sélectionné : {time_step}")

    # Charger la structure
    structure = post.get_structure(time_step=time_step)

    # Vérifier le champ dans le volume
    cell_mesh = structure["cell"]
    if args.field in cell_mesh.point_data:
        field_data = cell_mesh.point_data[args.field]
        print(f"\nChamp '{args.field}' dans le volume :")
        print(f"  Type : {type(field_data)}")
        print(f"  Shape : {field_data.shape}")
        if field_data.ndim == 1:
            print(f"  Min : {field_data.min():.4f}")
            print(f"  Max : {field_data.max():.4f}")
            print(f"  Mean : {field_data.mean():.4f}")
        else:
            print(f"  Min : {field_data.min(axis=0)}")
            print(f"  Max : {field_data.max(axis=0)}")
            print(f"  Mean : {field_data.mean(axis=0)}")
    else:
        print(f"\nChamp '{args.field}' NON trouvé dans le volume")
        print(f"Champs disponibles : {list(cell_mesh.point_data.keys())}")

    # Vérifier le champ dans les boundaries
    print(f"\nBoundaries disponibles : {list(structure['boundaries'].keys())}")
    for bname, bmesh in structure["boundaries"].items():
        if args.field in bmesh.point_data:
            bdata = bmesh.point_data[args.field]
            print(f"  {bname} : {bdata.shape}, min={bdata.min():.4f}, max={bdata.max():.4f}")
        else:
            print(f"  {bname} : champ '{args.field}' non trouvé")

    # Plot slice
    try:
        post.plot_slice(
            structure=structure,
            plane="z",
            scalars=args.field,
            path_filename=str(case_path / f"slice_{args.field}_t{time_step}.png"),
        )
        print(f"\nSlice sauvegardé : {case_path / f'slice_{args.field}_t{time_step}.png'}")
    except Exception as e:
        print(f"Erreur lors du plot : {e}")


if __name__ == "__main__":
    main()
