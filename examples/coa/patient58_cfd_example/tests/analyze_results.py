#!/usr/bin/env python3
"""
Analyse des résultats OpenFOAM après détection P04
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from foampilot.postprocess import OpenFOAMDirectReader

CASE_DIR = Path(__file__).resolve().parent.parent
TIME_STEP = "500"


def main():
    print("=" * 60)
    print("Analyse des résultats OpenFOAM")
    print("=" * 60)
    
    reader = OpenFOAMDirectReader(case_path=CASE_DIR)
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)
    
    print(f"\n[1] Maillage chargé:")
    print(f"  Points: {mesh.n_points}")
    print(f"  Cells: {mesh.n_cells}")
    
    print(f"\n[2] Conditions aux limites actuelles:")
    for name, info in reader.boundary_patches.items():
        print(f"  {name}: {info['nFaces']} faces")
    
    # Load field data
    U = mesh.cell_data["U"]
    p = mesh.cell_data["p"]
    
    print(f"\n[3] Champ de vitesse U:")
    print(f"  Magnitude min: {np.linalg.norm(U, axis=1).min():.6f}")
    print(f"  Magnitude max: {np.linalg.norm(U, axis=1).max():.6f}")
    print(f"  Magnitude mean: {np.linalg.norm(U, axis=1).mean():.6f}")
    
    print(f"\n[4] Champ de pression p:")
    print(f"  P min: {p.min():.6f}")
    print(f"  P max: {p.max():.6f}")
    print(f"  P mean: {p.mean():.6f}")
    
    # Check for convergence
    print(f"\n[5] Vérification de la convergence:")
    print(f"  Calcul terminé: {'Oui' if (CASE_DIR / str(TIME_STEP) / 'U').exists() else 'Non'}")


if __name__ == "__main__":
    main()
