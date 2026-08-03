#!/usr/bin/env python3
"""
Exemple de condition aux limite CSV spatiale avec interpolation.

Ce cas démontre l'utilisation de ``set_spatial_csv_condition`` pour appliquer
une distribution spatiale de température définie par des points source,
interpolée sur le maillage du patch.

Cas :
    - Canal 2D (L=2m, H=1m, W=0.1m)
    - Solveur : incompressibleFluid (transitoire, laminaire)
    - Température spatiale interpolée depuis un nuage de points CSV
    - Post-traitement pour vérifier les fichiers T nonuniform

Usage :
    cd examples/csv_example
    python run_spatial.py
"""

import math
import pandas as pd
from pathlib import Path

from foampilot.solver import Solver
from foampilot import Meshing, ValueWithUnit
from foampilot.boundaries import set_spatial_csv_condition


def generate_spatial_csv(case_path: Path, steady: bool = False) -> Path:
    """Génère un CSV de température spatiale (nuage de points).

    Le CSV contient des points (x, y, z) sur le plan z=0.05 avec une
    température variable dans le temps.

    Parameters
    ----------
    case_path : Path
        Répertoire du cas.
    steady : bool
        Si True, CSV à un seul pas de temps (stationnaire).

    Returns
    -------
    Path
        Chemin vers le fichier CSV généré.
    """
    csv_path = case_path / "inlet_temperature_spatial.csv"

    if steady:
        times = [0.0]
    else:
        times = [0.0, 0.1, 0.2]

    rows = []
    nx, ny = 10, 5
    for t in times:
        for i in range(nx):
            x = 0.0 + i * (2.0 / (nx - 1))
            for j in range(ny):
                y = 0.0 + j * (1.0 / (ny - 1))
                z = 0.05
                temp = 300 + 20 * (x / 2.0) + 10 * (y / 1.0) + 15 * math.sin(2 * math.pi * t)
                rows.append({
                    "time_s": round(t, 3),
                    "x": x,
                    "y": y,
                    "z": z,
                    "T_K": temp,
                })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, header=False)
    print(f"CSV spatial généré : {csv_path}")
    return csv_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CSV spatial demo")
    parser.add_argument("--steady", action="store_true", help="Lancer la version stationnaire")
    args = parser.parse_args()

    case_name = "case_spatial_steady" if args.steady else "case_spatial"
    case_path = Path(__file__).resolve().parent / case_name
    case_path.mkdir(parents=True, exist_ok=True)

    csv_file = generate_spatial_csv(case_path, steady=args.steady)

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = not args.steady
    solver.turbulence_model = "laminar"
    solver.energy_activated = True

    solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
    solver.constant.transportProperties.Pr = 0.85

    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 0.25 if not args.steady else 1000.0
    solver.system.controlDict.deltaT = 0.01 if not args.steady else 1.0
    solver.system.controlDict.writeControl = "runTime" if not args.steady else "timeStep"
    solver.system.controlDict.writeInterval = 0.05 if not args.steady else 100
    solver.system.controlDict.adjustTimeStep = not args.steady
    solver.system.controlDict.maxCo = 0.5 if not args.steady else 1.0
    solver.system.controlDict.purgeWrite = 0

    if not args.steady:
        solver.system.fvSolution.PIMPLE.update({
            "nOuterCorrectors": 2,
            "nCorrectors": 1,
            "nNonOrthogonalCorrectors": 0,
        })

    solver.system.write()
    solver.constant.write()

    mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = mesh.mesher

    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
        [0, 0, 0.1], [2, 0, 0.1], [2, 1, 0.1], [0, 1, 0.1],
    ]
    blockmesh.blocks = ["hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1 1 1)"]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"defaultFaces": "empty"}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [[3, 0, 4, 7]]},
        "outlet": {"type": "patch", "faces": [[2, 1, 5, 6]]},
        "top": {"type": "wall", "faces": [[2, 3, 7, 6]]},
        "bottom": {"type": "wall", "faces": [[0, 1, 5, 4]]},
        "frontAndBack": {"type": "empty", "faces": [[0, 3, 2, 1], [4, 5, 6, 7]]},
    }
    blockmesh.mergePatchPairs = []

    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()

    solver.boundary.initialize_boundary()

    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (1 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "p", {
        "type": "fixedValue",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("outlet", "U", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "T", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("top", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("top", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("top", "T", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("bottom", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("bottom", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("bottom", "T", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "U", {"type": "empty"})
    solver.boundary.set_raw_condition("frontAndBack", "p", {"type": "empty"})
    solver.boundary.set_raw_condition("frontAndBack", "T", {"type": "empty"})

    solver.boundary.set_raw_condition("inlet", "T", {
        "type": "fixedValue",
        "value": "uniform 0",
    })

    solver.boundary.write_boundary_conditions()

    set_spatial_csv_condition(
        boundary=solver.boundary,
        patch_name="inlet",
        field="T",
        data=csv_file,
        time_column=0,
        spatial_columns=[1, 2, 3, 4],
        header_lines=0,
        separator=",",
        default_value=300,
        interpolation_method="nearest",
    )

    mode = "stationnaire" if args.steady else "transitoire"
    print(f"\n{'=' * 60}")
    print(f"Lancement de la simulation spatiale {mode}")
    print(f"{'=' * 60}")
    solver.run_simulation()

    print(f"\n{'=' * 60}")
    print("Post-traitement")
    print(f"{'=' * 60}")

    times = sorted([
        d.name for d in case_path.iterdir()
        if d.is_dir() and d.name.replace(".", "", 1).isdigit()
    ], key=lambda x: float(x))

    print(f"\nPas de temps disponibles : {times}")

    for t in times:
        t_file = case_path / t / "T"
        if t_file.exists():
            content = t_file.read_text()
            has_nonuniform = "nonuniform" in content
            print(f"  t={t}s : {'nonuniformList' if has_nonuniform else 'uniform'}")

    print(f"\n{'=' * 60}")
    print(f"Simulation spatiale {mode} terminée !")
    print(f"Cas : {case_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
