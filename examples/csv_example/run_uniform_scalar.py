#!/usr/bin/env python3
"""
Exemple de condition aux limite CSV uniforme scalaire.

Deux variantes sont présentées :
- Stationnaire : CSV à une ligne, valeur constante
- Transitoire : CSV avec plusieurs pas de temps, interpolation linéaire

Cas :
    - Canal 2D (L=2m, H=1m, W=0.1m)
    - Solveur : incompressibleFluid (laminaire)
    - Champ scalaire T avec BC uniformFixedValue + table CSV

Usage :
    cd examples/csv_example
    python run_uniform_scalar.py
"""

import math
import pandas as pd
from pathlib import Path

from foampilot.solver import Solver
from foampilot import Meshing, ValueWithUnit
from foampilot.boundaries import set_csv_condition


def generate_scalar_csv(case_path: Path, steady: bool = False) -> Path:
    """Génère un CSV de température d'entrée.

    Parameters
    ----------
    case_path : Path
        Répertoire du cas.
    steady : bool
        Si True, CSV à une ligne (stationnaire).
        Si False, CSV avec pas de temps multiples (transitoire).

    Returns
    -------
    Path
        Chemin vers le fichier CSV généré.
    """
    csv_path = case_path / "inlet_temperature.csv"

    if steady:
        df = pd.DataFrame({
            "time_s": [0.0],
            "T_K": [350.0],
        })
    else:
        times = [round(t, 3) for t in [i * 0.1 for i in range(51)]]
        temps = [350 + 50 * math.sin(2 * math.pi * t / 5.0) for t in times]
        df = pd.DataFrame({
            "time_s": times,
            "T_K": temps,
        })

    df.to_csv(csv_path, index=False, header=False)
    print(f"CSV généré : {csv_path}")
    return csv_path


def setup_case(case_path: Path, transient: bool = True) -> Solver:
    """Configure le cas de base.

    Parameters
    ----------
    case_path : Path
        Répertoire du cas.
    transient : bool
        Si True, simulation transitoire.

    Returns
    -------
    Solver
        Objet solveur configuré.
    """
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = transient
    solver.turbulence_model = "laminar"
    solver.energy_activated = True

    solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
    solver.constant.transportProperties.Pr = 0.85

    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 5.0 if transient else 1000.0
    solver.system.controlDict.deltaT = 0.01 if transient else 1.0
    solver.system.controlDict.writeControl = "runTime" if transient else "timeStep"
    solver.system.controlDict.writeInterval = 0.1 if transient else 100
    solver.system.controlDict.adjustTimeStep = transient
    solver.system.controlDict.maxCo = 0.5 if transient else 1.0
    solver.system.controlDict.purgeWrite = 0

    if transient:
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
    blockmesh.blocks = ["hex (0 1 2 3 4 5 6 7) (40 20 1) simpleGrading (1 1 1)"]
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

    return solver


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CSV uniform scalaire demo")
    parser.add_argument("--steady", action="store_true", help="Lancer la version stationnaire")
    args = parser.parse_args()

    case_name = "case_uniform_scalar_steady" if args.steady else "case_uniform_scalar"
    case_path = Path(__file__).resolve().parent / case_name
    case_path.mkdir(parents=True, exist_ok=True)

    csv_file = generate_scalar_csv(case_path, steady=args.steady)
    solver = setup_case(case_path, transient=not args.steady)

    solver.boundary.initialize_boundary()

    set_csv_condition(
        boundary=solver.boundary,
        patch_name="inlet",
        field="T",
        data=csv_file,
        time_column=0,
        value_column=1,
        header_lines=0,
        separator=",",
        out_of_bounds="clamp",
        interpolation_scheme="linear",
        default_value=350,
    )

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

    solver.boundary.write_boundary_conditions()

    mode = "stationnaire" if args.steady else "transitoire"
    print(f"\n{'=' * 60}")
    print(f"Lancement de la simulation {mode} (scalaire uniforme)")
    print(f"{'=' * 60}")
    solver.run_simulation()

    print(f"\n{'=' * 60}")
    print(f"Simulation {mode} terminée !")
    print(f"Cas : {case_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
