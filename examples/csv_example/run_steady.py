#!/usr/bin/env python3
"""
Exemple stationnaire : condition aux limite CSV avec valeur constante.

Ce cas utilise le même canal 2D mais en régime stationnaire.
Le CSV est toujours lu, mais seule la première ligne est utilisée comme
valeur constante.

Usage :
    cd examples/csv_example
    python run_steady.py
"""

import math
import pandas as pd
from pathlib import Path

from foampilot.solver import Solver
from foampilot import Meshing, ValueWithUnit
from foampilot.boundaries import set_csv_condition


def generate_inlet_temperature_csv(case_path: Path) -> Path:
    """Génère un CSV de température d'entrée (pour usage stationnaire)."""
    csv_path = case_path / "inlet_temperature.csv"

    df = pd.DataFrame({
        "time_s": [0.0],
        "T_K": [350.0],
    })

    df.to_csv(csv_path, index=False, header=False)
    print(f"CSV généré : {csv_path}")
    return csv_path


def main():
    case_path = Path(__file__).resolve().parent / "case_steady"
    case_path.mkdir(parents=True, exist_ok=True)

    csv_file = generate_inlet_temperature_csv(case_path)

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = False
    solver.turbulence_model = "laminar"
    # On désactive l'énergie pour rester sur incompressibleFluid ;
    # l'exemple se concentre sur la BC CSV scalaire.
    solver.energy_activated = False

    solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")

    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 1000.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0

    solver.system.write()
    solver.constant.write()

    mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = mesh.mesher

    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],
        [2, 0, 0],
        [2, 1, 0],
        [0, 1, 0],
        [0, 0, 0.1],
        [2, 0, 0.1],
        [2, 1, 0.1],
        [0, 1, 0.1],
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (40 20 1) simpleGrading (1 1 1)"
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"defaultFaces": "empty"}
    blockmesh.boundary = {
        "inlet": {
            "type": "patch",
            "faces": [[3, 0, 4, 7]],
        },
        "outlet": {
            "type": "patch",
            "faces": [[2, 1, 5, 6]],
        },
        "top": {
            "type": "wall",
            "faces": [[2, 3, 7, 6]],
        },
        "bottom": {
            "type": "wall",
            "faces": [[0, 1, 5, 4]],
        },
        "frontAndBack": {
            "type": "empty",
            "faces": [[0, 3, 2, 1], [4, 5, 6, 7]],
        },
    }
    blockmesh.mergePatchPairs = []

    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()

    solver.boundary.initialize_boundary()

    # Température d'entrée constante (lue depuis CSV)
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

    # Vitesse à l'entrée (fixée)
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (2 0 0)",
    })

    # Pression à l'entrée (zeroGradient)
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})

    # Pression de sortie
    solver.boundary.set_raw_condition("outlet", "p", {
        "type": "fixedValue",
        "value": "uniform 0",
    })

    # Vitesse à la sortie (zeroGradient)
    solver.boundary.set_raw_condition("outlet", "U", {"type": "zeroGradient"})

    # Murs (top et bottom)
    solver.boundary.set_raw_condition("top", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("top", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("top", "T", {"type": "zeroGradient"})

    solver.boundary.set_raw_condition("bottom", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("bottom", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("bottom", "T", {"type": "zeroGradient"})

    solver.boundary.write_boundary_conditions()

    print("\n" + "=" * 60)
    print("Lancement de la simulation stationnaire")
    print("=" * 60)
    solver.run_simulation()

    print("\n" + "=" * 60)
    print("Simulation stationnaire terminée !")
    print(f"Cas : {case_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
