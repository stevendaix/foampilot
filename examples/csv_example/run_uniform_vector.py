#!/usr/bin/env python3
"""
Exemple de condition aux limite CSV uniforme vectoriel.

Cas transitoire avec vitesse d'entrée variable dans le temps.

Cas :
    - Canal 2D (L=2m, H=1m, W=0.1m)
    - Solveur : incompressibleFluid (laminaire)
    - Champ vectoriel U avec BC uniformFixedValue + table CSV

Usage :
    cd examples/csv_example
    python run_uniform_vector.py
"""

import math
import pandas as pd
from pathlib import Path

from foampilot.solver import Solver
from foampilot import Meshing, ValueWithUnit
from foampilot.boundaries import set_csv_condition


def generate_vector_csv(case_path: Path) -> Path:
    """Génère un CSV de vitesse d'entrée variable dans le temps.

    Returns
    -------
    Path
        Chemin vers le fichier CSV généré.
    """
    csv_path = case_path / "inlet_velocity.csv"

    times = [round(t, 3) for t in [i * 0.1 for i in range(51)]]
    velocities = [1.0 + 1.0 * (1 + math.sin(2 * math.pi * t / 5.0)) / 2 for t in times]

    df = pd.DataFrame({
        "time_s": times,
        "Ux_ms": velocities,
        "Uy_ms": [0.0] * len(times),
        "Uz_ms": [0.0] * len(times),
    })

    df.to_csv(csv_path, index=False, header=False)
    print(f"CSV généré : {csv_path}")
    return csv_path


def main():
    case_path = Path(__file__).resolve().parent / "case_uniform_vector"
    case_path.mkdir(parents=True, exist_ok=True)

    csv_file = generate_vector_csv(case_path)

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = True
    solver.turbulence_model = "laminar"

    solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")

    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 5.0
    solver.system.controlDict.deltaT = 0.01
    solver.system.controlDict.writeControl = "runTime"
    solver.system.controlDict.writeInterval = 0.1
    solver.system.controlDict.adjustTimeStep = True
    solver.system.controlDict.maxCo = 0.5
    solver.system.controlDict.purgeWrite = 0

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

    solver.boundary.initialize_boundary()

    set_csv_condition(
        boundary=solver.boundary,
        patch_name="inlet",
        field="U",
        data=csv_file,
        time_column=0,
        value_columns=[1, 2, 3],
        header_lines=0,
        separator=",",
        out_of_bounds="clamp",
        interpolation_scheme="linear",
        default_value="(1 0 0)",
    )

    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {
        "type": "fixedValue",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("outlet", "U", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("top", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("top", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("bottom", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("bottom", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "U", {"type": "empty"})
    solver.boundary.set_raw_condition("frontAndBack", "p", {"type": "empty"})

    solver.boundary.write_boundary_conditions()

    print("\n" + "=" * 60)
    print("Lancement de la simulation (vectoriel uniforme)")
    print("=" * 60)
    solver.run_simulation()

    print("\n" + "=" * 60)
    print("Simulation vectorielle terminée !")
    print(f"Cas : {case_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
