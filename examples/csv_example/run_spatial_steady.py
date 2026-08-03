#!/usr/bin/env python3
"""
Exemple de condition aux limite CSV spatiale stationnaire.
"""

import math
import pandas as pd
from pathlib import Path

from foampilot.solver import Solver
from foampilot import Meshing, ValueWithUnit
from foampilot.boundaries import set_spatial_csv_condition


def generate_spatial_csv(case_path: Path) -> Path:
    csv_path = case_path / "inlet_temperature_spatial.csv"
    times = [0.0]
    rows = []
    nx, ny = 10, 5
    for t in times:
        for i in range(nx):
            x = 0.0 + i * (2.0 / (nx - 1))
            for j in range(ny):
                y = 0.0 + j * (1.0 / (ny - 1))
                z = 0.05
                temp = 300 + 20 * (x / 2.0) + 10 * (y / 1.0)
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
    case_path = Path(__file__).resolve().parent / "case_spatial_steady"
    case_path.mkdir(parents=True, exist_ok=True)

    csv_file = generate_spatial_csv(case_path)

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = False
    solver.turbulence_model = "laminar"
    solver.energy_activated = True

    solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
    solver.constant.transportProperties.Pr = 0.85

    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 1000.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0

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
    solver.boundary.set_raw_condition("inlet", "T", {
        "type": "fixedValue",
        "value": "uniform 0",
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

    print("\n" + "=" * 60)
    print("Lancement de la simulation spatiale stationnaire")
    print("=" * 60)
    solver.run_simulation()

    print("\n" + "=" * 60)
    print("Simulation spatiale stationnaire terminée !")
    print(f"Cas : {case_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
