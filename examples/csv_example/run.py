#!/usr/bin/env python3
"""
Exemple d'utilisation des conditions aux limites variables dans le temps
à partir d'un fichier CSV.

Ce cas démontre l'utilisation de la fonctionnalité CSV BC de foampilot
pour appliquer une vitesse d'entrée variable dans le temps sur un canal
simple 2D.

Cas :
    - Canal 2D (L=2m, H=1m, W=0.1m)
    - Solveur : incompressibleFluid (transitoire incompressible, laminaire)
    - Vitesse d'entrée variable dans le temps (lue depuis un CSV)
    - Post-traitement pour vérifier l'évolution de la vitesse

Usage :
    cd examples/csv_example
    python run.py
"""

import math
import pandas as pd
from pathlib import Path

from foampilot.solver import Solver
from foampilot import Meshing, ValueWithUnit
from foampilot.boundaries import set_csv_condition


def generate_inlet_velocity_csv(case_path: Path) -> Path:
    """Génère un CSV de vitesse d'entrée variable dans le temps.

    La vitesse suit une variation sinusoïdale entre 1 et 3 m/s en x,
    nulle en y et z.

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
    case_path = Path(__file__).resolve().parent / "case"
    case_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Génération du CSV de vitesse d'entrée
    # ------------------------------------------------------------------
    csv_file = generate_inlet_velocity_csv(case_path)

    # ------------------------------------------------------------------
    # 2. Initialisation du solveur
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Maillage blockMesh
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Conditions aux limites
    # ------------------------------------------------------------------
    solver.boundary.initialize_boundary()

    # Vitesse d'entrée variable dans le temps (depuis CSV)
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

    solver.boundary.set_raw_condition("bottom", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("bottom", "p", {"type": "zeroGradient"})

    solver.boundary.write_boundary_conditions()

    # ------------------------------------------------------------------
    # 5. Lancement de la simulation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Lancement de la simulation")
    print("=" * 60)
    solver.run_simulation()

    # ------------------------------------------------------------------
    # 6. Post-traitement
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)

    post_process(case_path)

    print("\n" + "=" * 60)
    print("Simulation terminée avec succès !")
    print(f"Cas : {case_path}")
    print("=" * 60)


def post_process(case_path: Path):
    """Post-traitement : vérification de la vitesse au point de sortie."""
    import numpy as np

    log_file = case_path / "log.incompressibleFluid"
    if log_file.exists():
        from foampilot.utilities.residuals import ResidualsPost
        residuals = ResidualsPost(log_file)
        residuals.process(export_csv=True, export_png=True)
        print("Résidus exportés.")

    times = sorted([
        d.name for d in case_path.iterdir()
        if d.is_dir() and d.name.replace(".", "", 1).isdigit()
    ], key=lambda x: float(x))

    print(f"\nPas de temps disponibles : {times}")

    results = []
    for t in times:
        u_file = case_path / t / "U"
        if u_file.exists():
            content = u_file.read_text()
            results.append((float(t), content.count("nonuniform") > 0))

    if results:
        print("\nVérification des fichiers U :")
        for t, has_nonuniform in results:
            print(f"  t={t:.2f}s : {'nonuniformList' if has_nonuniform else 'uniform'}")


if __name__ == "__main__":
    main()
