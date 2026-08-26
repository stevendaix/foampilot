#!/usr/bin/env python3
"""Tutoriel 5 : Transport de scalaire passif (scalarTransport function object).

Reference OpenFOAM-13 : tutorials/incompressibleFluid/pitzDailyScalarTransport
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressibleFluid/pitzDailyScalarTransport

Écoulement turbulent dans pitzDaily avec transport d'un scalaire passif (T).
Le scalarTransport est configure comme function object dans controlDict.

Points cles :
- Solveur : incompressibleFluid (simpleFoam via foamRun -solver)
- Turbulence : laminar
- Maillage : blockMesh (canal 20x1x0.01 m)
- Fonction : scalarTransport (function object)
- Champs : U, p, T (scalaire passif)

Pipeline :
1. blockMesh -- maillage du canal
2. Setup des conditions aux limites (U, p, T)
3. Configuration du scalarTransport function object
4. Simulation + post-traitement

Usage :
    cd foampilot/tutorials/05_scalarTransport
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing
from foampilot.utilities.function import Functions


def main():
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur laminar ---
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "laminar"

    # ControlDict -- steady state
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 200.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0

    # SIMPLE
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
    }

    # Write system files
    solver.system.write()

    # --- 2. Maillage (blockMesh) ---
    # Canal: 20 x 1 x 0.01 m (flow in x, thin in z)
    bmd_mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = bmd_mesh.mesher
    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, -0.5, 0],    # 0: bottom-front-left
        [20, -0.5, 0],   # 1: bottom-front-right
        [20, 0.5, 0],    # 2: top-front-right
        [0, 0.5, 0],     # 3: top-front-left
        [0, -0.5, 0.01], # 4: bottom-back-left
        [20, -0.5, 0.01],# 5: bottom-back-right
        [20, 0.5, 0.01], # 6: top-back-right
        [0, 0.5, 0.01],  # 7: top-back-left
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (40 10 1) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [[0, 3, 7, 4]]},
        "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
        "walls": {"type": "wall", "faces": [[0, 1, 5, 4], [2, 3, 7, 6]]},
        "frontAndBack": {"type": "empty", "faces": [[0, 1, 2, 3], [4, 5, 6, 7]]},
    }
    blockmesh.mergePatchPairs = []
    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()

    # --- 3. Constant files ---
    print("2. Ecriture des proprietes physiques (laminar) ...")
    solver.constant.write()

    # --- 4. Generate 0/ field files ---
    solver.setup_case()

    # --- 5. Conditions aux limites ---
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # U -- inlet velocity 1 m/s
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (1 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "zeroGradient",
    })
    solver.boundary.set_raw_condition("walls", "U", {"type": "noSlip"})

    # p
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})
    solver.boundary.set_raw_condition("walls", "p", {"type": "zeroGradient"})

    # T (scalar passif) -- inlet 300 K, walls zeroGradient
    solver.boundary.set_raw_condition("inlet", "T", {
        "type": "fixedValue",
        "value": "uniform 300",
    })
    solver.boundary.set_raw_condition("outlet", "T", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("walls", "T", {"type": "zeroGradient"})

    # Write boundary condition files
    solver.boundary.write_boundary_conditions()

    # --- 6. Scalar transport function object ---
    # Generate the OpenFOAM 13 function object through Foampilot only.
    scalar_transport = Functions.scalar_transport(
        field="T",
        schemes_field="T",
        diffusivity="viscosity",
        alphal=1,
        alphat=0.85,
        write_control="timeStep",
        write_interval=50,
    )
    Functions.write_function_object(
        "scalarTransport",
        scalar_transport,
        case_path,
    )


    # --- 7. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleFluid -- scalarTransport)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 8. Post-traitement ---
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)
    log_file = case_path / "log.incompressibleFluid"
    if log_file.exists():
        from foampilot.utilities.residuals import ResidualsPost

        residuals = ResidualsPost(log_file)
        residuals.process(export_csv=True, export_png=True)
        print("Residus exportes (CSV + PNG).")

    times = sorted(
        [d.name for d in case_path.iterdir()
         if d.is_dir()
         and d.name not in ("constant", "system", "0", "postProcessing")
         and Functions.is_numeric(d.name)],
        key=float,
    )

    if times:
        print(f"Temps disponibles : {times}")

    print("\n" + "=" * 60)
    print("Simulation terminee avec succes !")
    print(f"  Cas      : {case_path}")
    print(f"  Log      : {case_path / 'log.incompressibleFluid'}")
    print(f"  Resultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
