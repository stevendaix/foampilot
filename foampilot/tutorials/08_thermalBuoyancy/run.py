#!/usr/bin/env python3
"""Tutoriel 8 : Convection thermique naturelle (buoyantSimpleFoam).

Référence OpenFOAM-13 : tutorials/fluid/buoyantCavity
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/fluid/buoyantCavity

Cet exemple simule la convection naturelle dans une pièce chauffée
avec le solveur compressible Boussinesq (fluid / buoyantSimpleFoam).

Points clés :
- Solveur : fluid (compressible Boussinesq) — necessite compressible=True
- Turbulence : kOmegaSST (RAS)
- Gravité : (0 0 -9.81) m/s² — agit sur l'écoulement par effet buoyant
- Thermophysique : Boussinesq (rho0, T0, beta)
- Champs : T, U, p_rgh, p, k, omega, nut, alphat

Chaque tâche :
1. blockMesh — génération du maillage de la pièce
2. Solver configuration — controlDict, fvSolution, fvSchemes
3. Constant files — momentumTransport, physicalProperties (Boussinesq)
4. Boundary conditions — hot/cold/adia walls
5. Simulation + post-traitement

Usage :
    cd foampilot/tutorials/08_thermalBuoyancy
    python run.py
"""

import sys
import os
from pathlib import Path

# Add src to path for tutorial execution from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver, OpenFOAMEnvironment
from foampilot import Meshing
from foampilot.utilities.function import Functions


def main():
    os.environ.update(OpenFOAMEnvironment().environment())
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur compressible Boussinesq ---
    solver = Solver(case_path)
    solver.compressible = True      # Boussinesq / fluid solver
    solver.with_gravity = True      # Gravité pour couplage thermo-fluid
    solver.turbulence_model = "kOmegaSST"
    solver.transient = False       # Stationnaire (SIMPLE)
    solver.fields_manager.register_field("k", 3.75e-4, "m^2/s^2")
    solver.fields_manager.register_field("omega", 0.12, "1/s")
    solver.fields_manager.register_field("nut", 0.0, "m^2/s")

    # Configure physicalProperties for Boussinesq
    solver.constant.physicalProperties.boussinesq = True
    solver.constant.physicalProperties.energy = True
    solver.constant.physicalProperties.Cp = 1004.4      # J/kg/K (air)
    solver.constant.physicalProperties.molWeight = 28.9 # kg/kmol (air)
    solver.constant.physicalProperties.rho0 = 1.225     # kg/m^3 (air à 15°C)
    solver.constant.physicalProperties.T0 = 288.15      # K (température de référence)
    solver.constant.physicalProperties.beta = 0.003     # 1/K (coefficient de dilatation)
    solver.constant.physicalProperties.mu = 1.831e-5    # Pa.s (viscosité dynamique)

    # --- 2. Configuration du controlDict ---
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 1000.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0

    # SIMPLE — steady-state
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p_rgh": "1e-4",
        "U": "1e-4",
        "h": "1e-4",
        "(k|epsilon|omega)": "1e-3",
    }

    # Relaxation factors (match buoyantCavity reference)
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"rho": "1.0", "p_rgh": "0.7"},
        "equations": {"U": "0.3", "h": "0.3", "(k|epsilon|omega)": "0.7"},
    }

    # Write system files
    solver.system.write()

    # --- 3. Maillage (blockMesh) ---
    # Référence OpenFOAM 13 : cavité 76 x 2180 x 520 mm.
    mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = mesh.mesher

    blockmesh.scale = 0.001
    blockmesh.vertices = [
        [0, 0, -260], [76, 0, -260], [76, 2180, -260], [0, 2180, -260],
        [0, 0, 260], [76, 0, 260], [76, 2180, 260], [0, 2180, 260],
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (35 150 15) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {}
    blockmesh.boundary = {
        "topAndBottom": {"type": "wall", "faces": [[0, 1, 5, 4], [2, 3, 7, 6]]},
        "frontAndBack": {"type": "wall", "faces": [[4, 5, 6, 7], [3, 2, 1, 0]]},
        "hot": {"type": "wall", "faces": [[6, 5, 1, 2]]},
        "cold": {"type": "wall", "faces": [[4, 7, 3, 0]]},
    }
    blockmesh.mergePatchPairs = []

    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()

    # --- 4. Écrire les fichiers constants ---
    print("2. Ecriture des proprietes physiques (Boussinesq kOmegaSST) ...")
    solver.constant.write()

    # --- 5. Generate 0/ field files (initial conditions) ---
    solver.setup_case()

    # --- 6. Conditions aux limites ---
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # Get all wall-type patches
    wall_patches = ["topAndBottom", "frontAndBack", "hot", "cold"]

    # U — no-slip on all walls (natural convection, no inlet)
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "U", {
            "type": "noSlip",
        })

    # T — hot wall 307.75K (34.6°C), cold wall 288.15K (15°C), others adiabatic
    solver.boundary.set_raw_condition("hot", "T", {
        "type": "fixedValue",
        "value": "uniform 307.75",
    })
    solver.boundary.set_raw_condition("cold", "T", {
        "type": "fixedValue",
        "value": "uniform 288.15",
    })
    for patch in ("topAndBottom", "frontAndBack"):
        solver.boundary.set_raw_condition(patch, "T", {
            "type": "zeroGradient",
        })

    # p_rgh — fixedFluxPressure on all walls
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "p_rgh", {
            "type": "fixedFluxPressure",
            "value": "$internalField",
        })

    # p — calculated on all walls (handled internally by Boussinesq)
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "p", {
            "type": "calculated",
            "value": "$internalField",
        })

    # k — kqRWallFunction on all walls
    k_val = 3.75e-04
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "k", {
            "type": "kqRWallFunction",
            "value": f"uniform {k_val}",
        })

    # omega — omegaWallFunction on all walls
    omega_val = 0.12
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "omega", {
            "type": "omegaWallFunction",
            "value": f"uniform {omega_val}",
        })

    # nut — nutUWallFunction on all walls
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "nut", {
            "type": "nutUWallFunction",
            "value": "uniform 0",
        })

    # alphat — compressible::alphatWallFunction on all walls
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "alphat", {
            "type": "compressible::alphatWallFunction",
            "Prt": "0.85",
            "value": "uniform 0",
        })

    # Write boundary condition files — OpenFOAMFile.write_boundary_file adds
    # #includeEtc "caseDicts/setConstraintTypes" by default.
    solver.boundary.write_boundary_conditions()

    # --- 7. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (fluid — thermalBuoyancy)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 8. Post-traitement ---
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)
    log_file = case_path / "log.fluid"
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
    print(f"  Log      : {case_path / 'log.fluid'}")
    print(f"  Resultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
