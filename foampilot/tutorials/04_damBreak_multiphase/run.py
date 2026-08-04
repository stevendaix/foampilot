#!/usr/bin/env python3
"""Tutoriel 4 : Cas de référence damBreak — écoulement bifluide VOF (interFoam).

Référence OpenFOAM-13 : tutorials/incompressibleVoF/damBreakLaminar
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressibleVoF/damBreakLaminar

Cet exemple montre la modélisation d'un écoulement à deux phases (eau/air)
avec le modèle VOF et le solveur incompressibleVoF (interFoam).

Le cas simule la chute libre d'une colonne d'eau dans un réservoir rectangulaire,
sous l'effet de la gravité, avec tension superficielle.
"""

import sys
from pathlib import Path

# Add src to path for tutorial execution from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing
from foampilot.utilities.function import Functions


def main():
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur VOF incompressible ---
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = True       # damBreak is gravity-driven
    solver.is_vof = True
    solver.transient = True
    solver.turbulence_model = "laminar"

    # Configure VoF constant files (phaseProperties, physicalProperties.<phase>,
    # momentumTransport) via the library
    solver.constant.configure_vof(
        phases=["water", "air"],
        sigma=0.07,
        phase_properties={
            "water": {"nu": 1e-6, "rho": 1000},
            "air": {"nu": 1.48e-05, "rho": 1.0},
        },
    )

    # --- 2. Configuration du controlDict ---
    # OpenFOAM 13 uses 'solver' keyword for foamRun -solver
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 1.0
    solver.system.controlDict.deltaT = 0.001
    solver.system.controlDict.writeControl = "adjustableRunTime"
    solver.system.controlDict.writeInterval = 0.05
    solver.system.controlDict.set_adaptive_time_step(
        adjustTimeStep=True,
        maxCo=1.0,
        maxAlphaCo=1.0,
        maxDeltaT=1.0,
    )

    # PIMPLE — match OpenFOAM 13 reference
    solver.system.fvSolution.set_pimple(
        nCorrectors=3,
        nNonOrthogonalCorrectors=0,
        momentumPredictor=False,
    )
    solver.system.fvSolution.PIMPLE["consistent"] = "yes"
    solver.system.fvSolution.PIMPLE.pop("pRefCell", None)
    solver.system.fvSolution.PIMPLE.pop("pRefValue", None)

    # Remove T/energy solvers (not used in laminar VoF)
    solver.system.fvSolution.solvers.pop("T", None)
    solver.system.fvSolution.solvers.pop("TFinal", None)

    # Match OpenFOAM 13 incompressibleVoF reference solver settings
    solver.system.fvSolution.solvers.pop("p", None)
    solver.system.fvSolution.solvers.pop("pFinal", None)
    # Remove default alpha.* entries — use regex patterns instead
    solver.system.fvSolution.solvers.pop("alpha.water", None)
    solver.system.fvSolution.solvers.pop("alpha.air", None)

    # alpha.water solvers (MULES-based)
    solver.system.fvSolution.solvers["alpha.water.*"] = {
        "nCorrectors": "2",
        "nSubCycles": "1",
        "MULESCorr": "yes",
        "MULES": {"nIter": "10", "tolerance": "1e-2"},
        "solver": "smoothSolver",
        "smoother": "symGaussSeidel",
        "tolerance": "1e-8",
        "relTol": "0",
    }

    # pcorr solvers
    solver.system.fvSolution.solvers["pcorr.*"] = {
        "solver": "PCG",
        "preconditioner": "DIC",
        "tolerance": "1e-5",
        "relTol": "0",
    }

    # p_rgh solvers
    solver.system.fvSolution.solvers["p_rgh"] = {
        "solver": "PCG",
        "preconditioner": "DIC",
        "tolerance": "1e-7",
        "relTol": "0.05",
    }
    solver.system.fvSolution.solvers["p_rghFinal"] = {
        "$p_rgh": "",
        "relTol": "0",
    }

    # U solvers
    solver.system.fvSolution.solvers["U"] = {
        "solver": "smoothSolver",
        "smoother": "symGaussSeidel",
        "tolerance": "1e-6",
        "relTol": "0",
        "minIter": "1",
    }
    solver.system.fvSolution.solvers["UFinal"] = {
        "$U": "",
        "tolerance": "1e-7",
    }

    # No under-relaxation for transient PIMPLE
    solver.system.fvSolution.relaxationFactors = {
        "fields": {},
        "equations": {".*": 1},
    }

    # fvSchemes — match OpenFOAM 13 reference
    solver.system.fvSchemes.divSchemes.pop("div(phi,U)", None)
    solver.system.fvSchemes.divSchemes["div(phi,alpha)"] = "Gauss interfaceCompression vanLeer 1"
    solver.system.fvSchemes.divSchemes["div(rhoPhi,U)"] = "Gauss linearUpwind grad(U)"
    solver.system.fvSchemes.divSchemes["div(((rho*nuEff)*dev2(T(grad(U)))))"] = "Gauss linear"
    solver.system.fvSchemes.laplacianSchemes["default"] = "Gauss linear uncorrected"
    solver.system.fvSchemes.snGradSchemes["default"] = "uncorrected"

    solver.system.write()

    # --- 3. Maillage (blockMesh) ---
    # Match OpenFOAM 13 damBreakLaminar reference geometry
    # Tank: 5.5 x 0.4 x 0.1 (x y z), water column: 0.146 x 0.292 x 0.1
    mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = mesh.mesher

    blockmesh.scale = 0.146
    blockmesh.vertices = [
        [0, 0, 0],
        [2, 0, 0],
        [2.16438, 0, 0],
        [4, 0, 0],
        [0, 0.32876, 0],
        [2, 0.32876, 0],
        [2.16438, 0.32876, 0],
        [4, 0.32876, 0],
        [0, 4, 0],
        [2, 4, 0],
        [2.16438, 4, 0],
        [4, 4, 0],
        [0, 0, 0.1],
        [2, 0, 0.1],
        [2.16438, 0, 0.1],
        [4, 0, 0.1],
        [0, 0.32876, 0.1],
        [2, 0.32876, 0.1],
        [2.16438, 0.32876, 0.1],
        [4, 0.32876, 0.1],
        [0, 4, 0.1],
        [2, 4, 0.1],
        [2.16438, 4, 0.1],
        [4, 4, 0.1],
    ]
    blockmesh.blocks = [
        "hex (0 1 5 4 12 13 17 16) (23 8 1) simpleGrading (1 1 1)",
        "hex (2 3 7 6 14 15 19 18) (19 8 1) simpleGrading (1 1 1)",
        "hex (4 5 9 8 16 17 21 20) (23 42 1) simpleGrading (1 1 1)",
        "hex (5 6 10 9 17 18 22 21) (4 42 1) simpleGrading (1 1 1)",
        "hex (6 7 11 10 18 19 23 22) (19 42 1) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "leftWall": {
            "type": "wall",
            "faces": [(0, 12, 16, 4), (4, 16, 20, 8)],
        },
        "rightWall": {
            "type": "wall",
            "faces": [(7, 19, 15, 3), (11, 23, 19, 7)],
        },
        "lowerWall": {
            "type": "wall",
            "faces": [
                (0, 1, 13, 12),
                (1, 5, 17, 13),
                (5, 6, 18, 17),
                (2, 14, 18, 6),
                (2, 3, 15, 14),
            ],
        },
        "atmosphere": {
            "type": "patch",
            "faces": [
                (8, 20, 21, 9),
                (9, 21, 22, 10),
                (10, 22, 23, 11),
            ],
        },
    }
    blockmesh.mergePatchPairs = []

    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()

    # --- 4. Écrire les fichiers constants (two-phase VoF) ---
    print("2. Ecriture des proprietes physiques (two-phase VoF) ...")
    solver.constant.write()

    # --- 5. Conditions aux limites ---
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # Remove alpha.air — incompressibleVoF computes alpha.air = 1 - alpha.water implicitly
    solver.boundary.fields.pop("alpha.air", None)
    if "alpha.air" in solver._solver.fields_manager.fields:
        solver._solver.fields_manager.fields.pop("alpha.air")

    # alpha.water — boundary conditions
    solver.boundary.set_raw_condition("atmosphere", "alpha.water", {
        "type": "inletOutlet",
        "inletValue": "$internalField",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("leftWall", "alpha.water", {
        "type": "zeroGradient",
    })
    solver.boundary.set_raw_condition("rightWall", "alpha.water", {
        "type": "zeroGradient",
    })
    solver.boundary.set_raw_condition("lowerWall", "alpha.water", {
        "type": "zeroGradient",
    })

    # U — boundary conditions
    solver.boundary.set_raw_condition("atmosphere", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("leftWall", "U", {
        "type": "noSlip",
    })
    solver.boundary.set_raw_condition("rightWall", "U", {
        "type": "noSlip",
    })
    solver.boundary.set_raw_condition("lowerWall", "U", {
        "type": "noSlip",
    })

    # p_rgh — boundary conditions
    solver.boundary.set_raw_condition("atmosphere", "p_rgh", {
        "type": "prghTotalPressure",
        "p0": "$internalField",
    })
    solver.boundary.set_raw_condition("leftWall", "p_rgh", {
        "type": "fixedFluxPressure",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("rightWall", "p_rgh", {
        "type": "fixedFluxPressure",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("lowerWall", "p_rgh", {
        "type": "fixedFluxPressure",
        "value": "$internalField",
    })

    # Write boundary condition files — OpenFOAMFile.write_boundary_file now
    # adds #includeEtc "caseDicts/setConstraintTypes" by default.
    solver.boundary.write_boundary_conditions()

    # --- 6. setFields (initialisation de la colonne d'eau) ---
    print("4. Initialisation avec setFields ...")
    Functions.write_set_fields_dict(
        name="setFieldsDict",
        base_path=case_path,
        folder="system",
        default_values={"alpha.water": "0"},
        zones=[
            {
                "name": "waterColumn",
                "type": "box",
                "box": [[0, 0, -1], [0.1461, 0.292, 1]],
                "values": {"alpha.water": "1"},
            }
        ],
    )
    solver.run_command(
        ["setFields", "-case", str(case_path)],
        log_filename="log.setFields",
    )

    # setFields rewrites alpha.water and strips #includeEtc — restore it
    Functions.restore_includetec_boundary(case_path, "alpha.water")

    # --- 7. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleVoF — damBreak)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 8. Post-traitement ---
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)
    log_file = case_path / "log.incompressibleVoF"
    if log_file.exists():
        from foampilot.utilities.residuals import ResidualsPost

        residuals = ResidualsPost(log_file)
        residuals.process(export_csv=True, export_png=True)
        print("Residus exportes (CSV + PNG).")

    times = sorted(
        [d.name for d in case_path.iterdir()
         if d.is_dir() and Functions.is_numeric(d.name)],
        key=float,
    )
    if times:
        print(f"Temps disponibles : {times}")

    if times:
        last_time = times[-1]
        alpha_file = case_path / last_time / "alpha.water"
        if alpha_file.exists():
            content = alpha_file.read_text()
            has_nonuniform = "nonuniform" in content
            print(f"alpha.water a t={last_time}: "
                  f"{'nonuniformList' if has_nonuniform else 'uniform'}")

    print("\n" + "=" * 60)
    print("Simulation terminee avec succes !")
    print(f"  Cas      : {case_path}")
    print(f"  Log      : {case_path / 'log.incompressibleVoF'}")
    print(f"  Resultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
