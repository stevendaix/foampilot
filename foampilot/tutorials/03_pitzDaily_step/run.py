#!/usr/bin/env python3
"""Tutoriel 3 : Écoulement sur marche descendante (pitzDaily).

Reference OpenFOAM-13 : tutorials/incompressibleFluid/pitzDaily
  (backward-facing step, 2D, transient, kEpsilon via PIMPLE)

Pipeline :
1. Gmsh -- géométrie 2D + extrusion 1 couche en Z (Layers{1})
2. Classify faces by bounding box + OCC center-of-mass -> physical groups
3. DirectOpenFOAMExporter -- export direct vers constant/polyMesh
4. Fix boundary file : frontAndBack=empty, walls=wall
5. Setup des conditions aux limites + solveurs
6. Simulation PIMPLE (transient) + post-traitement

Usage :
    cd foampilot/tutorials/03_pitzDaily_step
    python3 run.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot import Meshing
from foampilot.solver import Solver, OpenFOAMEnvironment
from foampilot.utilities.function import Functions


def build_geometry_and_mesh(case_path):
    """Recreate the official OpenFOAM 13 pitzDaily multi-block mesh."""
    meshing = Meshing(case_path, mesher="blockMesh")
    blockmesh = meshing.mesher
    blockmesh.scale = 0.001
    blockmesh.vertices = [
        [-20.6, 0, -0.5], [-20.6, 25.4, -0.5], [0, -25.4, -0.5],
        [0, 0, -0.5], [0, 25.4, -0.5], [206, -25.4, -0.5],
        [206, 0, -0.5], [206, 25.4, -0.5], [290, -16.6, -0.5],
        [290, 0, -0.5], [290, 16.6, -0.5], [-20.6, 0, 0.5],
        [-20.6, 25.4, 0.5], [0, -25.4, 0.5], [0, 0, 0.5],
        [0, 25.4, 0.5], [206, -25.4, 0.5], [206, 0, 0.5],
        [206, 25.4, 0.5], [290, -16.6, 0.5], [290, 0, 0.5],
        [290, 16.6, 0.5],
    ]
    blockmesh.blocks = [
        "hex (0 3 4 1 11 14 15 12) (18 30 1) simpleGrading (0.5 2.0 1)",
        "hex (2 5 6 3 13 16 17 14) (180 27 1) edgeGrading (4 4 4 4 0.3 1 1 0.3 1 1 1 1)",
        "hex (3 6 7 4 14 17 18 15) (180 30 1) edgeGrading (4 4 4 4 2 0.25 0.25 2 1 1 1 1)",
        "hex (5 8 9 6 16 19 20 17) (25 27 1) simpleGrading (2.5 1 1)",
        "hex (6 9 10 7 17 20 21 18) (25 30 1) simpleGrading (2.5 0.25 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [[0, 1, 12, 11]]},
        "outlet": {"type": "patch", "faces": [[8, 9, 20, 19], [9, 10, 21, 20]]},
        "upperWall": {"type": "wall", "faces": [[1, 4, 15, 12], [4, 7, 18, 15], [7, 10, 21, 18]]},
        "lowerWall": {"type": "wall", "faces": [[0, 3, 14, 11], [3, 2, 13, 14], [2, 5, 16, 13], [5, 8, 19, 16]]},
        "frontAndBack": {"type": "empty", "faces": [[0, 3, 4, 1], [2, 5, 6, 3], [3, 6, 7, 4], [5, 8, 9, 6], [6, 9, 10, 7], [11, 14, 15, 12], [13, 16, 17, 14], [14, 17, 18, 15], [16, 19, 20, 17], [17, 20, 21, 18]]},
    }
    blockmesh.mergePatchPairs = []
    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()


def main():
    environment = OpenFOAMEnvironment().environment()
    os.environ.update(environment)
    case_path = Path.cwd()

    # --- 1. Solver ---
    print("1. Initialisation du solveur (fluid, LES kEqn, PIMPLE transient) ...")
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.compressible = True
    solver.with_gravity = False
    solver.energy_activated = True
    solver.turbulence_model = "LES:kEqn"
    solver.transient = True
    solver.setup_case()
    solver.fields_manager.register_field("T", 300.0, "K")
    solver.fields_manager.register_field("k", 0.0, "m^2/s^2")
    solver.fields_manager.register_field("nut", 0.0, "m^2/s")
    solver.fields_manager.register_field("alphat", 0.0, "m^2/s")
    solver.fields_manager.register_field("muTilda", 0.0, "Pa*s")
    solver.constant.physicalProperties.configure_reference(
        thermo_type={
            "type": "hePsiThermo", "mixture": "pureMixture", "transport": "const",
            "thermo": "eConst", "equationOfState": "perfectGas",
            "specie": "specie", "energy": "sensibleInternalEnergy",
        },
        mixture={
            "specie": {"molWeight": 28.9},
            "thermodynamics": {"Cv": 712, "hf": 0},
            "transport": {"mu": 1.8e-05, "Pr": 0.7},
        },
    )

    # controlDict
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 0.3
    solver.system.controlDict.deltaT = 1e-05
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0
    solver.system.controlDict.adjustTimeStep = False
    solver.system.controlDict.maxCo = 0.5

    # PIMPLE
    solver.system.fvSolution.set_pimple(
        nOuterCorrectors=3,
        nCorrectors=1,
        nNonOrthogonalCorrectors=0,
    )

    # Pressure solver: GAMG with DICGaussSeidel
    solver.system.fvSolution.solvers["p"] = {
        "solver": "GAMG",
        "tolerance": "1e-07",
        "relTol": "0.01",
        "smoother": "DICGaussSeidel",
    }

    # No relaxation factors for transient PIMPLE
    solver.system.fvSolution.relaxationFactors = {"fields": {}, "equations": {}}

    # div schemes without "bounded" prefix (matches OF13 ref for transient)
    solver.system.fvSchemes.divSchemes = {
        "default": "none",
        "div(phi,U)": "Gauss LUST grad(U)",
        "div(phi,e)": "Gauss LUST grad(e)",
        "div(phi,K)": "Gauss linear",
        "div(phi,(p|rho))": "Gauss linear",
        "div(phi,k)": "Gauss limitedLinear 1",
        "div(phi,B)": "Gauss limitedLinear 1",
        "div(phi,muTilda)": "Gauss limitedLinear 1",
        "div(B)": "Gauss linear",
        "div(((rho*nuEff)*dev2(T(grad(U)))))": "Gauss linear",
    }

    solver.system.write()
    solver.constant.write()

    # --- 2. Geometry + mesh ---
    print("2. Géométrie + maillage (blockMesh Foampilot, référence OpenFOAM 13) ...")
    build_geometry_and_mesh(case_path)

    # --- 3. Boundary conditions ---
    print("3. Conditions aux limites ...")
    solver.boundary.initialize_boundary()

    U = 0.0
    k_val = 2e-05
    wall_patches = ("upperWall", "lowerWall")

    # U
    solver.boundary.set_raw_condition("inlet", "U", {"type": "turbulentInlet", "referenceField": "uniform (10 0 0)", "fluctuationScale": (0.02, 0.01, 0.01), "value": "uniform (10 0 0)"})
    solver.boundary.set_raw_condition("outlet", "U", {"type": "pressureInletOutletVelocity", "inletValue": "uniform (0 0 0)", "value": "uniform (0 0 0)"})
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "U", {"type": "fixedValue", "value": "uniform (0 0 0)"})
    solver.boundary.set_raw_condition("frontAndBack", "U", {"type": "empty"})

    # p
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "waveTransmissive", "gamma": 1.3, "fieldInf": 1e5, "lInf": 0.3, "value": "$internalField"})
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "p", {"type": "empty"})

    # LES k and turbulent viscosity
    solver.boundary.set_raw_condition("inlet", "k", {"type": "fixedValue", "value": f"uniform {k_val}"})
    solver.boundary.set_raw_condition("outlet", "k", {"type": "inletOutlet", "inletValue": "uniform 0", "value": "uniform 0"})
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "k", {"type": "fixedValue", "value": "uniform 0"})
    solver.boundary.set_raw_condition("frontAndBack", "k", {"type": "empty"})
    for patch in ("inlet", "outlet", *wall_patches):
        solver.boundary.set_raw_condition(patch, "nut", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "nut", {"type": "empty"})

    # Energy and thermophysical transport fields from the reference case.
    solver.boundary.set_raw_condition("inlet", "T", {"type": "fixedValue", "value": "uniform 300"})
    solver.boundary.set_raw_condition("outlet", "T", {"type": "inletOutlet", "inletValue": "uniform 300", "value": "uniform 300"})
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "T", {"type": "fixedValue", "value": "uniform 300"})
    solver.boundary.set_raw_condition("frontAndBack", "T", {"type": "empty"})
    for patch in ("inlet", "outlet", *wall_patches):
        solver.boundary.set_raw_condition(patch, "alphat", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "alphat", {"type": "empty"})
    solver.boundary.set_raw_condition("inlet", "muTilda", {"type": "fixedValue", "value": "uniform 0"})
    solver.boundary.set_raw_condition("outlet", "muTilda", {"type": "inletOutlet", "inletValue": "uniform 0", "value": "uniform 0"})
    for patch in wall_patches:
        solver.boundary.set_raw_condition(patch, "muTilda", {"type": "fixedValue", "value": "uniform 0"})
    solver.boundary.set_raw_condition("frontAndBack", "muTilda", {"type": "empty"})


    solver.boundary.write_boundary_conditions()

    # --- 4. Simulation ---
    print("\n" + "=" * 60)
    print("Lancement simulation (foamRun -solver incompressibleFluid)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 5. Post-traitement ---
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)

    log_file = case_path / "log.incompressibleFluid"
    if log_file.exists():
        from foampilot.utilities.residuals import ResidualsPost
        residuals = ResidualsPost(log_file)
        residuals.process(export_csv=True, export_png=True)
        print("  Résidus exportés (CSV + PNG).")

    times = sorted(
        [d.name for d in case_path.iterdir()
         if d.is_dir()
         and d.name not in ("constant", "system", "0", "postProcessing")
         and Functions.is_numeric(d.name)],
        key=float,
    )
    if times:
        print(f"  Temps disponibles : {times}")

    print("\n" + "=" * 60)
    print("Simulation terminée !")
    print(f"  Cas      : {case_path}")
    print(f"  Log      : {case_path / 'log.incompressibleFluid'}")
    print(f"  Résultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
