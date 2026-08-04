#!/usr/bin/env python3
"""Tutoriel 6 : Aérodynamique des bâtiments — écoulement turbulent (simpleFoam).

Référence OpenFOAM-13 : tutorials/incompressibleFluid/windAroundBuildings
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressibleFluid/windAroundBuildings

Cet exemple simule un écoulement turbulent extérieur autour d'un bâtiment
en utilisant blockMesh pour le maillage de base et snappyHexMesh pour
l'adaptation autour de la géométrie STL des bâtiments.

Le pipeline de maillage :
1. blockMesh — maillage de base rectangulaire (tunnel d'aération)
2. surfaceFeatureExtract — extraction des arêtes features depuis le STL
3. snappyHexMesh — raffinement du maillage autour du bâtiment STL
"""

import sys
from pathlib import Path

# Add src to path for tutorial execution from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing
from foampilot.mesh.snappymesh import SnappyMesher
from foampilot.utilities.function import Functions


def main():
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur incompressible turbulente ---
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kEpsilon"
    solver.transient = False  # Steady-state (simpleFoam)

    # --- 2. Configuration du controlDict ---
    # OpenFOAM 13 uses 'solver' keyword for foamRun -solver
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 100.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 50
    solver.system.controlDict.purgeWrite = 0

    # SIMPLE — steady-state solver configuration
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
        "(k|epsilon)": "1e-4",
    }

    # Relaxation factors — p=0.3, U/k/epsilon=0.7 (match OpenFOAM reference)
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"p": "0.3"},
        "equations": {"U": "0.7", "(k|epsilon).*": "0.7"},
    }

    # Write system files
    solver.system.write()

    # --- 3. Maillage (blockMesh + snappyHexMesh) ---
    # Step 3a: blockMesh for the wind tunnel background mesh
    # Wind tunnel domain: 200 x 100 x 50 m (x = streamwise, y = transverse, z = vertical)
    bmd_mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = bmd_mesh.mesher
    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],    # 0
        [200, 0, 0],  # 1
        [200, 100, 0],# 2
        [0, 100, 0],  # 3
        [0, 0, 50],   # 4
        [200, 0, 50], # 5
        [200, 100, 50],# 6
        [0, 100, 50], # 7
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (40 20 10) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [[0, 3, 7, 4]]},
        "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
        "ground": {"type": "wall", "faces": [[0, 1, 2, 3]]},
        "ceiling": {"type": "wall", "faces": [[4, 5, 6, 7]]},
        "leftWall": {"type": "wall", "faces": [[0, 1, 5, 4]]},
        "rightWall": {"type": "wall", "faces": [[3, 2, 6, 7]]},
    }
    blockmesh.mergePatchPairs = []
    blockmesh.write(case_path / "system" / "blockMeshDict")

    # Step 3b: blockMesh, surfaceFeatureExtract, snappyHexMesh
    stl_file = case_path / "constant" / "triSurface" / "buildings.stl"
    stl_file.parent.mkdir(parents=True, exist_ok=True)

    # Create STL if it does not exist (protects against rm -rf constant cleanup)
    if not stl_file.exists():
        building_stl = """solid building
facet normal 0 0 1
outer loop
vertex 100 100 0
vertex 120 100 0
vertex 120 120 0
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 120 120 0
vertex 100 120 0
vertex 100 100 0
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 100 100 20
vertex 120 100 20
vertex 120 120 20
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 120 120 20
vertex 100 120 20
vertex 100 100 20
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 100 100 0
vertex 120 100 0
vertex 120 100 20
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 120 100 20
vertex 100 100 20
vertex 100 100 0
endloop
endfacet
facet normal 0 1 0
outer loop
vertex 100 120 0
vertex 100 120 20
vertex 120 120 20
endloop
endfacet
facet normal 0 1 0
outer loop
vertex 120 120 20
vertex 120 120 0
vertex 100 120 0
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 100 100 0
vertex 100 100 20
vertex 100 120 20
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 100 120 20
vertex 100 120 0
vertex 100 100 0
endloop
endfacet
facet normal 1 0 0
outer loop
vertex 120 100 0
vertex 120 120 0
vertex 120 120 20
endloop
endfacet
facet normal 1 0 0
outer loop
vertex 120 120 20
vertex 120 100 20
vertex 120 100 0
endloop
endfacet
endsolid
"""
        stl_file.write_text(building_stl)
    snappy = SnappyMesher(
        parent=solver._solver,
        stl_file=str(stl_file),
        castellatedMesh=True,
        snap=True,
        addLayers=False,
    )
    snappy.locationInMesh = (10, 50, 5)  # point inside fluid domain
    snappy.castellatedMeshControls["maxLocalCells"] = 200000
    snappy.castellatedMeshControls["maxGlobalCells"] = 4000000

    # Write surfaceFeaturesDict and snappyHexMeshDict
    snappy.write_surface_features_dict(
        stl_list_for_emesh=["buildings.eMesh"],
        included_angle=60,
    )
    snappy.write_snappyHexMeshDict()

    # Run: blockMesh → surfaceFeatureExtract → snappyHexMesh
    snappy.run()

    # --- 4. Écrire les fichiers constants (turbulent k-epsilon) ---
    print("2. Ecriture des proprietes physiques (turbulent k-epsilon) ...")
    solver.constant.write()

    # --- 5. Generate 0/ field files (initial conditions) ---
    solver.setup_case()

    # --- 6. Conditions aux limites ---
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # U — vitesse d'entrée 10 m/s (wind)
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (10 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
    })

    # Set wall conditions for all wall-type patches
    # (buildings, ground, leftWall, rightWall, ceiling)
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "U", {
            "type": "noSlip",
        })

    # k — turbulence intensity I=0.1 → k = 1.5*(I*U)^2 = 1.5*(1)^2 = 1.5
    k_inlet = 1.5
    solver.boundary.set_raw_condition("inlet", "k", {
        "type": "fixedValue",
        "value": f"uniform {k_inlet}",
    })
    solver.boundary.set_raw_condition("outlet", "k", {
        "type": "inletOutlet",
        "inletValue": f"uniform {k_inlet}",
        "value": f"uniform {k_inlet}",
    })
    for patch_name in solver.boundary.fields["k"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "k", {
            "type": "kqRWallFunction",
            "value": f"uniform {k_inlet}",
        })

    # epsilon — C_mu^0.75 * k^1.5 / L, L=10 → 0.03
    eps_inlet = 0.03
    solver.boundary.set_raw_condition("inlet", "epsilon", {
        "type": "fixedValue",
        "value": f"uniform {eps_inlet}",
    })
    solver.boundary.set_raw_condition("outlet", "epsilon", {
        "type": "inletOutlet",
        "inletValue": f"uniform {eps_inlet}",
        "value": f"uniform {eps_inlet}",
    })
    for patch_name in solver.boundary.fields["epsilon"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "epsilon", {
            "type": "epsilonWallFunction",
            "value": f"uniform {eps_inlet}",
        })

    # p — pression
    solver.boundary.set_raw_condition("inlet", "p", {
        "type": "zeroGradient",
    })
    solver.boundary.set_raw_condition("outlet", "p", {
        "type": "fixedValue",
        "value": "uniform 0",
    })
    for patch_name in solver.boundary.fields["p"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "p", {
            "type": "zeroGradient",
        })

    # nut — turbulent viscosity (calculated on inlet/outlet, wall function on walls)
    solver.boundary.set_raw_condition("inlet", "nut", {
        "type": "calculated",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("outlet", "nut", {
        "type": "calculated",
        "value": "uniform 0",
    })
    for patch_name in solver.boundary.fields["nut"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "nut", {
            "type": "nutkWallFunction",
            "value": "uniform 0",
        })

    # Write boundary condition files — OpenFOAMFile.write_boundary_file adds
    # #includeEtc "caseDicts/setConstraintTypes" by default.
    solver.boundary.write_boundary_conditions()

    # --- 7. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleFluid — buildingAero)")
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
