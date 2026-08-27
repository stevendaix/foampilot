#!/usr/bin/env python3
"""Tutoriel 2 : Écoulement turbulent autour d'un véhicule (simpleFoam).

Reference OpenFOAM-13 : tutorials/incompressibleFluid/drivaerFastback
https://develop.openfoam.com/Development/openfoam/-/tree/master/tutorials/incompressibleFluid/drivaerFastback

Écoulement turbulent externe autour d'un véhicule simplifié avec
le solveur simpleFoam et le modèle k-omega SST.

Points clés :
- Solveur : incompressibleFluid (simpleFoam via foamRun)
- Turbulence : kOmegaSST (RAS)
- Maillage : blockMesh + snappyHexMesh
- Domaine : tunnel d'aération 30 x 10 x 10 m
- Véhicule : box simplifié (8 x 2 x 1.5 m) au centre du tunnel
- Vitesse d'entrée : 30 m/s

Pipeline :
1. blockMesh -- maillage de base du tunnel
2. surfaceFeatureExtract -- extraction des features
3. snappyHexMesh -- raffinement autour du véhicule STL
4. Setup des conditions aux limites
5. Simulation simpleFoam (stationnaire) + post-traitement

Usage :
    cd foampilot/tutorials/02_simpleCar_turbulent
    python run.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver, OpenFOAMEnvironment
from foampilot import Meshing
from foampilot.mesh.snappymesh import SnappyMesher
from foampilot.utilities.function import Functions


def main():
    environment = OpenFOAMEnvironment().environment()
    os.environ.update(environment)
    case_path = Path.cwd()

    # --- 1. Initialiser le solveur ---
    print("1. Initialisation du solveur (simpleFoam + kOmegaSST) ...")
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kOmegaSST"
    solver.transient = False  # Stationnaire (SIMPLE)
    solver.setup_case()

    # Configuration du controlDict
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 300.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0

    # SIMPLE -- steady-state
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
        "(k|omega)": "1e-4",
    }
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"p": "0.3"},
        "equations": {"U": "0.7", "(k|omega).*": "0.7"},
    }

    # Write system files
    solver.system.write()

    # --- 2. Maillage (blockMesh + snappyHexMesh) ---
    # Domain: 30 x 10 x 10 m (x = longitudial, y = lateral, z = vertical)
    print("2. Generation du maillage (blockMesh + snappyHexMesh) ...")
    bmd_mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = bmd_mesh.mesher
    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],    # 0
        [30, 0, 0],   # 1
        [30, 10, 0],  # 2
        [0, 10, 0],   # 3
        [0, 0, 10],   # 4
        [30, 0, 10],  # 5
        [30, 10, 10], # 6
        [0, 10, 10],  # 7
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (30 10 10) simpleGrading (1 1 1)",
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

    # Create car STL (simplified box: 8x2x1.5 m at position 10,4,1)
    stl_file = case_path / "constant" / "triSurface" / "simpleCar.stl"
    stl_file.parent.mkdir(parents=True, exist_ok=True)

    if not stl_file.exists():
        car_stl = """solid simpleCar
facet normal 0 0 1
outer loop
vertex 10 4 1.75
vertex 18 4 1.75
vertex 18 6 1.75
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 18 6 1.75
vertex 10 6 1.75
vertex 10 4 1.75
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 10 4 2.5
vertex 18 4 2.5
vertex 18 6 2.5
endloop
endfacet
facet normal 0 0 -1
outer loop
vertex 18 6 2.5
vertex 10 6 2.5
vertex 10 4 2.5
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 10 4 1.75
vertex 18 4 1.75
vertex 18 4 2.5
endloop
endfacet
facet normal 0 -1 0
outer loop
vertex 18 4 2.5
vertex 10 4 2.5
vertex 10 4 1.75
endloop
endfacet
facet normal 0 1 0
outer loop
vertex 10 6 1.75
vertex 10 6 2.5
vertex 18 6 2.5
endloop
endfacet
facet normal 0 1 0
outer loop
vertex 18 6 2.5
vertex 18 6 1.75
vertex 10 6 1.75
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 10 4 1.75
vertex 10 4 2.5
vertex 10 6 2.5
endloop
endfacet
facet normal -1 0 0
outer loop
vertex 10 6 2.5
vertex 10 6 1.75
vertex 10 4 1.75
endloop
endfacet
facet normal 1 0 0
outer loop
vertex 18 4 1.75
vertex 18 6 1.75
vertex 18 6 2.5
endloop
endfacet
facet normal 1 0 0
outer loop
vertex 18 6 2.5
vertex 18 4 2.5
vertex 18 4 1.75
endloop
endfacet
endsolid
"""
        stl_file.write_text(car_stl)
        print(f"  Created simpleCar.stl ({stl_file.stat().st_size} bytes)")

    snappy = SnappyMesher(
        parent=solver._solver,
        stl_file=str(stl_file),
        castellatedMesh=True,
        snap=True,
        addLayers=False,
    )
    snappy.locationInMesh = (5, 5, 5)  # point inside fluid domain
    snappy.castellatedMeshControls["maxLocalCells"] = 100000
    snappy.castellatedMeshControls["maxGlobalCells"] = 2000000
    snappy.refinementSurfaces = {
        "simpleCar": {"level": (2, 3)},
    }
    snappy.write_surface_features_dict(
        stl_list_for_emesh=["simpleCar.stl"],
        included_angle=60,
    )
    snappy.add_feature("simpleCar.eMesh", 0)
    snappy.write_snappyHexMeshDict()

    # Run: blockMesh → surfaceFeatureExtract → snappyHexMesh
    snappy.run()

    # --- 3. Constant files ---
    print("3. Ecriture des proprietes physiques (kOmegaSST) ...")
    solver.constant.write()

    # --- 4. Generate 0/ field files (initial conditions) ---

    # --- 5. Conditions aux limites ---
    print("4. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # U -- vitesse d'entrée 30 m/s
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (30 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
    })

    # Walls -- no-slip (ground, ceiling, leftWall, rightWall, simpleCar)
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "U", {
            "type": "noSlip",
        })

    # p
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "p", {"type": "zeroGradient"})

    # k -- turbulence intensity I=5% -> k = 1.5*(I*U)^2 = 1.5*(0.05*30)^2 = 0.3375
    k_inlet = 0.3375
    solver.boundary.set_raw_condition("inlet", "k", {
        "type": "fixedValue",
        "value": f"uniform {k_inlet}",
    })
    solver.boundary.set_raw_condition("outlet", "k", {
        "type": "inletOutlet",
        "inletValue": f"uniform {k_inlet}",
        "value": f"uniform {k_inlet}",
    })
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "k", {
            "type": "kqRWallFunction",
            "value": f"uniform {k_inlet}",
        })

    # omega -- specific dissipation rate
    omega_inlet = 1.0
    solver.boundary.set_raw_condition("inlet", "omega", {
        "type": "fixedValue",
        "value": f"uniform {omega_inlet}",
    })
    solver.boundary.set_raw_condition("outlet", "omega", {
        "type": "inletOutlet",
        "inletValue": f"uniform {omega_inlet}",
        "value": f"uniform {omega_inlet}",
    })
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "omega", {
            "type": "omegaWallFunction",
            "value": f"uniform {omega_inlet}",
        })

    # nut -- turbulent viscosity
    solver.boundary.set_raw_condition("inlet", "nut", {"type": "calculated", "value": "uniform 0"})
    solver.boundary.set_raw_condition("outlet", "nut", {"type": "calculated", "value": "uniform 0"})
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "nut", {
            "type": "nutkWallFunction",
            "value": "uniform 0",
        })

    # Write boundary condition files
    # (write_boundary_file adds #includeEtc "caseDicts/setConstraintTypes" by default)
    solver.boundary.write_boundary_conditions()

    # --- 6. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleFluid -- simpleCar)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 7. Post-traitement ---
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
    print("Simulation terminée avec succès !")
    print(f"  Cas      : {case_path}")
    print(f"  Log      : {case_path / 'log.incompressibleFluid'}")
    print(f"  Résultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
