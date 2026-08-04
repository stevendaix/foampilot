#!/usr/bin/env python3
"""Tutoriel 7 : Ecoulement autour d'une moto (motorBike, simpleFoam).

Reference OpenFOAM-13 : tutorials/incompressibleFluid/motorBike

Ecoulement turbulent exterieur a haute vitesse autour d'une moto.
Utilise blockMesh pour le maillage de base et snappyHexMesh pour
l'adaptation autour de la geometrie STL complexe de la moto.

Points cles :
- Solveur : incompressibleFluid (simpleFoam via foamRun -solver)
- Turbulence : SpalartAllmaras (RAS)
- Maillage : blockMesh + snappyHexMesh
- Domaine : tunnel d'aeration 20x8x8 m (symetrie frontale/back)
- Gravité : None (ecoulement horizontal)
- Geometrie : motorBike.obj copie depuis $FOAM_TUTORIALS/resources/geometry/

Pipeline :
1. blockMesh -- maillage de base rectangulaire
2. surfaceFeatureExtract -- extraction des features depuis le STL
3. snappyHexMesh -- raffinement autour de la moto (motorBike.obj)
4. Setup des conditions aux limites
5. Simulation simpleFoam + post-traitement

Usage :
    cd foampilot/tutorials/07_motorBike
    python run.py
"""

import sys
from pathlib import Path

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
    solver.turbulence_model = "SpalartAllmaras"
    solver.transient = False  # Stationnaire (SIMPLE)

    # Configuration du controlDict
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 200.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.controlDict.purgeWrite = 0

    # SIMPLE -- steady-state solver configuration
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
        "nuTilda": "1e-4",
    }
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"p": "0.3"},
        "equations": {"U": "0.7", "nuTilda": "0.5"},
    }

    # Write system files
    solver.system.write()

    # --- 2. Maillage (blockMesh + snappyHexMesh) ---
    # Step 2a: blockMesh for wind tunnel background mesh
    # Domain: 20 x 8 x 8 m (matching OpenFOAM reference)
    bmd_mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = bmd_mesh.mesher
    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],    # 0
        [20, 0, 0],   # 1
        [20, 8, 0],   # 2
        [0, 8, 0],    # 3
        [0, 0, 8],    # 4
        [20, 0, 8],   # 5
        [20, 8, 8],   # 6
        [0, 8, 8],    # 7
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (20 8 8) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [[0, 3, 7, 4]]},
        "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
        "lowerWall": {"type": "wall", "faces": [[0, 1, 2, 3]]},
        "upperWall": {"type": "symmetryPlane", "faces": [[4, 5, 6, 7]]},
        "front": {"type": "symmetryPlane", "faces": [[0, 1, 5, 4]]},
        "back": {"type": "symmetryPlane", "faces": [[3, 2, 6, 7]]},
    }
    blockmesh.mergePatchPairs = []
    blockmesh.write(case_path / "system" / "blockMeshDict")

    # Step 2b: Copy motorBike.obj from OpenFOAM resources
    stl_file = case_path / "constant" / "triSurface" / "motorBike.obj"
    stl_file.parent.mkdir(parents=True, exist_ok=True)

    if not stl_file.exists():
        import subprocess
        result = subprocess.run(
            ["sh", "-c", "echo $FOAM_TUTORIALS"],
            capture_output=True, text=True,
        )
        foam_tutorials = result.stdout.strip()
        if not foam_tutorials:
            foam_tutorials = "/opt/openfoam13/tutorials"

        obj_gz = Path(foam_tutorials) / "resources/geometry/motorBike.obj.gz"
        if obj_gz.exists():
            import gzip
            with gzip.open(obj_gz, "rb") as f_in:
                stl_file.write_bytes(f_in.read())
            print(f"Copied motorBike.obj from {obj_gz} to {stl_file}")
        else:
            raise FileNotFoundError(f"Cannot find motorBike.obj.gz at {obj_gz}")

    # Step 2c: snappyHexMesh
    snappy = SnappyMesher(
        parent=solver._solver,
        stl_file=str(stl_file),
        castellatedMesh=True,
        snap=True,
        addLayers=False,
    )
    snappy.locationInMesh = (5, 4, 2)  # point inside fluid domain
    snappy.castellatedMeshControls["maxLocalCells"] = 100000
    snappy.castellatedMeshControls["maxGlobalCells"] = 7000000

    # Add surface refinement (matching reference level 6-8)
    snappy.refinementSurfaces = {
        "motorBike": {"level": (6, 8)},
    }

    snappy.write_surface_features_dict(
        stl_list_for_emesh=["motorBike.obj"],
        included_angle=60,
    )
    snappy.add_feature("motorBike.eMesh", 0)
    snappy.write_snappyHexMeshDict()

    # Run: blockMesh -> surfaceFeatureExtract -> snappyHexMesh
    snappy.run()

    # --- 3. Ecrire les fichiers constants ---
    print("2. Ecriture des proprietes physiques (SpalartAllmaras) ...")
    solver.constant.write()

    # --- 4. Generate 0/ field files ---
    solver.setup_case()

        # --- 5. Conditions aux limites ---
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # U -- inlet velocity 20 m/s (matching reference)
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (20 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "inletOutlet",
        "inletValue": "uniform (0 0 0)",
        "value": "uniform (20 0 0)",
    })

    # p -- zeroGradient on inlet, fixedValue(0) on outlet
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})

    # nut -- calculated on inlet/outlet, wall function on walls
    solver.boundary.set_raw_condition("inlet", "nut", {"type": "calculated", "value": "uniform 0"})
    solver.boundary.set_raw_condition("outlet", "nut", {"type": "calculated", "value": "uniform 0"})

    # nuTilda -- inlet fixedValue, outlet inletOutlet
    solver.boundary.set_raw_condition("inlet", "nuTilda", {"type": "fixedValue", "value": "uniform 0.05"})
    solver.boundary.set_raw_condition("outlet", "nuTilda", {
        "type": "inletOutlet",
        "inletValue": "uniform 0.05",
        "value": "uniform 0.05",
    })

    # Symmetry planes
    for patch in ("front", "back", "upperWall"):
        solver.boundary.set_raw_condition(patch, "U", {"type": "symmetryPlane"})
        solver.boundary.set_raw_condition(patch, "p", {"type": "symmetryPlane"})
        solver.boundary.set_raw_condition(patch, "nut", {"type": "symmetryPlane"})
        solver.boundary.set_raw_condition(patch, "nuTilda", {"type": "symmetryPlane"})

    # lowerWall -- wall with no-slip
    solver.boundary.set_raw_condition("lowerWall", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("lowerWall", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("lowerWall", "nut", {
        "type": "nutUSpaldingWallFunction",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("lowerWall", "nuTilda", {
        "type": "fixedValue",
        "value": "uniform 0",
    })

    # motorBike surfaces -- noSlip walls
    solver.boundary.set_raw_condition("motorBike_r.", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("motorBike_r.", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("motorBike_r.", "nut", {
        "type": "nutUSpaldingWallFunction",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("motorBike_r.", "nuTilda", {
        "type": "fixedValue",
        "value": "uniform 0",
    })

    # Write boundary condition files
    solver.boundary.write_boundary_conditions()

    # --- 6. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleFluid -- motorBike)")
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
