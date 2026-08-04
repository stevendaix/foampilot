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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import gmsh
import numpy as np
from foampilot.solver import Solver
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter
from foampilot.utilities.function import Functions


def build_geometry_and_mesh(case_path, lc=0.1):
    """Build backward-facing step geometry + mesh in Gmsh.

    Creates a 2D profile, extrudes it 1 layer in Z (0.01 m), classifies
    surface faces by geometric position, creates physical groups, and
    generates the 3D mesh.
    """
    gmsh.initialize()
    gmsh.model.add("pitzDaily")

    p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.occ.addPoint(1.0, 0.0, 0.0, lc)
    p3 = gmsh.model.occ.addPoint(1.0, 0.5, 0.0, lc)
    p4 = gmsh.model.occ.addPoint(6.0, 0.5, 0.0, lc)
    p5 = gmsh.model.occ.addPoint(6.0, 0.6, 0.0, lc)
    p6 = gmsh.model.occ.addPoint(1.0, 0.6, 0.0, lc)
    p7 = gmsh.model.occ.addPoint(1.0, 1.0, 0.0, lc)
    p8 = gmsh.model.occ.addPoint(0.0, 1.0, 0.0, lc)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p5)
    l5 = gmsh.model.occ.addLine(p5, p6)
    l6 = gmsh.model.occ.addLine(p6, p7)
    l7 = gmsh.model.occ.addLine(p7, p8)
    l8 = gmsh.model.occ.addLine(p8, p1)

    outer_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
    surf = gmsh.model.occ.addPlaneSurface([outer_loop])
    gmsh.model.occ.synchronize()

    # Mesh 2D surface, then extrude with 1 layer
    gmsh.model.mesh.generate(2)
    gmsh.model.occ.extrude([(2, surf)], 0, 0, 0.01, numElements=[1])
    gmsh.model.occ.synchronize()

    # Classify surface faces by geometric position (after extrusion)
    face_groups = {"inlet": [], "outlet": [], "walls": [], "frontAndBack": []}
    faces_2d = gmsh.model.getEntities(dim=2)

    for dim, tag in faces_2d:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        bbox = gmsh.model.getBoundingBox(dim, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox

        if abs(zmax - zmin) < 0.001:
            face_groups["frontAndBack"].append(tag)
        elif abs(com[0]) < 0.15:
            face_groups["inlet"].append(tag)
        elif abs(com[0] - 6.0) < 0.15:
            face_groups["outlet"].append(tag)
        else:
            face_groups["walls"].append(tag)

    print("=== Classification des faces ===")
    for name, tags in face_groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, name=name)
            print(f"  {name}: {len(tags)} faces")

    volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
    gmsh.model.addPhysicalGroup(3, volumes, name="fluid")

    gmsh.model.mesh.generate(3)
    coords = np.array(gmsh.model.mesh.getNodes()[1]).reshape(-1, 3)
    n_z = len(np.unique(np.round(coords[:, 2], 8)))
    print(f"  Maillage : {len(gmsh.model.mesh.getNodes()[0])} nodes ({n_z} Z-layers)")

    # Direct export to polyMesh
    exporter = DirectOpenFOAMExporter(case_path)
    exporter.export_single_region()
    print("  Export direct vers constant/polyMesh OK")

    # Fix patch types: frontAndBack=empty, walls=wall
    boundary_file = case_path / "constant" / "polyMesh" / "boundary"
    boundary_text = boundary_file.read_text()
    boundary_text = boundary_text.replace(
        "frontAndBack\n    {\n        type            patch;",
        "frontAndBack\n    {\n        type            empty;",
    )
    boundary_text = boundary_text.replace(
        "walls\n    {\n        type            patch;",
        "walls\n    {\n        type            wall;",
    )
    boundary_file.write_text(boundary_text)
    print("  Correction boundary : frontAndBack=empty, walls=wall")

    gmsh.finalize()


def main():
    case_path = Path.cwd()

    # --- 1. Solver ---
    print("1. Initialisation du solveur (kEpsilon, PIMPLE transient) ...")
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kEpsilon"
    solver.transient = True

    # controlDict
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 0.3
    solver.system.controlDict.deltaT = 0.001
    solver.system.controlDict.writeControl = "adjustableRunTime"
    solver.system.controlDict.writeInterval = 0.05
    solver.system.controlDict.purgeWrite = 0
    solver.system.controlDict.adjustTimeStep = True
    solver.system.controlDict.maxCo = 5
    solver.system.controlDict.maxDeltaT = 0.001

    # PIMPLE
    solver.system.fvSolution.set_pimple(
        nOuterCorrectors=1,
        nCorrectors=2,
        nNonOrthogonalCorrectors=1,
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
        "div(phi,U)": "Gauss linearUpwind grad(U)",
        "div(phi,k)": "Gauss upwind",
        "div(phi,epsilon)": "Gauss upwind",
        "div(phi,nuTilda)": "Gauss upwind",
        "div((nuEff*dev2(T(grad(U)))))": "Gauss linear",
    }

    solver.system.write()
    solver.constant.write()
    solver.setup_case()

    # --- 2. Geometry + mesh ---
    print("2. Géométrie + maillage (Gmsh + export direct) ...")
    build_geometry_and_mesh(case_path, lc=0.1)

    # --- 3. Boundary conditions ---
    print("3. Conditions aux limites ...")
    solver.boundary.initialize_boundary()

    U = 10.0
    k_val = 1.5 * (0.05 * U) ** 2     # 0.375 (I=5%)
    eps_val = k_val ** 1.5 / 0.3      # 14.855

    # U
    solver.boundary.set_raw_condition("inlet", "U", {"type": "fixedValue", "value": f"uniform ({U} 0 0)"})
    solver.boundary.set_raw_condition("outlet", "U", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("walls", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("frontAndBack", "U", {"type": "empty"})

    # p
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})
    solver.boundary.set_raw_condition("walls", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "p", {"type": "empty"})

    # k
    solver.boundary.set_raw_condition("inlet", "k", {"type": "fixedValue", "value": f"uniform {k_val}"})
    solver.boundary.set_raw_condition("outlet", "k", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("walls", "k", {"type": "kqRWallFunction", "value": f"uniform {k_val}"})
    solver.boundary.set_raw_condition("frontAndBack", "k", {"type": "empty"})

    # epsilon
    solver.boundary.set_raw_condition("inlet", "epsilon", {"type": "fixedValue", "value": f"uniform {eps_val}"})
    solver.boundary.set_raw_condition("outlet", "epsilon", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("walls", "epsilon", {"type": "epsilonWallFunction", "value": f"uniform {eps_val}"})
    solver.boundary.set_raw_condition("frontAndBack", "epsilon", {"type": "empty"})

    # nut
    solver.boundary.set_raw_condition("inlet", "nut", {"type": "calculated", "value": "uniform 0"})
    solver.boundary.set_raw_condition("outlet", "nut", {"type": "calculated", "value": "uniform 0"})
    solver.boundary.set_raw_condition("walls", "nut", {"type": "nutkWallFunction", "value": "uniform 0"})
    solver.boundary.set_raw_condition("frontAndBack", "nut", {"type": "empty"})

    # nuTilda
    solver.boundary.set_raw_condition("inlet", "nuTilda", {"type": "fixedValue", "value": "uniform 0"})
    solver.boundary.set_raw_condition("outlet", "nuTilda", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("walls", "nuTilda", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "nuTilda", {"type": "empty"})

    solver.boundary.write_boundary_conditions()

    # Fix frontAndBack -> empty in all field files
    for field in ["p", "k", "epsilon", "nut", "nuTilda"]:
        f = case_path / "0" / field
        if f.exists():
            text = f.read_text()
            for old, new in [
                ('"frontAndBack"\n    {\n        type            zeroGradient;',
                 '"frontAndBack"\n    {\n        type            empty;'),
                ('"frontAndBack"\n    {\n        type            calculated;',
                 '"frontAndBack"\n    {\n        type            empty;'),
                ('"frontAndBack"\n    {\n    }',
                 '"frontAndBack"\n    {\n        type            empty;\n    }'),
            ]:
                text = text.replace(old, new)
            f.write_text(text)

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
