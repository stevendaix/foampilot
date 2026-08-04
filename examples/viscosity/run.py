#!/usr/bin/env python3
"""
Exemple complet : écoulement à deux phases (eau/air) avec solveur VoF
et foampilot.

Ce cas est une adaptation foampilot du cas Template fourni dans
``examples/viscosity/Template/``.  Il simule l'introduction d'un
liquide visqueux (propriétés proches de l'huile) au moyen d'un
injectionneur situé sur le fond d'une cuve cubique, sous l'effet de la
gravité, avec tension superficielle.

Cas :
    - Domaine cubique 0.1 m x 0.1 m x 0.1 m (blockMesh)
    - Solveur : incompressibleVoF (interFoam — OpenFOAM 13)
    - Phase 1 : eau   (nu=72e-06, rho=915)   — liquide visqueux
    - Phase 2 : air   (nu=1.48e-05, rho=1.2)
    - Gravite : (0 0 -9.81) m/s^2
    - Turbulence : laminaire
    - Algorithme : PIMPLE (transient)
    - AMR : dynamicRefine sur alpha.water

Usage :
    cd examples/viscosity
    python run.py

Author: foampilot
"""

from pathlib import Path

from foampilot.solver import Solver
from foampilot import Meshing
from foampilot.system.decomposeParDictFile import DecomposeParDictFile
from foampilot.utilities.function import Functions

CASE_REL_PATH = "case"


def main():
    case_path = Path(__file__).resolve().parent / CASE_REL_PATH
    case_path.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. Initialisation du solveur VoF
    # ==================================================================
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = True
    solver.is_vof = True
    solver.transient = True
    solver.turbulence_model = "laminar"

    # Configure VoF constant files (phaseProperties, physicalProperties.<phase>,
    # momentumTransport) via the library
    solver.constant.configure_vof(
        phases=["water", "air"],
        sigma=0.032,
        phase_properties={
            "water": {"nu": 72e-06, "rho": 915},
            "air": {"nu": 1.48e-05, "rho": 1.2},
        },
    )

    # ==================================================================
    # 2. Configuration du controlDict
    # ==================================================================
    # OpenFOAM 13 uses 'solver' keyword (not 'application') with foamRun -solver.
    # Using 'application' can cause the solver to expect extra fields
    # (e.g. T) that are not needed for VoF.
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 0.2
    solver.system.controlDict.deltaT = 0.001
    solver.system.controlDict.writeControl = "adjustableRunTime"
    solver.system.controlDict.writeInterval = 0.01
    solver.system.controlDict.set_adaptive_time_step(
        adjustTimeStep=True,
        maxCo=0.9,
        maxAlphaCo=0.5,
        maxDeltaT=1.0,
    )

    # PIMPLE — configuration matching the Template reference case
    solver.system.fvSolution.set_pimple(
        nCorrectors=3,
        nNonOrthogonalCorrectors=0,
        momentumPredictor=True,
    )
    solver.system.fvSolution.PIMPLE["consistent"] = "yes"
    solver.system.fvSolution.PIMPLE.pop("pRefCell", None)
    solver.system.fvSolution.PIMPLE.pop("pRefValue", None)

    # Remove T solvers (not used in VoF without energy)
    solver.system.fvSolution.solvers.pop("T", None)
    solver.system.fvSolution.solvers.pop("TFinal", None)
    # VoF uses p_rgh, not p
    solver.system.fvSolution.solvers.pop("p", None)
    solver.system.fvSolution.solvers.pop("pFinal", None)
    solver.system.fvSolution.solvers.pop("alpha.water", None)
    solver.system.fvSolution.solvers.pop("alpha.air", None)

    # Match official OpenFOAM 13 incompressibleVoF tutorial solver settings
    solver.system.fvSolution.solvers["U"] = {
        "solver": "smoothSolver",
        "smoother": "GaussSeidel",
        "tolerance": "1e-6",
        "relTol": "0",
        "nSweeps": "1",
    }
    solver.system.fvSolution.solvers["UFinal"] = {
        "$U": "",
        "tolerance": "1e-7",
    }
    solver.system.fvSolution.solvers["p_rgh"] = {
        "solver": "GAMG",
        "tolerance": "1e-8",
        "relTol": "0.01",
        "smoother": "DIC",
        "cacheAgglomeration": "no",
    }
    solver.system.fvSolution.solvers["p_rghFinal"] = {
        "$p_rgh": "",
        "relTol": "0",
    }
    solver.system.fvSolution.solvers["alpha.water.*"] = {
        "nCorrectors": "1",
        "nSubCycles": "polynomial (0 4)",
    }
    solver.system.fvSolution.solvers["pcorr.*"] = {
        "$p_rghFinal": "",
        "tolerance": "1e-4",
    }
    solver.system.fvSolution.solvers.pop("Phi", None)

    # No under-relaxation for transient PIMPLE (matching Template reference).
    # foampilot auto-generates 0.7 relaxation factors which cause instability
    # in transient VoF — override all to 1.0 (no relaxation).
    solver.system.fvSolution.relaxationFactors = {
        "fields": {},
        "equations": {".*": 1},
    }

    # Clean foampilot-generated fvSchemes for VoF:
    # - Remove MULES-style div(phirb,alpha.water) — MPLIC used instead
    # - Remove alpha.air schemes (computed implicitly as 1 - alpha.water)
    # - Remove turbulence div schemes (laminar flow)
    solver.system.fvSchemes.divSchemes.pop("div(phirb,alpha.water)", None)
    solver.system.fvSchemes.divSchemes.pop("div(phirb,alpha.air)", None)
    solver.system.fvSchemes.divSchemes.pop("div(phi,omega)", None)
    solver.system.fvSchemes.divSchemes.pop("div(phi,k)", None)

    # Override VoF div schemes to match the Template reference case:
    # - MPLIC + interfaceCompression vanLeer 1 for alpha (more stable than MULES)
    # - linearUpwindV for momentum (vector version of linearUpwind)
    # - div(rho*nuEff*dev2(T(grad(U)))) for viscous stress (required!)
    # - Remove div(phi,U) — VoF uses div(rhoPhi,U) instead
    solver.system.fvSchemes.divSchemes.pop("div(phi,U)", None)
    solver.system.fvSchemes.divSchemes["div(phi,alpha)"] = "Gauss MPLIC interfaceCompression vanLeer 1"
    solver.system.fvSchemes.divSchemes["div(rhoPhi,U)"] = "Gauss linearUpwindV grad(U)"
    solver.system.fvSchemes.divSchemes["div(((rho*nuEff)*dev2(T(grad(U)))))"] = "Gauss linear"

    solver.system.fvSchemes.laplacianSchemes["default"] = "Gauss linear corrected"
    solver.system.fvSchemes.snGradSchemes["default"] = "corrected"
    solver.system.write()

    # ==================================================================
    # 3. Maillage (blockMesh)
    # ==================================================================
    print("1. Generation du maillage (blockMesh) ...")
    mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = mesh.mesher

    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],
        [0.1, 0, 0],
        [0.1, 0.1, 0],
        [0, 0.1, 0],
        [0, 0, 0.1],
        [0.1, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0, 0.1, 0.1],
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (40 40 40) simpleGrading (1 1 1)"
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"defaultFaces": "empty"}
    blockmesh.boundary = {
        "atmosphere": {
            "type": "patch",
            "faces": [[4, 5, 6, 7]],
        },
        "walls": {
            "type": "wall",
            "faces": [
                [0, 4, 7, 3],
                [1, 2, 6, 5],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [0, 3, 2, 1],
            ],
        },
    }
    blockmesh.mergePatchPairs = []

    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()

    # Create annulus inlet patch at center of top face via createPatch
    Functions.write_create_patch_dict(
        name="createPatchDict",
        base_path=case_path,
        folder="system",
        patches=[
            {
                "name": "inlet",
                "patchInfo": {"type": "patch"},
                "constructFrom": "zone",
                "zone": {
                    "type": "annulus",
                    "point1": [0.05, 0.05, 0.09995],
                    "point2": [0.05, 0.05, 0.101],
                    "outerRadius": 0.005,
                    "innerRadius": 0,
                },
            }
        ],
    )
    solver.run_command(
        ["createPatch", "-overwrite", "-case", str(case_path)],
        log_filename="log.createPatch",
    )

    # Create cellZone for dynamic mesh refinement
    Functions.write_create_zones_dict(
        name="createZonesDict",
        base_path=case_path,
        folder="system",
        zones=[
            {
                "refineZone": {
                    "type": "box",
                    "zoneType": "cell",
                    "boxes": [[-1, -1, -1], [1, 1, 0.084]],
                }
            }
        ],
    )
    solver.run_command(
        ["createZones", "-case", str(case_path)],
        log_filename="log.createZones",
    )

    # Enable dynamic mesh refinement
    Functions.write_dynamic_mesh_dict(
        name="dynamicMeshDict",
        base_path=case_path,
        folder="constant",
        refinement_regions=[{
            "name": "refineZone",
            "refineZone": "refineZone",
            "cellZone": "refineZone",
            "field": "alpha.water",
            "lowerRefineLevel": 0.001,
            "upperRefineLevel": 0.999,
            "maxRefinement": 1,
        }],
    )

    # Write decomposeParDict (used if nb_proc >= 2) via foampilot's DecomposeParDictFile
    decompose_file = DecomposeParDictFile(parent=solver._solver, nb_proc=48)
    decompose_file.write(case_path / "system" / "decomposeParDict")

    # ==================================================================
    # 4. Fichiers constants (two-phase VoF)
    # ==================================================================
    print("2. Ecriture des proprietes physiques (two-phase VoF) ...")
    solver.constant.write()

    # ==================================================================
    # 5. Conditions aux limites
    # ==================================================================
    print("3. Configuration des conditions aux limites ...")
    solver.boundary.initialize_boundary()

    # Remove alpha.air — interFoam computes alpha.air = 1 - alpha.water implicitly
    solver.boundary.fields.pop("alpha.air", None)
    if "alpha.air" in solver._solver.fields_manager.fields:
        solver._solver.fields_manager.fields.pop("alpha.air")

    # alpha.water — conditions aux limites
    solver.boundary.set_raw_condition("inlet", "alpha.water", {
        "type": "fixedValue",
        "value": "uniform 1",
    })
    solver.boundary.set_raw_condition("atmosphere", "alpha.water", {
        "type": "inletOutlet",
        "inletValue": "uniform 0",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("walls", "alpha.water", {
        "type": "zeroGradient",
    })

    # U — vitesse
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "uniformFixedValue",
        "uniformValue": "constant (0 0 -0.5)",
    })
    solver.boundary.set_raw_condition("atmosphere", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
    })
    solver.boundary.set_raw_condition("walls", "U", {
        "type": "uniformFixedValue",
        "uniformValue": "constant (0 0 0)",
    })

    # p_rgh — pression
    solver.boundary.set_raw_condition("inlet", "p_rgh", {
        "type": "prghTotalPressure",
        "psi": "none",
        "gamma": 1,
        "p0": "$internalField",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("atmosphere", "p_rgh", {
        "type": "prghTotalPressure",
        "psi": "none",
        "gamma": 1,
        "p0": "$internalField",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("walls", "p_rgh", {
        "type": "fixedFluxPressure",
        "gradient": "$internalField",
        "value": "$internalField",
    })

    # Write boundary condition files — OpenFOAMFile.write_boundary_file now
    # adds #includeEtc "caseDicts/setConstraintTypes" by default and uses
    # correct field dimensions (p_rgh → [1 -1 -2 0 0 0 0], alpha.water → []).
    solver.boundary.write_boundary_conditions()

    # ==================================================================
    # 6. setFields (initialisation avec un goutte d'eau)
    # ==================================================================
    print("4. Initialisation avec setFields ...")

    Functions.write_set_fields_dict(
        name="setFieldsDict",
        base_path=case_path,
        folder="system",
        default_values={"alpha.water": "0"},
        zones=[
            {
                "name": "waterDrop",
                "type": "sphere",
                "centre": [0.05, 0.05, 0.09],
                "radius": 0.0035,
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

    # ==================================================================
    # 7. Lancement de la simulation
    # ==================================================================
    print("\n" + "=" * 60)
    print("Lancement de la simulation VoF (incompressibleVoF)")
    print("=" * 60)
    solver.run_simulation()

    # ==================================================================
    # 8. Post-traitement
    # ==================================================================
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
