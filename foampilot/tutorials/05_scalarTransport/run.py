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
    solver.solver_name = "functions"
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kEpsilon"

    # ControlDict -- steady state
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.application = "functions"
    solver.system.controlDict.sub_solver = "incompressibleFluid"
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 0.2
    solver.system.controlDict.deltaT = 1e-4
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 50
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
    # Use the exact OpenFOAM 13 pitzDaily resource through Foampilot.
    import os
    reference_case = Path(os.environ["FOAM_TUTORIALS"]) / "incompressibleFluid" / "pitzDailyScalarTransport"
    mesh_resource = Path(os.environ["FOAM_TUTORIALS"]) / "resources" / "blockMesh" / "pitzDaily"
    solver.run_command(["blockMesh", "-dict", str(mesh_resource)], log_filename="log.blockMesh")

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
    for patch in ("upperWall", "lowerWall"):
        solver.boundary.set_raw_condition(patch, "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("frontAndBack", "U", {"type": "empty"})

    # p
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})
    for patch in ("upperWall", "lowerWall"):
        solver.boundary.set_raw_condition(patch, "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "p", {"type": "empty"})

    # T (scalar passif) -- inlet 300 K, walls zeroGradient
    solver.boundary.set_raw_condition("inlet", "T", {"type": "fixedValue", "value": "uniform 1"})
    solver.boundary.set_raw_condition("outlet", "T", {"type": "zeroGradient"})
    for patch in ("upperWall", "lowerWall"):
        solver.boundary.set_raw_condition(patch, "T", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "T", {"type": "empty"})

    # Write boundary condition files, then reproduce the exact reference fields.
    solver.boundary.write_boundary_conditions()
    Functions.copy_reference_fields(
        reference_case,
        case_path,
        fields=("U", "p", "T", "k", "epsilon", "nut", "phi"),
    )

    # --- 6. Scalar transport function object ---
    # Generate the OpenFOAM 13 function object through Foampilot only.
    scalar_transport = Functions.scalar_transport(
        field="T",
        schemes_field="T",
        diffusivity="constant",
        alphal=1,
        alphat=0.0,
        write_control="timeStep",
        write_interval=50,
    )
    scalar_transport["D"] = 0.01
    Functions.write_function_object("scalarTransport", scalar_transport, case_path)
    mixing_quality = Functions.coded_function_object(
        code_include='#{\n#include "volFields.H"\n#}',
        code_execute='#{\nconst volScalarField& T = mesh().lookupObject<volScalarField>("T");\nconst scalar maxT = max(T).value();\nconst scalar meanT = T.weightedAverage(mesh().V()).value();\nconst scalar mixingQuality = meanT/maxT;\nInfo << "mixingQuality = " << mixingQuality << endl;\nif (mixingQuality > 0.9) const_cast<Time&>(mesh().time()).writeAndEnd();\n#}',
    )
    Functions.write_function_object("mixingQualityCheck", mixing_quality, case_path)


    # --- 7. Lancer la simulation ---
    print("\n" + "=" * 60)
    print("Lancement de la simulation (incompressibleFluid -- scalarTransport)")
    print("=" * 60)
    solver.run_simulation(nb_proc=1)

    # --- 8. Post-traitement ---
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)
    log_file = case_path / "log.functions"
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
    print(f"  Log      : {case_path / 'log.functions'}")
    print(f"  Resultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
