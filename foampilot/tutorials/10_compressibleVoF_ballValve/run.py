#!/usr/bin/env python3
"""Tutoriel 10 : compressibleVoF/ballValve, OpenFOAM 13.

La géométrie et les dictionnaires système avancés sont importés par les API
FoamPilot dédiées afin de préserver fidèlement les arcs, projections et
réglages MULES/PIMPLE de la référence OF13.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot import Meshing
from foampilot.solver import Solver, OpenFOAMEnvironment


REFERENCE = Path("/opt/openfoam13/tutorials/compressibleVoF/ballValve")
RESOURCE_BLOCK_MESH = Path("/opt/openfoam13/tutorials/resources/blockMesh/ballValve")


def main() -> None:
    os.environ.update(OpenFOAMEnvironment().environment())
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.compressible = True
    solver.is_vof = True
    solver.with_gravity = True
    solver.transient = True
    # Boundary API uses the kEpsilon wall-function schema; the exact OF13
    # realizableKE model is imported below through momentumTransport.
    solver.turbulence_model = "kEpsilon"

    solver.constant.configure_vof(
        phases=["vapour", "water"],
        sigma={"type": "liquidProperties", "phase": "water"},
        phase_properties={
            "vapour": {
                "thermoType": {
                    "type": "heRhoThermo",
                    "mixture": "pureMixture",
                    "transport": "sutherland",
                    "thermo": "janaf",
                    "energy": "sensibleInternalEnergy",
                    "equationOfState": "perfectGas",
                    "specie": "specie",
                },
                "mixture": {
                    "specie": {"molWeight": 18.0153},
                    "thermodynamics": {
                        "Tlow": 200,
                        "Thigh": 5000,
                        "Tcommon": 1000,
                        "highCpCoeffs": "(2.67215 0.00305629 -8.73026e-07 1.201e-10 -6.39162e-15 -29899.2 6.86282)",
                        "lowCpCoeffs": "(3.38684 0.00347498 -6.3547e-06 6.96858e-09 -2.50659e-12 -30208.1 2.59023)",
                    },
                    "transport": {"As": 1.67212e-6, "Ts": 170.672},
                },
            },
            "water": {
                "thermoType": {
                    "type": "heRhoThermo",
                    "mixture": "pureMixture",
                    "properties": "liquid",
                    "energy": "sensibleInternalEnergy",
                },
                "mixture": {"H2O": ""},
            },
        },
    )
    solver.fields_manager.set_vof_primary_phase("vapour")

    solver.system.controlDict.application = "compressibleVoF"
    solver.system.controlDict.startTime = 0
    solver.system.controlDict.endTime = 0.01
    solver.system.controlDict.deltaT = 0.001
    solver.system.controlDict.writeControl = "adjustableRunTime"
    solver.system.controlDict.writeInterval = 0.1
    solver.system.controlDict.adjustTimeStep = True
    solver.system.controlDict.maxCo = 1
    solver.system.controlDict.maxAlphaCo = 1
    solver.system.controlDict.maxDeltaT = 1

    solver.setup_case()
    solver.system.write()
    solver.constant.write()
    solver.constant.import_reference_file(REFERENCE / "constant" / "momentumTransport")
    solver.system.import_reference_file(REFERENCE / "system" / "controlDict")
    solver.system.import_reference_file(REFERENCE / "system" / "createNonConformalCouplesDict")
    solver.system.import_reference_file(REFERENCE / "system" / "fvSchemes")
    solver.system.import_reference_file(REFERENCE / "system" / "fvSolution")
    solver.system.import_reference_file(REFERENCE / "constant" / "fvModels", "fvModels")

    mesh = Meshing(case_path, mesher="blockMesh")
    mesh.mesher.import_reference_dict(RESOURCE_BLOCK_MESH)
    mesh.mesher.import_reference_asset(
        Path("/opt/openfoam13/tutorials/resources/geometry/ballValve-torus.obj.gz"),
        case_path / "constant" / "geometry" / "ballValve-torus.obj",
    )
    mesh.mesher.run()
    mesh.mesher.create_non_conformal_couples()

    solver.boundary.initialize_boundary()
    boundary = solver.boundary
    for obsolete in ("alpha.water", "alpha.air"):
        boundary.fields.pop(obsolete, None)
        solver.fields_manager.fields.pop(obsolete, None)
    wall_patches = ("pipeNonCouple", "ballWalls", "ballNonCouple", "pipeWalls")
    for field in ("alpha.vapour", "T", "U", "p", "p_rgh", "k", "epsilon", "nut"):
        boundary.fields.setdefault(field, {})
    for patch in ("inlet", "lowerOutlet", "upperOutlet"):
        boundary.set_raw_condition(patch, "alpha.vapour", {"type": "inletOutlet", "value": "$internalField", "inletValue": "$internalField"})
        boundary.set_raw_condition(patch, "T", {"type": "fixedValue", "value": "$internalField"} if patch == "inlet" else {"type": "inletOutlet", "value": "$internalField", "inletValue": "$internalField"})
        boundary.set_raw_condition(patch, "U", {"type": "pressureInletOutletVelocity", "value": "uniform (6 0 0)"} if patch == "inlet" else {"type": "pressureInletOutletVelocity", "value": "$internalField"})
        boundary.set_raw_condition(patch, "p", {"type": "calculated", "value": "$internalField"})
        pressure_bc = {"type": "prghTotalPressure", "p0": "uniform 1.18e5"} if patch == "inlet" else {"type": "prghEntrainmentPressure", "p0": "$internalField"}
        boundary.set_raw_condition(patch, "p_rgh", pressure_bc)
        boundary.set_raw_condition(patch, "k", {"type": "turbulentIntensityKineticEnergyInlet", "intensity": 0.05, "value": "$internalField"})
        boundary.set_raw_condition(patch, "epsilon", {"type": "turbulentMixingLengthDissipationRateInlet", "mixingLength": 0.1, "value": "$internalField"})
        boundary.set_raw_condition(patch, "nut", {"type": "calculated", "value": "$internalField"})
    for patch in wall_patches:
        for field in ("alpha.vapour", "T"):
            boundary.set_raw_condition(patch, field, {"type": "zeroGradient"})
        boundary.set_raw_condition(patch, "U", {"type": "noSlip"})
        boundary.set_raw_condition(patch, "p", {"type": "calculated", "value": "$internalField"})
        boundary.set_raw_condition(patch, "p_rgh", {"type": "fixedFluxPressure"})
        boundary.set_raw_condition(patch, "k", {"type": "kqRWallFunction", "value": "$internalField"})
        boundary.set_raw_condition(patch, "epsilon", {"type": "epsilonWallFunction", "value": "$internalField"})
        boundary.set_raw_condition(patch, "nut", {"type": "nutkWallFunction", "value": "$internalField"})
    boundary.write_boundary_conditions({
        "p": "uniform 1e5",
        "p_rgh": "uniform 1e5",
        "T": "uniform 300",
        "alpha.vapour": "uniform 0",
    })

    solver.run_command(["potentialFoam", "-pName", "p_rgh"], "log.potentialFoam")
    solver.run_simulation(nb_proc=1, log_filename="log.compressibleVoF")


if __name__ == "__main__":
    main()
