"""OpenFOAM 13 XiFluid/stratified generated declaratively with FoamPilot."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from foampilot import Meshing
from foampilot.solver import Solver
from foampilot.solver.environment import OpenFOAMEnvironment
from foampilot.utilities.dictonnary import OpenFOAMDictAddFile


def build_mesh(solver):
    mesh = Meshing(solver.case_path, mesher="blockMesh")
    block = mesh.mesher
    block.scale = 0.001
    block.vertices = [[0, 0, 0], [0, 35, 0], [70, 0, 0], [70, 35, 0],
                      [0, 0, 1], [0, 35, 1], [70, 0, 1], [70, 35, 1]]
    block.blocks = ["hex (0 2 3 1 4 6 7 5) (70 35 1) simpleGrading (1 1 1)"]
    block.boundary = {
        "left": {"type": "symmetryPlane", "faces": [[0, 4, 5, 1]]},
        "right": {"type": "symmetryPlane", "faces": [[2, 3, 7, 6]]},
        "top": {"type": "symmetryPlane", "faces": [[1, 5, 7, 3]]},
        "bottom": {"type": "symmetryPlane", "faces": [[0, 2, 6, 4]]},
        "frontAndBack": {"type": "empty", "faces": [[4, 5, 7, 6], [0, 1, 3, 2]]},
    }
    block.write(solver.case_path / "system" / "blockMeshDict")
    block.run()


def build_properties(solver):
    thermo = {"type": "heheuPsiThermo", "mixture": "homogeneousMixture", "transport": "const",
              "thermo": "janaf", "equationOfState": "perfectGas", "specie": "specie", "energy": "absoluteEnthalpy"}
    reactants = {"specie": {"molWeight": 29.4649}, "thermodynamics": {"Tlow": 200, "Thigh": 6000, "Tcommon": 1000,
        "highCpCoeffs": "(3.24515 0.00202212 -6.98806e-07 1.11477e-10 -6.60444e-15 -1601.58 4.60831)",
        "lowCpCoeffs": "(3.60909 -0.000628822 4.45105e-06 -3.81328e-09 1.0553e-12 -1587.86 3.21309)"},
        "transport": {"mu": 1e-5, "Pr": 1}}
    products = {"specie": {"molWeight": 28.3233}, "thermodynamics": {"Tlow": 200, "Thigh": 6000, "Tcommon": 1000,
        "highCpCoeffs": "(3.10561 0.00179748 -5.94701e-07 9.05612e-11 -5.08447e-15 -11003.6 5.12109)",
        "lowCpCoeffs": "(3.498 0.000638554 -1.83885e-07 1.20991e-09 -7.68702e-13 -11080.6 3.1819)"},
        "transport": {"mu": 1e-5, "Pr": 1}}
    OpenFOAMDictAddFile(object_name="physicalProperties", thermoType=thermo,
                        stoichiometricAirFuelMassRatio=15.675, reactants=reactants, products=products).write(
                            "physicalProperties", solver.case_path, folder="constant")
    OpenFOAMDictAddFile(object_name="combustionProperties", laminarFlameSpeed={"model": "unstrained",
        "unstrainedLaminarFlameSpeed": {"model": "Gulder", "equivalenceRatio": 1, "fuel": "Propane",
        "Propane": {"W": 0.446, "eta": 0.12, "xi": 4.95, "alpha": 1.77, "beta": -0.2, "f": 2.3}}},
        flameWrinkling={"model": "transport", "equilibrium": {"model": "Gulder"}, "profile": {"model": "linear"},
                        "generationRate": {"model": "KTS"}}).write("combustionProperties", solver.case_path, folder="constant")
    OpenFOAMDictAddFile(object_name="momentumTransport", simulationType="RAS",
                        RAS={"model": "kEpsilon", "turbulence": "on", "printCoeffs": "on"}).write(
                            "momentumTransport", solver.case_path, folder="constant")
    OpenFOAMDictAddFile(object_name="fvModels", models={}).write("fvModels", solver.case_path, folder="constant")


def build_fields(solver):
    solver.boundary.initialize_boundary()
    fields = ("T", "Tu", "U", "Xi", "b", "egr", "ft", "fu", "alphat", "k", "epsilon", "nut", "p")
    patches = ("left", "right", "top", "bottom", "frontAndBack")
    for patch in patches:
        for field in fields:
            solver.boundary.set_raw_condition(patch, field, {"type": "empty"} if patch == "frontAndBack" else {"type": "symmetry"})
    solver.boundary.write_boundary_conditions({"T": "uniform 700", "Tu": "uniform 700", "U": "uniform (0 0 0)",
        "Xi": "uniform 1", "b": "uniform 1", "egr": "uniform 0", "ft": "uniform 0", "fu": "uniform 0",
        "alphat": "uniform 0", "k": "uniform 0.23", "epsilon": "uniform 0.125", "nut": "uniform 0", "p": "uniform 1e5"})


def build_set_fields(solver):
    OpenFOAMDictAddFile(object_name="setFieldsDict", defaultValues={"ft": 0, "fu": 0, "egr": 0}, zones={
        "fuelAir": {"type": "box", "box": "(0 0 -1) (1 0.01 1)", "values": {"ft": 0.07, "fu": 0.07, "egr": 0.01}}
    }).write("setFieldsDict", solver.case_path, folder="system")


def main():
    os.environ.update(OpenFOAMEnvironment().environment())
    solver = Solver(Path.cwd())
    solver.solver_name = "XiFluid"
    solver.compressible = True
    solver.transient = True
    solver.setup_case()
    solver.system.controlDict.application = "XiFluid"
    solver.system.controlDict.endTime = 0.015
    solver.system.controlDict.deltaT = 1e-5
    solver.system.controlDict.writeInterval = 50
    solver.system.controlDict.adjustTimeStep = True
    solver.system.controlDict.maxCo = 0.5
    solver.system.fvSchemes.divSchemes.update({
        "div(phi,U)": "Gauss limitedLinearV 1", "div(phi,h)": "bounded Gauss upwind",
        "div(phi,(p|rho))": "Gauss limitedLinear 1", "div(phiXi,Xi)": "Gauss limitedLinear 1",
        "div(phiSt,b)": "Gauss limitedLinear01 1", "div(phi,k)": "Gauss limitedLinear 1",
        "div(phi,epsilon)": "Gauss limitedLinear 1", "div(phi,ft_b_ha_hau)": "Gauss limitedLinear01 1",
    })
    solver.system.fvSolution.solvers.update({
        "rho.*": {"solver": "diagonal"}, "rhoFinal": {"solver": "diagonal"},
        "p": {"solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": "1e-8", "relTol": "0.01"},
        "pFinal": {"solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": "1e-8", "relTol": "0"},
        "epsilon": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "epsilonFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
        "(U|h|k|ft|fu|b|Xi|ha|hau|Su)": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "haFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
        "hauFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
        "SuFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
    })
    solver.system.fvSolution.PIMPLE = {"momentumPredictor": "yes", "nOuterCorrectors": 2,
                                         "nCorrectors": 2, "nNonOrthogonalCorrectors": 0}
    solver.system.write(); solver.constant.write()
    build_properties(solver); build_mesh(solver); build_fields(solver); build_set_fields(solver)
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_simulation(nb_proc=1, log_filename="log.XiFluid")


if __name__ == "__main__":
    main()
