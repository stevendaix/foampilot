"""OpenFOAM 13 XiFluid/moriyoshiHomogeneous generated declaratively."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from foampilot import Meshing
from foampilot.solver import Solver
from foampilot.solver.environment import OpenFOAMEnvironment
from foampilot.utilities.dictonnary import OpenFOAMDictAddFile


def write_block_mesh(solver):
    mesh = Meshing(solver.case_path, mesher="blockMesh")
    block = mesh.mesher
    block.scale = 0.001
    block.vertices = [[0, 0, 0], [0, 35, 0], [70, 0, 0], [70, 35, 0],
                      [0, 0, 1], [0, 35, 1], [70, 0, 1], [70, 35, 1]]
    block.blocks = ["hex (0 2 3 1 4 6 7 5) (70 35 1) simpleGrading (1 1 1)"]
    block.edges = []
    block.boundary = {
        "left": {"type": "symmetryPlane", "faces": [[0, 4, 5, 1]]},
        "right": {"type": "symmetryPlane", "faces": [[2, 3, 7, 6]]},
        "top": {"type": "symmetryPlane", "faces": [[1, 5, 7, 3]]},
        "bottom": {"type": "symmetryPlane", "faces": [[0, 2, 6, 4]]},
        "frontAndBack": {"type": "empty", "faces": [[4, 5, 7, 6], [0, 1, 3, 2]]},
    }
    block.mergePatchPairs = []
    block.write(solver.case_path / "system" / "blockMeshDict")
    block.run()


def write_xifluid_properties(solver, hydrogen):
    thermo = {"type": "heheuPsiThermo", "mixture": "homogeneousMixture",
              "transport": "const", "thermo": "janaf",
              "equationOfState": "perfectGas", "specie": "specie", "energy": "absoluteEnthalpy"}
    species = "hydrogen" if hydrogen else "propane"
    reactant_mw = 2.016 if hydrogen else 29.4649
    product_mw = 18.0 if hydrogen else 28.3233
    reactants = {
        "specie": {"molWeight": reactant_mw},
        "thermodynamics": {"Tlow": 200, "Thigh": 6000, "Tcommon": 1000,
                           "highCpCoeffs": "(3.24515 0.00202212 -6.98806e-07 1.11477e-10 -6.60444e-15 -1601.58 4.60831)",
                           "lowCpCoeffs": "(3.60909 -0.000628822 4.45105e-06 -3.81328e-09 1.0553e-12 -1587.86 3.21309)"},
        "transport": {"mu": 1e-5, "Pr": 1},
    }
    products = {
        "specie": {"molWeight": product_mw},
        "thermodynamics": {"Tlow": 200, "Thigh": 6000, "Tcommon": 1000,
                           "highCpCoeffs": "(3.10561 0.00179748 -5.94701e-07 9.05612e-11 -5.08447e-15 -11003.6 5.12109)",
                           "lowCpCoeffs": "(3.498 0.000638554 -1.83885e-07 1.20991e-09 -7.68702e-13 -11080.6 3.1819)"},
        "transport": {"mu": 1e-5, "Pr": 1},
    }
    physical = OpenFOAMDictAddFile(object_name="physicalProperties", thermoType=thermo,
                                   stoichiometricAirFuelMassRatio=34.3 if hydrogen else 15.675,
                                   reactants=reactants, products=products)
    physical.write("physicalProperties", solver.case_path, folder="constant")
    combustion = OpenFOAMDictAddFile(
        object_name="combustionProperties",
        laminarFlameSpeed={"model": "unstrained", "unstrainedLaminarFlameSpeed": {
            "model": "Gulder", "equivalenceRatio": 1, "fuel": "Hydrogen" if hydrogen else "Propane",
            ("Hydrogen" if hydrogen else "Propane"): {"W": 0.446 if not hydrogen else 0.435,
                "eta": 0.12, "xi": 4.95, "alpha": 1.77, "beta": -0.2, "f": 2.3},
        }},
        flameWrinkling={"model": "transport", "equilibrium": {"model": "Gulder"},
                        "profile": {"model": "linear"}, "generationRate": {"model": "KTS"}},
    )
    combustion.write("combustionProperties", solver.case_path, folder="constant")
    OpenFOAMDictAddFile(object_name="momentumTransport", simulationType="RAS",
                        RAS={"model": "kEpsilon", "turbulence": "on", "printCoeffs": "on"}).write(
                            "momentumTransport", solver.case_path, folder="constant")
    OpenFOAMDictAddFile(object_name="fvModels", models={}).write("fvModels", solver.case_path, folder="constant")


def write_fields(solver, hydrogen):
    solver.boundary.initialize_boundary()
    patches = ("left", "right", "top", "bottom", "frontAndBack")
    fields = ("T", "Tu", "U", "Xi", "b", "ft", "fu", "alphat", "k", "epsilon", "nut", "p")
    for field in fields:
        solver.boundary.fields.setdefault(field, {})
    for patch in patches:
        for field in fields:
            if patch == "frontAndBack":
                condition = {"type": "empty"}
            elif field == "U":
                condition = {"type": "symmetry"}
            else:
                condition = {"type": "symmetry"}
            solver.boundary.set_raw_condition(patch, field, condition)
    fuel_temperature = 300.0 if not hydrogen else 300.0
    solver.boundary.write_boundary_conditions({
        "T": f"uniform {fuel_temperature}", "Tu": f"uniform {fuel_temperature}",
        "U": "uniform (0 0 0)", "Xi": "uniform 1", "b": "uniform 1",
        "ft": "uniform 0", "fu": "uniform 0", "alphat": "uniform 0",
        "k": "uniform 0.23", "epsilon": "uniform 0.125", "nut": "uniform 0",
        "p": "uniform 1e5",
    })


def prepare_case(case_path, hydrogen=False):
    solver = Solver(case_path)
    solver.compressible = True
    solver.solver_name = "XiFluid"
    solver.transient = True
    solver.setup_case()
    solver.system.controlDict.application = "XiFluid"
    solver.system.controlDict.startTime = 0
    solver.system.controlDict.endTime = 0.015
    solver.system.controlDict.deltaT = 1e-5
    solver.system.controlDict.writeInterval = 50
    solver.system.controlDict.adjustTimeStep = True
    solver.system.controlDict.maxCo = 0.5
    solver.system.fvSchemes.divSchemes.update({
        "div(phi,U)": "Gauss limitedLinearV 1",
        "div(phi,K)": "Gauss limitedLinear 1",
        "div(phi,(p|rho))": "Gauss limitedLinear 1",
        "div(phiXi,Xi)": "Gauss limitedLinear 1",
        "div(phiSt,b)": "Gauss limitedLinear01 1",
        "div(phi,k)": "Gauss limitedLinear 1",
        "div(phi,epsilon)": "Gauss limitedLinear 1",
        "div(phi,omega)": "Gauss limitedLinear 1",
        "div(phi,ft_b_ha_hau)": "Gauss multivariateSelection { fu limitedLinear01 1; ft limitedLinear01 1; b limitedLinear01 1; ha limitedLinear 1; hau limitedLinear 1; }",
    })
    solver.system.fvSolution.solvers.update({
        "rho.*": {"solver": "diagonal"}, "rhoFinal": {"solver": "diagonal"},
        "p": {"solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": "1e-8", "relTol": "0.01"},
        "pFinal": {"solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": "1e-8", "relTol": "0"},
        "(U|h|k|omega|ft|fu|b|Xi)": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "ha": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "hau": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "Su": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "haFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
        "hauFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
        "SuFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
    })
    solver.system.fvSolution.PIMPLE = {"momentumPredictor": "yes", "nOuterCorrectors": 2,
                                         "nCorrectors": 2, "nNonOrthogonalCorrectors": 0}
    solver.system.write()
    solver.constant.write()
    write_xifluid_properties(solver, hydrogen)
    write_block_mesh(solver)
    write_fields(solver, hydrogen)
    return solver


def main():
    os.environ.update(OpenFOAMEnvironment().environment())
    prepare_case(Path.cwd() / "moriyoshiHomogeneous", hydrogen=False).run_simulation(nb_proc=1, log_filename="log.XiFluid.propane")
    prepare_case(Path.cwd() / "moriyoshiHomogeneousHydrogen", hydrogen=True).run_simulation(nb_proc=1, log_filename="log.XiFluid.hydrogen")


if __name__ == "__main__":
    main()
