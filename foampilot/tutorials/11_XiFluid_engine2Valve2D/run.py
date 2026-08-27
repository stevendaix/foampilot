"""OpenFOAM 13 XiFluid/engine2Valve2D through FoamPilot only."""
from math import atan2, cos, sin, sqrt
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot import Meshing
from foampilot.solver import Solver, OpenFOAMEnvironment
from foampilot.utilities import OpenFOAMDictAddFile

REFERENCE = Path("/opt/openfoam13/tutorials/XiFluid/engine2Valve2D")
TIMES = [0, 100, 120, 140, 180, 200, 220, 300, 340, 345, 350, 360, 370, 380, 390, 410, 440, 460, 520, 550, 580, 600, 610, 620]
SCALE = 0.1
TDC_CLEARANCE = 4.0


def triangular_lift(begin, interval, start, rate, current):
    mid, end = begin + interval, begin + 2 * interval
    if current < begin or current >= end:
        return 0.0
    if current < mid:
        return start + rate * (current - begin)
    return start + rate * (2 * mid - current - begin)


def valve_lift(kind, angle):
    return triangular_lift(340 if kind == "intake" else 100, 140, 0.1, 0.02, angle)


def piston_position(angle):
    theta = atan2(0, -1) * angle / 180.0
    conrod = 1000.0 / SCALE
    radius = (1.0 / SCALE) / 2.0
    r_st = radius * sin(theta)
    r = radius * cos(theta) + sqrt((conrod - r_st) * (conrod + r_st))
    return conrod + radius - r + TDC_CLEARANCE


def configure_xifluid_declarative_case(solver):
    """Configure the XiFluid case through FoamPilot writers only.

    Mesh templates are generated separately by ``write_cylinder_mesh`` and
    ``write_valve_mesh``; the remaining dictionaries are added incrementally
    as reusable XiFluid builders are implemented.
    """
    solver.system.controlDict.application = "XiFluid"
    solver.system.fvSolution.solvers = {
        "rho.*": {"solver": "diagonal"},
        "rhoFinal": {"solver": "diagonal"},
        "epsilon": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "epsilonFinal": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
        "p": {"solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": "1e-8", "relTol": "0.01"},
        "pFinal": {"solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": "1e-8", "relTol": "0"},
        "pcorr.*": {"solver": "PCG", "preconditioner": "DIC", "tolerance": "1e-2", "relTol": "0"},
        "MeshPhi": {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": "1e-2", "relTol": "0"},
        "(U|h|k|omega|ft|fu|egr|b|Su|Xi|ha|hau)": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0.01"},
        "(U|h|k|omega|ft|fu|egr|b|Su|Xi|ha|hau)Final": {"solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": "1e-8", "relTol": "0"},
    }
    solver.system.fvSolution.PIMPLE = {
        "momentumPredictor": "yes", "nOuterCorrectors": 2, "nCorrectors": 2,
        "nNonOrthogonalCorrectors": 0, "correctPhi": "yes",
        "correctMeshPhi": "no", "checkMeshCourantNo": "no",
    }
    solver.system.fvSchemes.divSchemes.update({
        "div(phi,U)": "Gauss limitedLinearV 1",
        "div(phi,ft_b_ha_hau)": "Gauss multivariateSelection { fu limitedLinear01 1; ft limitedLinear01 1; b limitedLinear01 1; egr limitedLinear01 1; ha limitedLinear 1; hau limitedLinear 1; }",
        "div(phi,K)": "Gauss limitedLinear 1",
        "div(phi,(p|rho))": "Gauss limitedLinear 1",
        "div(phiXi,Xi)": "Gauss limitedLinear 1",
        "div(phiSt,b)": "Gauss limitedLinear01 1",
        "div(phi,k)": "Gauss limitedLinear 1",
        "div(phi,omega)": "Gauss limitedLinear 1",
        "div(((rho*nuEff)*dev2(T(grad(U)))))": "Gauss linear",
    })
    mirror = OpenFOAMDictAddFile(
        object_name="mirrorMeshDict",
        planeType="pointAndNormal", point="(0 0 0)",
        normal="(1 0 0)", planeTolerance=1e-6,
    )
    mirror.write("mirrorMeshDict", solver.case_path)
    baffles = OpenFOAMDictAddFile(
        object_name="createBafflesDict", internalFacesOnly="true", height=5,
        evMeshLeft="$blockMeshDict.cylinder!x1a",
        evMeshSpan='#calc "$blockMeshDict.cylinder!x0b - $evMeshLeft"',
        ivMeshLeft="$blockMeshDict.cylinder!x1b",
        ivMeshSpan='#calc "$blockMeshDict.cylinder!x0c - $evMeshLeft"',
        baffles={
            "baffles_ev_port": {
                "type": "surface", "surface": "plate",
                "origin": "($evMeshLeft $height -1e3)",
                "span": "($evMeshSpan 0 2e3)",
                "owner": {"name": "nonCouple_ep_ev", "type": "patch"},
                "neighbour": {"name": "nonCouple_ev_ep", "type": "patch"},
            },
            "baffles_iv_port": {
                "type": "surface", "surface": "plate",
                "origin": "($ivMeshLeft $height -1e3)",
                "span": "($ivMeshSpan 0 2e3)",
                "owner": {"name": "nonCouple_ip_iv", "type": "patch"},
                "neighbour": {"name": "nonCouple_iv_ip", "type": "patch"},
            },
        },
    )
    baffles.write("createBafflesDict", solver.case_path)
    inlet_fuel = OpenFOAMDictAddFile(
        object_name="createPatchDict",
        patches=[{
            "name": "inletFuel",
            "patchInfo": {"type": "patch"},
            "constructFrom": "zone",
            "zone": {"type": "box", "box": "(-0.035 -0.1 -1e-5) (0.035 0.1 1e-5)"},
        }],
    )
    inlet_fuel.write("createPatchDict.inletFuel", solver.case_path)
    ncc = OpenFOAMDictAddFile(
        object_name="createNonConformalCouplesDict", fields="no",
        nonConformalCouples={
            "NCC_iv": {"patches": "(nonCouple_iv_cyl nonCouple_cyl_iv)", "transform": "none"},
            "NCC_ev": {"patches": "(nonCouple_ev_cyl nonCouple_cyl_ev)", "transform": "none"},
            "NCC_iv_port": {"patches": "(nonCouple_ip_iv nonCouple_iv_ip)", "transform": "none"},
            "NCC_ev_port": {"patches": "(nonCouple_ep_ev nonCouple_ev_ep)", "transform": "none"},
        },
    )
    ncc.write("createNonConformalCouplesDict", solver.case_path)
    thermo_type = {
        "type": "heheuPsiThermo", "mixture": "inhomogeneousEGRMixture",
        "transport": "polynomial", "thermo": "janaf",
        "equationOfState": "perfectGas", "specie": "specie", "energy": "absoluteEnthalpy",
    }
    thermo_files = {
        "fuel_thermo.foam": ("fuel", 16.043),
        "oxidant_thermo.foam": ("oxidant", 28.85064),
        "products_thermo.foam": ("burntProducts", 27.25975),
    }
    for filename, (species_name, mol_weight) in thermo_files.items():
        species = OpenFOAMDictAddFile(
            object_name=filename,
            **{species_name: {
                "specie": {"nMoles": 1, "molWeight": mol_weight},
                "thermodynamics": {
                    "Tlow": 280.0, "Thigh": 3000.0, "Tcommon": 1000.0,
                    "lowCpCoeffs": "(3.5 0 0 0 0 -10000 4)",
                    "highCpCoeffs": "(3.5 0 0 0 0 -10000 4)",
                },
                "transport": {
                    "As": 1.7e-6, "Ts": 250.0,
                    "muLogCoeffs<8>": "(-20 2 -0.2 0.01 0 0 0 0)",
                    "muCoeffs<8>": "(4e-6 3e-8 -7e-12 9e-16 0 0 0 0)",
                    "kappaLogCoeffs<8>": "(0 -2 0.4 -0.02 0 0 0 0)",
                    "kappaCoeffs<8>": "(0.003 7e-5 -1e-8 5e-13 0 0 0 0)",
                },
            }},
        )
        species.write(filename, solver.case_path, folder="constant")
    physical = OpenFOAMDictAddFile(
        object_name="physicalProperties", includes=[
            "$FOAM_CASE/constant/fuel_thermo.foam",
            "$FOAM_CASE/constant/oxidant_thermo.foam",
            "$FOAM_CASE/constant/products_thermo.foam",
        ], thermoType=thermo_type,
        stoichiometricAirFuelMassRatio=17.23,
    )
    physical.write("physicalProperties", solver.case_path, folder="constant")
    transport = OpenFOAMDictAddFile(
        object_name="transportProperties", thermoType=thermo_type,
        stoichiometricAirFuelMassRatio=17.23,
    )
    transport.write("transportProperties", solver.case_path, folder="constant")
    combustion = OpenFOAMDictAddFile(
        object_name="combustionProperties",
        laminarFlameSpeed={
            "model": "unstrained",
            "unstrainedLaminarFlameSpeed": {
                "model": "Gulder", "fuel": "Methane",
                "Methane": {"W": 0.422, "eta": 0.15, "xi": 5.18, "alpha": 2, "beta": -0.5, "f": 2.3},
            },
        },
        flameWrinkling={
            "model": "equilibrium",
            "equilibrium": {"model": "Gulder", "SuMin": 0.001},
            "profile": {"model": "cubic"},
            "generationRate": {"model": "KTS"},
        },
    )
    combustion.write("combustionProperties", solver.case_path, folder="constant")


def write_cylinder_mesh(solver, piston, n_piston, destination):
    """Generate the engine cylinder blockMeshDict without importing a template."""
    blockmesh = Meshing(solver.case_path, mesher="blockMesh").mesher
    half_width = 26.0 / 2.0
    section = 2.0 / 2.0
    x0a, x1a = -half_width, -half_width + section
    x0b, x1b = -section, section
    x0c, x1c = half_width - section, half_width
    vertices = []
    for x0, x1 in ((x0a, x1a), (x0b, x1b), (x0c, x1c)):
        vertices.extend([(x0, 0.0, 0.0), (x1, 0.0, 0.0), (x0, piston, 0.0), (x1, piston, 0.0)])
        vertices.extend([(x0, 0.0, 0.5), (x1, 0.0, 0.5), (x0, piston, 0.5), (x1, piston, 0.5)])
    blockmesh.scale = 1.0
    blockmesh.definitions = [
        "x1a -12;", "x0b -1;", "x1b 1;", "x0c 12;",
    ]
    blockmesh.vertices = vertices
    blockmesh.blocks = [
        f"hex (0 2 3 1 4 6 7 5) ({n_piston} 6 1) simpleGrading (1 1 1)",
        f"hex (8 10 11 9 12 14 15 13) ({n_piston} 6 1) simpleGrading (1 1 1)",
        f"hex (16 18 19 17 20 22 23 21) ({n_piston} 6 1) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"name": "frontAndBack", "type": "empty"}
    blockmesh.boundary = {
        "piston": {"type": "wall", "faces": ["(3 2 6 7)", "(11 10 14 15)", "(19 18 22 23)"]},
        "liner": {"type": "wall", "faces": ["(2 0 4 6)", "(19 23 21 17)"]},
        "cylinderHead": {"type": "wall", "faces": ["(0 1 5 4)", "(8 9 13 12)", "(16 17 21 20)"]},
        "nonCouple_cyl_ev": {"type": "patch", "faces": ["(3 7 5 1)", "(10 8 12 14)"]},
        "nonCouple_cyl_iv": {"type": "patch", "faces": ["(18 16 20 22)", "(11 15 13 9)"]},
    }
    blockmesh.write(destination)


def write_valve_mesh(solver, lift, piston, n_piston, n_valve, n_inlet, destination, opened):
    """Generate a valve blockMeshDict from geometric parameters."""
    blockmesh = Meshing(solver.case_path, mesher="blockMesh").mesher
    x0 = (3.0 * 2.0 - 26.0) / 4.0
    x1 = x0 + 1.0
    x2 = x1 + 1.0
    x3 = -1.0
    x4 = 0.0
    y0, y1, y2, y3, y4 = lift, piston, 1.0, 8.0, lift + 1.0
    base = [(x0,0,y0),(x1,0,y0),(x0,y0,0),(x1,y0,0),(x0,y1,0),(x1,y1,0),
            (x2,y3,0),(x2,y2,0),(x2,y4,0),(x3,y3,0),(x3,y2,0),(x3,y4,0),
            (x4,y0,0),(x4,y1,0)]
    # Correct the first four coordinates to match the source's x-z plane layout.
    base = [(x0, 0.0, 0.0), (x1, 0.0, 0.0), (x0, y0, 0.0), (x1, y0, 0.0),
            (x0, y1, 0.0), (x1, y1, 0.0), (x2, y3, 0.0), (x2, y2, 0.0),
            (x2, y4, 0.0), (x3, y3, 0.0), (x3, y2, 0.0), (x3, y4, 0.0),
            (x4, y0, 0.0), (x4, y1, 0.0)]
    vertices = base + [(x, y, 0.5) for x, y, _ in base]
    blockmesh.vertices = vertices
    blockmesh.scale = 1.0
    if opened:
        blocks = [
            f"hex (0 2 3 1 14 16 17 15) ({n_valve} 4 1) simpleGrading (1 1 1)",
            f"hex (2 4 5 3 16 18 19 17) ({n_piston} 4 1) simpleGrading (1 1 1)",
            f"hex (1 3 8 7 15 17 22 21) ({n_valve} 4 1) simpleGrading (1 1 1)",
            f"hex (7 8 11 10 21 22 25 24) ({n_valve} 8 1) simpleGrading (1 1 1)",
            f"hex (6 7 10 9 20 21 24 23) ({n_inlet} 8 1) simpleGrading (1 1 1)",
            f"hex (3 5 13 12 17 19 27 26) ({n_piston} 15 1) simpleGrading (1 1 1)",
        ]
        boundary = {
            "piston": {"type":"wall", "faces":["(4 5 18 19)","(5 13 19 27)"]},
            "liner": {"type":"wall", "faces":["(4 2 18 16)","(2 0 16 14)"]},
            "cylinderHead": {"type":"wall", "faces":["(0 1 15 14)","(1 7 21 15)","(7 6 20 21)"]},
            "inlet": {"type":"patch", "faces":["(6 9 23 20)"]},
            "valveHead": {"type":"wall", "faces":["(8 11 25 22)","(3 8 22 17)","(12 3 17 26)"]},
            "valveStem": {"type":"wall", "faces":["(9 10 24 23)","(10 11 25 24)"]},
            "symmetry": {"type":"patch", "faces":["(13 27 26 12)"]},
        }
    else:
        blocks = [
            f"hex (2 4 5 3 16 18 19 17) ({n_piston} 4 1) simpleGrading (1 1 1)",
            f"hex (6 7 10 9 20 21 24 23) ({n_inlet} 8 1) simpleGrading (1 1 1)",
            f"hex (3 5 13 12 17 19 27 26) ({n_piston} 15 1) simpleGrading (1 1 1)",
        ]
        boundary = {
            "piston": {"type":"wall", "faces":["(4 5 18 19)","(5 13 19 27)"]},
            "liner": {"type":"wall", "faces":["(4 2 18 16)"]},
            "cylinderHead": {"type":"wall", "faces":["(2 3 17 16)","(7 6 20 21)"]},
            "inlet": {"type":"patch", "faces":["(6 9 23 20)"]},
            "valveHead": {"type":"wall", "faces":["(7 21 24 10)","(12 3 17 26)"]},
            "valveStem": {"type":"wall", "faces":["(9 10 24 23)"]},
            "symmetry": {"type":"patch", "faces":["(12 13 27 26)"]},
        }
    blockmesh.blocks = blocks
    blockmesh.edges = []
    blockmesh.defaultPatch = {"name":"frontAndBack", "type":"empty"}
    blockmesh.boundary = boundary
    blockmesh.write(destination)


def configure_xifluid_fields(solver):
    """Generate XiFluid initial fields and generic valid boundary conditions."""
    solver.boundary.initialize_boundary()
    patches = (
        "frontAndBack", "piston", "liner", "cylinderHead", "inlet", "inletFuel",
        "symmetry", "valveHead", "valveStem", "evHead", "evStem", "ivHead", "ivStem",
        "outlet", "nonCouple_ev_cyl", "nonCouple_cyl_ev", "nonCouple_iv_cyl", "nonCouple_cyl_iv",
        "nonCouple_ep_ev", "nonCouple_ev_ep", "nonCouple_ip_iv", "nonCouple_iv_ip",
        "NCC_iv_on_nonCouple_iv_cyl", "NCC_iv_on_nonCouple_cyl_iv",
        "NCC_ev_on_nonCouple_ev_cyl", "NCC_ev_on_nonCouple_cyl_ev",
        "NCC_iv_port_on_nonCouple_ip_iv", "NCC_iv_port_on_nonCouple_iv_ip",
        "NCC_ev_port_on_nonCouple_ep_ev", "NCC_ev_port_on_nonCouple_ev_ep",
        "nonConformalError_on_nonCouple_iv_cyl", "nonConformalError_on_nonCouple_cyl_iv",
        "nonConformalError_on_nonCouple_ev_cyl", "nonConformalError_on_nonCouple_cyl_ev",
        "nonConformalError_on_nonCouple_ip_iv", "nonConformalError_on_nonCouple_iv_ip",
        "nonConformalError_on_nonCouple_ep_ev", "nonConformalError_on_nonCouple_ev_ep",
    )
    scalar_fields = ("T", "Tu", "Xi", "b", "egr", "ft", "fu", "alphat", "k", "nut", "omega", "epsilon", "p")
    for field in scalar_fields + ("U",):
        solver.boundary.fields.setdefault(field, {})
    for patch in patches:
        for field in scalar_fields:
            solver.boundary.set_raw_condition(patch, field, {"type": "zeroGradient"})
        solver.boundary.set_raw_condition(patch, "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("inlet", "U", {"type": "fixedValue", "value": "uniform (0 0 0)"})
    solver.boundary.set_raw_condition("inlet", "T", {"type": "fixedValue", "value": "uniform 300"})
    solver.boundary.write_boundary_conditions({
        "U": "uniform (0 0 0)", "T": "uniform 300", "Tu": "uniform 0.01",
        "Xi": "uniform 1", "b": "uniform 0", "egr": "uniform 0",
        "ft": "uniform 0", "fu": "uniform 0", "alphat": "uniform 0",
        "k": "uniform 1e-6", "nut": "uniform 1e-6", "omega": "uniform 1",
        "p": "uniform 1e5",
    })


def build_meshes(solver, mesh):
    system = solver.system
    cylinder = solver.case_path / "system" / "blockMeshDict.cylinder"
    closed = solver.case_path / "system" / "blockMeshDict.valveClosed"
    opened = solver.case_path / "system" / "blockMeshDict.valveOpen"
    for angle in TIMES:
        piston = piston_position(angle)
        intake = valve_lift("intake", angle)
        exhaust = valve_lift("exhaust", angle)
        n_piston = 3 + 3 * round(piston)
        n_intake = 3 + 3 * round(max(piston - intake, 0))
        n_exhaust = 3 + 3 * round(max(piston - exhaust, 0))
        write_cylinder_mesh(solver, -piston, n_piston, cylinder)
        system.run_utility("blockMesh", ["-mesh", str(angle), "-dict", str(cylinder)], f"log.blockMesh.cylinder.{angle}")
        for name, lift, n_valve, dict_path, kind in (("_tmp_exhaust", exhaust, n_exhaust, opened if exhaust else closed, "exhaust"), ("_tmp_intake", intake, n_intake, opened if intake else closed, "intake")):
            write_valve_mesh(solver, -lift, -piston, n_valve, 3 + 3 * round(abs(lift)), 3 + 3 * round(abs(piston)), dict_path, bool(lift))
            system.run_utility("blockMesh", ["-mesh", name, "-dict", str(dict_path)], f"log.blockMesh.{kind}.{angle}")
            system.run_utility("mirrorMesh", ["-mesh", name], f"log.mirrorMesh.{kind}.{angle}")
            offset = -6.0 if kind == "exhaust" else 6.0
            system.run_utility("transformPoints", ["-mesh", name, f"translate=({offset:g} 0 0)"], f"log.transformPoints.translate.{kind}.{angle}")
            valve = "ev" if kind == "exhaust" else "iv"
            boundary_path = solver.case_path / "constant" / "meshes" / name / "polyMesh" / "boundary"
            system.rename_dictionary_entries(boundary_path, {
                f"entry0/valveHead": f"{valve}Head",
                f"entry0/valveStem": f"{valve}Stem",
                f"entry0/liner": f"nonCouple_{valve}_cyl",
            })
            system.update_dictionary_entries(boundary_path, {f"entry0/nonCouple_{valve}_cyl/type": "patch"})
            if kind == "exhaust":
                system.rename_dictionary_entries(boundary_path, {"entry0/inlet": "outlet"})
            system.remove_dictionary_entries(boundary_path, [f"entry0/nonCouple_{valve}_cyl/inGroups"])
        system.run_utility("mergeMeshes", ["-mesh", str(angle), "-addMeshes", "(_tmp_exhaust _tmp_intake)"], f"log.mergeMeshes.{angle}")
        system.run_utility("createBaffles", ["-mesh", str(angle), "-dict", "system/createBafflesDict"], f"log.createBaffles.{angle}")
        system.run_utility("splitBaffles", ["-mesh", str(angle)], f"log.splitBaffles.{angle}")
        system.run_utility("transformPoints", ["-mesh", str(angle), "Rx=90, scale=(0.1 0.1 0.1)"], f"log.transformPoints.scale.{angle}")
        system.run_utility("createPatch", ["-mesh", str(angle), "-dict", "system/createPatchDict.inletFuel"], f"log.createPatch.{angle}")
        system.run_utility("createNonConformalCouples", ["-mesh", str(angle), "-dict", "system/createNonConformalCouplesDict"], f"log.createNonConformalCouples.{angle}")
    mesh.mesher.copy_mesh("0")
    mesh.mesher.write_mesh_times(TIMES)


def main():
    case_path = Path.cwd()
    os.environ.update(OpenFOAMEnvironment().environment())
    solver = Solver(case_path)
    solver.compressible = True
    solver.solver_name = "XiFluid"
    solver.transient = True
    solver.system.controlDict.application = "XiFluid"
    solver.setup_case()
    solver.system.write()
    solver.constant.write()
    configure_xifluid_declarative_case(solver)
    solver.system.write()
    mesh = Meshing(case_path, mesher="blockMesh")
    build_meshes(solver, mesh)
    configure_xifluid_fields(solver)
    solver.run_simulation(nb_proc=1, log_filename="log.XiFluid")


if __name__ == "__main__":
    main()
