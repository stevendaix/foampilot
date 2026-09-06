"""OpenFOAM 13 XiFluid/engine2Valve2D through FoamPilot only."""
from math import atan2, cos, sin, sqrt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot import Meshing
from foampilot.solver import Solver

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


def import_reference_case(solver):
    for path in (REFERENCE / "system").iterdir():
        if path.is_file() and path.name not in {"blockMeshDict.cylinder.orig", "blockMeshDict.valveClosed.orig", "blockMeshDict.valveOpen.orig"}:
            solver.system.import_reference_file(path)
    for path in (REFERENCE / "constant").iterdir():
        if path.is_file():
            solver.constant.import_reference_file(path)
    for path in (REFERENCE / "0").iterdir():
        if path.is_file() and path.name != "U.orig":
            solver.fields_manager.import_reference_field(path, solver.case_path)
    solver.system.import_reference_file(REFERENCE / "system" / "blockMeshDict.cylinder.orig", "blockMeshDict.cylinder")
    solver.system.import_reference_file(REFERENCE / "system" / "blockMeshDict.valveClosed.orig", "blockMeshDict.valveClosed")
    solver.system.import_reference_file(REFERENCE / "system" / "blockMeshDict.valveOpen.orig", "blockMeshDict.valveOpen")


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
        system.update_dictionary_entries(cylinder, {"pistonPos": f"-{piston:g}", "nPiston": str(n_piston)})
        system.run_utility("blockMesh", ["-mesh", str(angle), "-dict", str(cylinder)], f"log.blockMesh.cylinder.{angle}")
        for name, lift, n_valve, dict_path, kind in (("_tmp_exhaust", exhaust, n_exhaust, opened if exhaust else closed, "exhaust"), ("_tmp_intake", intake, n_intake, opened if intake else closed, "intake")):
            system.update_dictionary_entries(dict_path, {"valveLift": f"-{lift:g}", "pistonPos": f"-{piston:g}", "nPiston": str(n_valve), "nValve": str(3 + 3 * round(lift))})
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
    solver = Solver(case_path)
    solver.solver_name = "XiFluid"
    solver.transient = True
    solver.system.controlDict.application = "XiFluid"
    solver.setup_case()
    solver.system.write()
    solver.constant.write()
    import_reference_case(solver)
    mesh = Meshing(case_path, mesher="blockMesh")
    build_meshes(solver, mesh)
    solver.run_simulation(nb_proc=1, log_filename="log.XiFluid")


if __name__ == "__main__":
    main()
