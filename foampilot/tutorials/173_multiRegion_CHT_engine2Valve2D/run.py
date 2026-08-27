"""OpenFOAM 13 multiRegion/CHT/engine2Valve2D via FoamPilot.

The runner deliberately expands the tutorial's Allmesh operations into
FoamPilot-managed commands; it never invokes the tutorial shell scripts.
"""
from math import cos, pi
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/engine2Valve2D")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
NPROCS = 4
MESH_TIMES = [0, 100, 120, 140, 180, 200, 220, 300, 340, 345, 350, 360, 370,
              380, 390, 410, 440, 460, 520, 550, 580, 600, 610, 620]
LIFTS = {0: (0, 0), 100: (0, .1), 120: (0, .5), 140: (0, .9),
         180: (0, 1.7), 200: (0, 2.1), 220: (0, 2.5), 300: (0, 1.7),
         340: (.1, .9), 345: (.2, .8), 350: (.3, .7), 360: (.5, .5),
         370: (.7, .3), 380: (.9, 0), 390: (1.1, 0), 410: (1.5, 0),
         440: (2.1, 0), 460: (2.5, 0), 520: (2.1, 0), 550: (1.5, 0),
         580: (.9, 0), 600: (.5, 0), 610: (.3, 0), 620: (0, 0)}


def import_reference_case(solver: Solver, case_path: Path) -> None:
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            relative = source.relative_to(REFERENCE / "0")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=str(relative)
            )
    for root in ("constant", "system"):
        for source in (REFERENCE / root).rglob("*"):
            if source.is_file():
                relative = source.relative_to(REFERENCE)
                destination = str(relative)
                if root == "system" and source.name.endswith(".orig"):
                    destination = str(relative.with_name(source.name[:-5]))
                solver.import_reference_asset(source, case_path / destination)


def run(solver: Solver, name: str, *args: str) -> None:
    solver.run_command([str(OF13_BIN / name), *args], log_filename=f"log.{name}.{len(args)}")


def mesh_arg(mesh: int | None) -> list[str]:
    return [] if mesh is None else ["-mesh", str(mesh)]


def set_dict(solver: Solver, path: str, expression: str, tag: str) -> None:
    solver.run_command(
        [str(OF13_BIN / "foamDictionary"), path, "-set", expression],
        log_filename=f"log.foamDictionary.{tag}",
    )


def rename_dict(solver: Solver, path: str, expression: str, tag: str) -> None:
    solver.run_command(
        [str(OF13_BIN / "foamDictionary"), path, "-rename", expression],
        log_filename=f"log.foamDictionary.rename.{tag}",
    )


def create_solids(solver: Solver) -> None:
    set_dict(solver, "system/blockMeshDict.liner",
             "linerTop=0, linerBottom=-30, linerRadius=13, linerThickness=1", "liner")
    run(solver, "blockMesh", "-region", "liner", "-dict", "system/blockMeshDict.liner")
    run(solver, "transformPoints", "-region", "liner", "Rx=90, scale=(0.1 0.1 0.1)")
    for region in ("cylinderHead", "exhaustValve", "intakeValve"):
        run(solver, "blockMesh", "-region", region, "-dict", f"system/blockMeshDict.{region}")
        run(solver, "transformPoints", "-region", region, "Rx=90, scale=(0.1 0.1 0.1)")
    set_dict(solver, "system/blockMeshDict.piston",
             "pPos=-10, pPos2=-15, pRes=18", "piston")
    run(solver, "blockMesh", "-region", "piston", "-dict", "system/blockMeshDict.piston")
    run(solver, "transformPoints", "-region", "piston", "Rx=90, scale=(0.1 0.1 0.1)")


def piston_position(cad: int) -> float:
    theta = cad * pi / 180.0
    stroke = 10.0
    conrod = 10000.0
    r = stroke * cos(theta) / 2.0 + (conrod**2 - (stroke * __import__("math").sin(theta) / 2.0) ** 2) ** 0.5
    return conrod + stroke / 2.0 - r + 10.0


def boundary_edits(solver: Solver, mesh_name: str, side: str) -> None:
    prefix = f"constant/meshes/{mesh_name}/polyMesh/boundary"
    if side == "exhaust":
        rename = "entry0/valveHead=evHead, entry0/valveStem=evStem, entry0/liner=nonCouple_ev_cyl, entry0/inlet=outlet"
        patch = "entry0/nonCouple_ev_cyl/type=patch"
        remove = "entry0/nonCouple_ev_cyl/inGroups"
    else:
        rename = "entry0/valveHead=ivHead, entry0/valveStem=ivStem, entry0/liner=nonCouple_iv_cyl"
        patch = "entry0/nonCouple_iv_cyl/type=patch"
        remove = "entry0/nonCouple_iv_cyl/inGroups"
    rename_dict(solver, prefix, rename, f"{side}.{mesh_name}")
    set_dict(solver, prefix, patch, f"patch.{side}.{mesh_name}")
    solver.run_command(
        [str(OF13_BIN / "foamDictionary"), prefix, "-remove", "-entry", remove],
        log_filename=f"log.foamDictionary.remove.{side}.{mesh_name}",
    )


def create_fluid_mesh(solver: Solver, mesh: int | None, iv_lift: float, ev_lift: float) -> None:
    pos = piston_position(0 if mesh is None else mesh)
    n_iv = 3 + 3 * round(iv_lift)
    n_ev = 3 + 3 * round(ev_lift)
    n_cyl = 3 + 3 * round(pos)
    n_cyl_iv = 3 + 3 * round(pos - iv_lift)
    n_cyl_ev = 3 + 3 * round(pos - ev_lift)
    set_dict(solver, "system/blockMeshDict.cylinder",
             f"pistonPos={-pos}, nPiston={n_cyl}", f"cylinder.{mesh}")
    run(solver, "blockMesh", *mesh_arg(mesh), "-region", "fluid",
        "-dict", "system/blockMeshDict.cylinder")
    for side, lift, n_valve, n_clearance, translate in (
        ("exhaust", ev_lift, n_ev, n_cyl_ev, "translate=(-6 0 0)"),
        ("intake", iv_lift, n_iv, n_cyl_iv, "translate=(6 0 0)"),
    ):
        state = "valveClosed" if lift == 0 else "valveOpen"
        set_dict(solver, f"system/blockMeshDict.{state}",
                 f"valveLift={-lift}, nValve={n_valve}, x0=-5, nXLeft=4, pistonPos={-pos}, nPiston={n_clearance}",
                 f"{side}.{mesh}")
        tmp = "_tmp_exhaust" if side == "exhaust" else "_tmp_intake"
        run(solver, "blockMesh", "-mesh", tmp, "-dict", f"system/blockMeshDict.{state}")
        run(solver, "mirrorMesh", "-mesh", tmp, "-dict", "system/mirrorMeshDict")
        run(solver, "transformPoints", "-mesh", tmp, translate)
        boundary_edits(solver, tmp, side)
    merge_args = mesh_arg(mesh) + ["-region", "fluid", "-addMeshes", "(_tmp_exhaust _tmp_intake)"]
    run(solver, "mergeMeshes", *merge_args)
    run(solver, "createBaffles", *mesh_arg(mesh), "-region", "fluid", "-dict", "system/createBafflesDict")
    run(solver, "splitBaffles", *mesh_arg(mesh), "-region", "fluid")
    run(solver, "transformPoints", *mesh_arg(mesh), "-region", "fluid", "Rx=90, scale=(0.1 0.1 0.1)")
    run(solver, "createNonConformalCouples", *mesh_arg(mesh), "-dict", "system/createNonConformalCouplesDict")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)
    create_solids(solver)
    create_fluid_mesh(solver, None, 0, 0)
    for mesh in MESH_TIMES:
        iv, ev = LIFTS[mesh]
        create_fluid_mesh(solver, mesh, iv, ev)
    solver.write_text_asset(
        "constant/fluid/meshTimes",
        "\n".join(str(t) for t in MESH_TIMES) + "\n",
    )
    solver.run_command(
        [str(OF13_BIN / "decomposePar"), "-allRegions"],
        log_filename="log.decomposePar.allRegions",
    )
    for mesh in MESH_TIMES:
        solver.run_command(
            [str(OF13_BIN / "decomposePar"), "-mesh", str(mesh), "-region", "fluid"],
            log_filename=f"log.decomposePar.mesh.{mesh}",
        )
    solver.run_command(
        ["/usr/bin/mpirun", "--oversubscribe", "-np", str(NPROCS),
         str(OF13_BIN / "foamMultiRun"), "-parallel"],
        log_filename="log.foamMultiRun.parallel",
    )
    solver.run_command(
        [str(OF13_BIN / "reconstructPar"), "-allRegions"],
        log_filename="log.reconstructPar.allRegions",
    )


if __name__ == "__main__":
    main()
