"""OpenFOAM 13 windAroundBuildings, généré avec FoamPilot."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver, OpenFOAMEnvironment
from foampilot import Meshing
from foampilot.mesh import SnappyMesher


def main() -> None:
    os.environ.update(OpenFOAMEnvironment().environment())
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = False
    solver.turbulence_model = "kEpsilon"
    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.endTime = 500.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100
    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "0"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.write()

    mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = mesh.mesher
    blockmesh.vertices = [
        [-20, -50, 0], [330, -50, 0], [330, 230, 0], [-20, 230, 0],
        [-20, -50, 140], [330, -50, 140], [330, 230, 140], [-20, 230, 140],
    ]
    blockmesh.blocks = ["hex (0 1 2 3 4 5 6 7) (25 20 10) simpleGrading (1 1 1)"]
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "inlet": {"type": "patch", "faces": [(0, 3, 7, 4)]},
        "outlet": {"type": "patch", "faces": [(1, 5, 6, 2)]},
        "ground": {"type": "wall", "faces": [(0, 1, 2, 3)]},
        "frontAndBack": {"type": "symmetry", "faces": [(0, 4, 5, 1), (3, 2, 6, 7), (4, 7, 6, 5)]},
    }
    blockmesh.write(case_path / "system" / "blockMeshDict")

    obj_source = Path(os.environ["FOAM_TUTORIALS"]) / "incompressibleFluid" / "windAroundBuildings" / "constant" / "geometry" / "buildings.obj.gz"
    snappy = SnappyMesher(parent=solver._solver, castellatedMesh=True, snap=True, addLayers=False)
    surface = snappy.import_reference_surface(obj_source, target_name="buildings.obj")
    snappy.locationInMesh = (1, 1, 1)
    snappy.castellatedMeshControls["refinementSurfaces"] = {"buildings": {"level": (3, 3)}}
    snappy.add_searchable_box("refinementBox", (0, 0, 0), (250, 180, 90))
    snappy.add_refinement_region("refinementBox", "inside", 2)
    snappy.write_surface_features_dict([surface.name], included_angle=30)
    snappy.add_feature("buildings.eMesh", 1)
    snappy.write_snappyHexMeshDict()
    snappy.run()

    solver.constant.write()
    solver.setup_case()
    solver.boundary.initialize_boundary()
    solver.boundary.set_raw_condition("inlet", "U", {"type": "fixedValue", "value": "uniform (10 0 0)"})
    solver.boundary.set_raw_condition("outlet", "U", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("ground", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})
    solver.boundary.set_raw_condition("ground", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("inlet", "k", {"type": "fixedValue", "value": "uniform 0.375"})
    solver.boundary.set_raw_condition("inlet", "epsilon", {"type": "fixedValue", "value": "uniform 0.1"})
    solver.boundary.set_raw_condition("inlet", "nut", {"type": "calculated", "value": "uniform 0"})
    for patch in ("outlet", "ground", "buildings"):
        solver.boundary.set_raw_condition(patch, "U", {"type": "zeroGradient" if patch == "outlet" else "noSlip"})
        solver.boundary.set_raw_condition(patch, "p", {"type": "fixedValue", "value": "uniform 0"} if patch == "outlet" else {"type": "zeroGradient"})
        solver.boundary.set_raw_condition(patch, "k", {"type": "zeroGradient" if patch == "outlet" else "kqRWallFunction", "value": "uniform 0.375"} if patch != "outlet" else {"type": "zeroGradient"})
        solver.boundary.set_raw_condition(patch, "epsilon", {"type": "zeroGradient" if patch == "outlet" else "epsilonWallFunction", "value": "uniform 0.1"} if patch != "outlet" else {"type": "zeroGradient"})
        solver.boundary.set_raw_condition(patch, "nut", {"type": "calculated", "value": "uniform 0"} if patch == "outlet" else {"type": "nutkWallFunction", "value": "uniform 0"})
    for field in ("U", "p", "k", "epsilon", "nut"):
        solver.boundary.set_raw_condition("frontAndBack", field, {"type": "symmetryPlane"})
    solver.boundary.write_boundary_conditions()
    solver.run_simulation(nb_proc=1)


if __name__ == "__main__":
    main()
