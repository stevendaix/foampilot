from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


def load_snappy():
    source = Path(__file__).parents[2] / "foampilot" / "src" / "foampilot" / "mesh" / "snappymesh.py"
    spec = importlib.util.spec_from_file_location("foampilot_snappymesh_standalone", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SnappyMesher


def foam_header(obj, loc):
    return f'''FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "{loc}";
    object {obj};
}}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patch_dir", type=Path)
    parser.add_argument("case", type=Path)
    parser.add_argument("--location", nargs=3, type=float, default=(223, 139, 24))
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--first-layer", type=float, default=0.12)
    args = parser.parse_args()

    case = args.case.resolve()
    tri = case / "constant" / "triSurface"
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "0").mkdir(parents=True, exist_ok=True)
    tri.mkdir(parents=True, exist_ok=True)

    names = ["inlet", "outlet_0", "outlet_1", "wall"]
    for name in names:
        source = args.patch_dir / f"{name}.stl"
        shutil.copy2(source, tri / source.name)

    SnappyMesher = load_snappy()
    mesher = SnappyMesher(case_path=case, stl_file=None, castellatedMesh=True, snap=True, addLayers=True)
    for name in names:
        mesher.add_geometry(name, tri / f"{name}.stl")
    mesher.locationInMesh = tuple(args.location)
    mesher.castellatedMeshControls["locationInMesh"] = tuple(args.location)
    mesher.castellatedMeshControls["refinementSurfaces"] = {name: {"level": (2, 3)} for name in names}
    mesher.addLayersControls.update({
        "relativeSizes": False,
        "expansionRatio": 1.2,
        "firstLayerThickness": args.first_layer,
        "finalLayerThickness": args.first_layer * (1.2 ** (args.layers - 1)),
        "minThickness": args.first_layer * 0.35,
        "nLayerIter": 50,
    })
    mesher.add_layer("wall", args.layers)
    mesher.write_block_mesh_dict(padding=0.20, base_cell_size=0.75)
    mesher.write_surface_features_dict(names, included_angle=30)
    mesher.write()

    (case / "system" / "controlDict").write_text(foam_header("controlDict", "system") + "application foamRun;\nsolver incompressibleFluid;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
    (case / "constant" / "transportProperties").write_text(foam_header("transportProperties", "constant") + "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 3.77e-06;\n")
    (case / "system" / "fvSchemes").write_text(foam_header("fvSchemes", "system") + "ddtSchemes { default steadyState; }\ngraduSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n")
    (case / "system" / "fvSolution").write_text(foam_header("fvSolution", "system") + "solvers { p { solver GAMG; tolerance 1e-7; relTol 0; } U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; } }\nSIMPLE { nNonOrthogonalCorrectors 1; }\n")
    (case / "0" / "U").write_text(foam_header("U", "0") + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n    inlet { type fixedValue; value uniform (0 0 0); }\n    outlet_0 { type zeroGradient; }\n    outlet_1 { type zeroGradient; }\n    wall { type noSlip; }\n}\n")
    (case / "0" / "p").write_text(foam_header("p", "0") + "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    inlet { type zeroGradient; }\n    outlet_0 { type fixedValue; value uniform 0; }\n    outlet_1 { type fixedValue; value uniform 0; }\n    wall { type zeroGradient; }\n}\n")
    print(f"Generated patched foampilot case: {case}")


if __name__ == "__main__":
    main()
