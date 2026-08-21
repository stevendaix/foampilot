from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


def load_snappy_mesher():
    source = Path(__file__).parents[2] / "foampilot" / "src" / "foampilot" / "mesh" / "snappymesh.py"
    spec = importlib.util.spec_from_file_location("foampilot_snappymesh_standalone", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SnappyMesher from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SnappyMesher


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a medical STL into a foampilot snappyHexMesh case")
    parser.add_argument("stl", type=Path)
    parser.add_argument("case", type=Path)
    parser.add_argument("--location", nargs=3, type=float, default=(223.0, 139.0, 24.0))
    parser.add_argument("--padding", type=float, default=0.20)
    parser.add_argument("--cell-size", type=float, default=0.75)
    args = parser.parse_args()

    case = args.case.resolve()
    tri_surface = case / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "0").mkdir(parents=True, exist_ok=True)

    target = tri_surface / "aorta_wall.stl"
    shutil.copy2(args.stl, target)

    SnappyMesher = load_snappy_mesher()
    mesher = SnappyMesher(case_path=case, stl_file=target, castellatedMesh=True, snap=True, addLayers=False)
    mesher.locationInMesh = tuple(args.location)
    mesher.castellatedMeshControls["locationInMesh"] = tuple(args.location)
    mesher.castellatedMeshControls["refinementSurfaces"] = {"aorta_wall": {"level": (2, 3)}}
    mesher.write_block_mesh_dict(padding=args.padding, base_cell_size=args.cell_size)
    mesher.write_surface_features_dict(["aorta_wall"], included_angle=30)
    mesher.write()

    print(f"Generated foampilot snappy case: {case}")
    print(f"STL: {target}")
    print(f"locationInMesh: {args.location}")


if __name__ == "__main__":
    main()
