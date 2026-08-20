#!/usr/bin/env python3
"""
Read VoxCity model and extract data for CFD geometry generation.

Usage:
    PYTHONPATH=../../../src python3 src/read_voxcity.py \
        --output output/voxcity_test \
        --meshsize 5.0 \
        --use-voxcity \
        --rectangle-vertices 2.3225 48.8515 2.3225 48.8528 2.3240 48.8528 2.3240 48.8515
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from foampilot.urban.readers.voxcity_reader import VoxCityReader


def parse_args():
    parser = argparse.ArgumentParser(description="Read VoxCity and extract CFD data")
    parser.add_argument("--output", default="output/voxcity_test", help="Output directory")
    parser.add_argument("--meshsize", type=float, default=5.0, help="VoxCity mesh size")
    parser.add_argument("--use-voxcity", action="store_true", help="Use real VoxCity/EE data")
    parser.add_argument(
        "--rectangle-vertices",
        nargs="+",
        type=float,
        default=None,
        help="Lon/lat vertices for VoxCity",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_voxcity and args.rectangle_vertices:
        coords = args.rectangle_vertices
        rectangle_vertices = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
        reader = VoxCityReader(meshsize=args.meshsize)
        urban, terrain = reader.read(rectangle_vertices)
        print(f"VoxCity: {urban.building_count()} buildings, terrain={terrain.source}")
    else:
        from foampilot.urban import Building, UrbanModel
        from foampilot.urban.model.terrain import CFDTerrain
        from shapely.geometry import Polygon

        urban = UrbanModel()
        urban.add_building(Building(
            id="B1",
            footprint=Polygon([(0, 0), (20, 0), (20, 12), (0, 12)]),
            ground_z=0.0,
            roof_z=15.0,
            source="synthetic",
        ))
        terrain = CFDTerrain.flat(z=0.0)
        print("Using synthetic data")

    # Save urban model
    urban.to_geojson(output_dir / "urban_model.geojson")

    # Save terrain bounds
    bounds = terrain.get_bounds()
    terrain_info = {
        "source": terrain.source,
        "bounds": bounds,
        "points": [{"x": p.x, "y": p.y, "z": p.z} for p in terrain.points[:10]],
    }
    with open(output_dir / "terrain_info.json", "w") as f:
        json.dump(terrain_info, f, indent=2)

    print(f"Output written to {output_dir}")
    print(f"  urban_model.geojson")
    print(f"  terrain_info.json")


if __name__ == "__main__":
    main()
