#!/usr/bin/env python3
"""
Geometry mapping example for urban CFD.

Generates footprint, domain, and 3D plots for a sample neighborhood
to visually verify the geometry pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.urban import (
    Building,
    UrbanModel,
    CFDDomain,
    WindFrame,
    CFDLOD,
    CFDSimplifier,
    GmshQuarterBuilder,
    MeshConfig,
    GeometryMapper,
)
from shapely.geometry import Polygon


def main():
    urban = UrbanModel()
    urban.add_building(Building(
        id="B1",
        footprint=Polygon([(0, 0), (40, 0), (40, 20), (0, 20)]),
        ground_z=0.0,
        roof_z=15.0,
    ))
    urban.add_building(Building(
        id="B2",
        footprint=Polygon([(50, 10), (80, 10), (80, 30), (50, 30)]),
        ground_z=0.0,
        roof_z=22.0,
    ))
    urban.add_building(Building(
        id="B3",
        footprint=Polygon([(20, 40), (60, 40), (60, 70), (20, 70)]),
        ground_z=0.0,
        roof_z=18.0,
    ))

    wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
    domain = CFDDomain(
        upstream=8.0,
        downstream=15.0,
        lateral=4.0,
        top=2.5,
        extent_units="href",
        reference_height_method="Hmax",
    )

    geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
        wind_frame=wind_frame,
        domain=domain,
    )

    mapper = GeometryMapper(urban, geometry)

    output_dir = Path(__file__).parent / "geometry_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving geometry maps...")
    saved = mapper.save_plots(output_dir, prefix="urban")

    for path in saved:
        print(f"  {path}")

    summary = mapper.summary()
    print("\nGeometry summary:")
    print(f"  Buildings: {summary['n_buildings']}")
    print(f"  CFD buildings: {summary.get('cfd_buildings', 'N/A')}")
    print(f"  Height range: {summary['height_stats']['min']:.1f} - {summary['height_stats']['max']:.1f} m")
    print(f"  Total area: {summary['area_stats']['total']:.1f} m²")

    if "domain_box" in summary:
        db = summary["domain_box"]
        print(f"  Domain box: ({db['xmin']:.1f}, {db['ymin']:.1f}, {db['zmin']:.1f}) -> ({db['xmax']:.1f}, {db['ymax']:.1f}, {db['zmax']:.1f})")

    print(f"\nOpen the images in {output_dir} to verify the geometry.")


if __name__ == "__main__":
    main()
