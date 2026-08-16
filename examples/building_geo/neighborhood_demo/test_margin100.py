from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voxcity_export_work" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gmsh
import shapely.ops
import pyproj
from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from voxcity.io import load_voxcity
from vector_builder import VectorGmshBuilder

v = load_voxcity('output/voxcity.h5')
gdf = v.extras.get('building_gdf')
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
gdf_proj = gdf.copy()
gdf_proj.geometry = gdf_proj.geometry.apply(lambda geom: shapely.ops.transform(project, geom) if geom is not None else None)

urban = UrbanModel()
counter = 0
for idx, row in gdf.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    footprints = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    projected_footprints = None
    try:
        projected_row = gdf_proj.loc[idx]
        projected_geom = projected_row.geometry
        if projected_geom is not None and not projected_geom.is_empty:
            if projected_geom.geom_type == 'Polygon':
                projected_footprints = [projected_geom]
            elif projected_geom.geom_type == 'MultiPolygon':
                projected_footprints = list(projected_geom.geoms)
    except Exception:
        projected_footprints = None
    for footprint_idx, footprint in enumerate(footprints):
        if projected_footprints is not None and footprint_idx < len(projected_footprints):
            use_footprint = projected_footprints[footprint_idx]
        else:
            use_footprint = footprint
        if not use_footprint.is_valid:
            use_footprint = use_footprint.buffer(0)
        if use_footprint.is_empty:
            continue
        urban.add_building(Building(
            id=f"v{idx}_{counter}",
            footprint=use_footprint,
            ground_z=0.0,
            roof_z=float(getattr(row, 'height', 9.0) or 9.0),
            source="voxcity",
            confidence=0.7,
        ))
        counter += 1

builder = VectorGmshBuilder(urban, CFDTerrain.flat(z=0.0), mesh_size=10.0)
builder.build(margin=100.0, bottom_offset=5.0)
builder.assign_patches()
builder.build_mesh(mesh_size=10.0)

print(f"Nodes: {len(gmsh.model.mesh.getNodes()[0])}")

all_elements = gmsh.model.mesh.getElements(3)
print(f"getElements(3): {len(all_elements)} parts")
for i, part in enumerate(all_elements):
    print(f"  part {i}: len={len(part) if hasattr(part, '__len__') else 'N/A'}")

phys3 = gmsh.model.getPhysicalGroups(dim=3)
print(f"dim=3 physical groups: {phys3}")
for dim, ptag in phys3:
    pname = gmsh.model.getPhysicalName(dim, ptag)
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
    print(f"  '{pname}' (tag={ptag}): entities={entities}")
    for ent in entities:
        etypes, _, enodes = gmsh.model.mesh.getElements(3, ent)
        print(f"    entity={ent}: etypes={etypes}, n_elements={sum(len(e) for e in enodes)}")

try:
    builder.export_openfoam(Path("case_margin100"))
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")

builder.finalize()
