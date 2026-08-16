from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voxcity_export_work" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gmsh
import shapely.ops
import pyproj
from voxcity.io import load_voxcity
from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from vector_builder import VectorGmshBuilder
from shapely.validation import make_valid

v = load_voxcity('output/voxcity.h5')
gdf = v.extras.get('building_gdf')
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
gdf_proj = gdf.copy()
gdf_proj.geometry = gdf_proj.geometry.apply(lambda geom: shapely.ops.transform(project, geom) if geom is not None else None)

def clean_footprint(geom, min_area_m2=1.0, simplify_tol=0.5, rounding_precision=1):
    if geom is None or geom.is_empty:
        return None
    try:
        geom = make_valid(geom)
        geom = geom.buffer(0.0)
        if geom.is_empty:
            return None
        import shapely
        geom = shapely.wkt.loads(shapely.wkt.dumps(geom, rounding_precision=rounding_precision))
        geom = make_valid(geom)
        geom = geom.buffer(0.0)
        if geom.is_empty:
            return None
        if simplify_tol > 0.0:
            geom = geom.simplify(simplify_tol, preserve_topology=True)
            geom = make_valid(geom)
            geom = geom.buffer(0.0)
        if geom.is_empty:
            return None
        if geom.area < min_area_m2:
            return None
        if geom.geom_type == "MultiPolygon":
            polygons = [p for p in geom.geoms if not p.is_empty and p.area >= min_area_m2]
            if not polygons:
                return None
            geom = shapely.ops.unary_union(polygons)
            geom = make_valid(geom)
            geom = geom.buffer(0.0)
        if geom.is_empty or geom.area < min_area_m2:
            return None
        return geom
    except Exception:
        return None

def merge_nearby_buildings(polys, distance=0.5):
    if not polys:
        return []
    merged = shapely.ops.unary_union(polys)
    merged = make_valid(merged)
    merged = merged.buffer(0.0)
    if merged.is_empty:
        return []
    if merged.geom_type == "MultiPolygon":
        result = [p for p in merged.geoms if not p.is_empty]
    else:
        result = [merged] if not merged.is_empty else []
    return result

cleaned_footprints = []
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
        cleaned = clean_footprint(use_footprint, min_area_m2=1.0, simplify_tol=0.5, rounding_precision=1)
        if cleaned is None:
            continue
        cleaned_footprints.append(cleaned)

merged_footprints = merge_nearby_buildings(cleaned_footprints, distance=0.5)
print(f"Cleaned: {len(cleaned_footprints)} -> Merged: {len(merged_footprints)}")

urban = UrbanModel()
for i, fp in enumerate(merged_footprints):
    urban.add_building(Building(
        id=f"merged_{i}",
        footprint=fp,
        ground_z=0.0,
        roof_z=27.0,
        source="voxcity_merged",
        confidence=0.7,
    ))

builder = VectorGmshBuilder(urban, CFDTerrain.flat(z=0.0), mesh_size=10.0)
builder.build(margin=100.0, bottom_offset=5.0)

print(f"Buildings: {len(builder.buildings)}")
print(f"Fluid tag: {builder.fluid_tag}")

vols = gmsh.model.getEntities(dim=3)
print(f"Volumes before mesh: {len(vols)}")
for dim, tag in vols:
    bbox = gmsh.model.occ.getBoundingBox(3, tag)
    mass = abs(gmsh.model.occ.getMass(3, tag))
    print(f"  tag={tag}, bbox={bbox}, mass={mass:.2f}")

builder.assign_patches()

# Write geometry for inspection
gmsh.write("debug_merged.msh")

gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.5)

# Generate 2D mesh
gmsh.model.mesh.generate(2)
gmsh.write("debug_merged_2d.msh")
print(f"2D mesh nodes: {len(gmsh.model.mesh.getNodes()[0])}")

# Generate 3D mesh
gmsh.model.mesh.clear()
gmsh.model.mesh.generate(3)
print(f"3D mesh nodes: {len(gmsh.model.mesh.getNodes()[0])}")

# Check elements
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

builder.finalize()
