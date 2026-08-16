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

v = load_voxcity('output/voxcity.h5')
gdf = v.extras.get('building_gdf')
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
gdf_proj = gdf.copy()
gdf_proj.geometry = gdf_proj.geometry.apply(lambda geom: shapely.ops.transform(project, geom) if geom is not None else None)

def clean_footprint(geom, min_area_m2=1.0, simplify_tol=0.5, rounding_precision=1):
    if geom is None or geom.is_empty:
        return None
    try:
        from shapely.validation import make_valid
        import shapely
        geom = make_valid(geom)
        geom = geom.buffer(0.0)
        if geom.is_empty:
            return None
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

print("=" * 80)
print("VOXCITY FOOTPRINT CLEANING REPORT")
print("=" * 80)

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
        area_before = use_footprint.area
        cleaned = clean_footprint(use_footprint, min_area_m2=1.0, simplify_tol=0.5, rounding_precision=1)
        if cleaned is None:
            print(f"v{idx}_{footprint_idx}: FILTERED OUT")
            continue
        area_after = cleaned.area
        ncoords_before = len(list(use_footprint.exterior.coords)) if hasattr(use_footprint, 'exterior') else 0
        ncoords_after = len(list(cleaned.exterior.coords)) if hasattr(cleaned, 'exterior') else 0
        print(f"v{idx}_{footprint_idx}: valid={cleaned.is_valid}, "
              f"area={area_before:.2f}->{area_after:.2f} m², "
              f"coords={ncoords_before}->{ncoords_after}, "
              f"type={cleaned.geom_type}")
        urban.add_building(Building(
            id=f"v{idx}_{counter}",
            footprint=cleaned,
            ground_z=0.0,
            roof_z=float(getattr(row, 'height', 9.0) or 9.0),
            source="voxcity",
            confidence=0.7,
        ))
        counter += 1

print(f"\nTotal buildings kept: {urban.building_count()}")

builder = VectorGmshBuilder(urban, CFDTerrain.flat(z=0.0), mesh_size=10.0)
builder.build(margin=100.0, bottom_offset=5.0)

print(f"\n=== AFTER BUILD ===")
print(f"Buildings: {len(builder.buildings)}")
print(f"Building tags: {builder.building_tags}")
print(f"Fluid tag: {builder.fluid_tag}")

vols = gmsh.model.getEntities(dim=3)
print(f"dim=3 entities: {len(vols)}")
for dim, tag in vols:
    bbox = gmsh.model.occ.getBoundingBox(3, tag)
    mass = abs(gmsh.model.occ.getMass(3, tag))
    print(f"  tag={tag}, bbox={bbox}, mass={mass:.2f}")

builder.assign_patches()

# 2D mesh inspect
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.5)
gmsh.model.mesh.generate(2)

surfs = gmsh.model.getEntities(dim=2)
print(f"\ndim=2 entities: {len(surfs)}")
for dim, tag in surfs:
    etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
    n = sum(len(e) for e in enodes)
    if n > 0:
        print(f"  tag={tag}: etypes={etypes}, n_elements={n}")

# 3D attempt
gmsh.model.mesh.clear()
gmsh.model.mesh.generate(3)

print(f"\nAfter generate(3): nodes={len(gmsh.model.mesh.getNodes()[0])}")
vols_after = gmsh.model.getEntities(dim=3)
print(f"dim=3 entities: {len(vols_after)}")
for dim, tag in vols_after:
    bbox = gmsh.model.occ.getBoundingBox(3, tag)
    print(f"  tag={tag}, bbox={bbox}")

phys3 = gmsh.model.getPhysicalGroups(dim=3)
print(f"dim=3 physical groups: {phys3}")
for dim, ptag in phys3:
    pname = gmsh.model.getPhysicalName(dim, ptag)
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
    print(f"  '{pname}' (tag={ptag}): entities={entities}")
    for ent in entities:
        etypes, _, enodes = gmsh.model.mesh.getElements(3, ent)
        print(f"    entity={ent}: etypes={etypes}, n_elements={sum(len(e) for e in enodes)}")

all_elements = gmsh.model.mesh.getElements(3)
print(f"getElements(3) without tag: {len(all_elements)} parts")
for i, part in enumerate(all_elements):
    print(f"  part {i}: len={len(part) if hasattr(part, '__len__') else 'N/A'}")

builder.finalize()
