from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voxcity_export_work" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapely.ops
import pyproj
from voxcity.io import load_voxcity
from shapely.validation import make_valid
from shapely.ops import unary_union

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
            geom = unary_union(polygons)
            geom = make_valid(geom)
            geom = geom.buffer(0.0)
        if geom.is_empty or geom.area < min_area_m2:
            return None
        return geom
    except Exception:
        return None

raw_polys = []
clean_polys = []
labels = []
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
            use_fp = projected_footprints[footprint_idx]
        else:
            use_fp = footprint
        cleaned = clean_footprint(use_fp, min_area_m2=1.0, simplify_tol=0.5, rounding_precision=1)
        if cleaned is None:
            continue
        raw_polys.append(use_fp)
        clean_polys.append(cleaned)
        labels.append(f"v{idx}_{footprint_idx}")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

def plot_polys(ax, polys, title, color):
    for poly in polys:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.35, facecolor=color, edgecolor="black", linewidth=0.8)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, alpha=0.35, facecolor="white", edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)

plot_polys(axes[0], raw_polys, f"Raw VoxCity footprints: {len(raw_polys)}", "#d9d9d9")
plot_polys(axes[1], clean_polys, f"Cleaned footprints: {len(clean_polys)}", "#4c78a8")

fig.suptitle("VoxCity footprint cleaning — EPSG:32631", fontsize=14)
fig.tight_layout()
out = Path("footprint_cleaning.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out.resolve()}")
