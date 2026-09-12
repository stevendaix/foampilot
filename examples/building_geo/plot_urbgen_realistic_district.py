from pathlib import Path
from dataclasses import replace
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon
from foampilot.urban.generation import UrbGENConfig, generate_urbgen

OUT = Path(__file__).with_name('urbgen_validation_images')
OUT.mkdir(exist_ok=True)

# District synthétique mais structuré comme les exemples UrbGEN : plusieurs îlots,
# une place centrale et des bandes bâties périphériques. Les rues ne sont pas
# générées comme bâtiments ; elles restent les vides entre parcelles.
parcels = {
    'west_low': Polygon([(8, 35), (38, 35), (38, 58), (8, 58)]),
    'west_mid': Polygon([(8, 64), (38, 64), (38, 88), (8, 88)]),
    'north_west': Polygon([(43, 83), (70, 83), (70, 108), (43, 108)]),
    'north_center': Polygon([(75, 83), (104, 83), (104, 108), (75, 108)]),
    'north_east': Polygon([(109, 83), (142, 83), (142, 108), (109, 108)]),
    'east_mid': Polygon([(114, 52), (142, 52), (142, 77), (114, 77)]),
    'east_low': Polygon([(112, 16), (142, 16), (142, 43), (112, 43)]),
    'south_center': Polygon([(70, 8), (104, 8), (104, 35), (70, 35)]),
    'south_west': Polygon([(42, 10), (65, 10), (65, 34), (42, 34)]),
    'central_west': Polygon([(42, 39), (66, 39), (66, 77), (42, 77)]),
    'central_core': Polygon([(70, 39), (105, 39), (105, 78), (70, 78)]),
}

configs = {
    'west_low': UrbGENConfig(bcr=.38, far=1.8, setback=2.0, min_width=7, min_footprint_per_tower=35, max_footprint_per_tower=130, min_tower_distance=3, tower_typology_mode=1, global_rotation_mode=0, podium_floors=1, height_variation=.15, seed=101),
    'west_mid': UrbGENConfig(bcr=.42, far=2.2, setback=1.5, min_width=6, min_footprint_per_tower=30, max_footprint_per_tower=140, min_tower_distance=2.5, tower_typology_mode=1, global_rotation_mode=0, podium_floors=1, height_variation=.20, seed=102),
    'north_west': UrbGENConfig(bcr=.48, far=3.2, setback=2.0, min_width=7, min_footprint_per_tower=35, max_footprint_per_tower=150, min_tower_distance=3, tower_typology_mode=6, global_rotation_mode=2, podium_floors=2, height_variation=.30, seed=103),
    'north_center': UrbGENConfig(bcr=.50, far=3.8, setback=2.0, min_width=7, min_footprint_per_tower=35, max_footprint_per_tower=160, min_tower_distance=3, tower_typology_mode=6, global_rotation_mode=2, podium_floors=2, height_variation=.35, seed=104),
    'north_east': UrbGENConfig(bcr=.44, far=2.8, setback=2.0, min_width=7, min_footprint_per_tower=35, max_footprint_per_tower=150, min_tower_distance=3, tower_typology_mode=7, courtyard_break_count=5, courtyard_break_width=5, courtyard_layout_mode=0, podium_floors=0, seed=105),
    'east_mid': UrbGENConfig(bcr=.44, far=2.6, setback=1.5, min_width=6, min_footprint_per_tower=30, max_footprint_per_tower=140, min_tower_distance=2.5, tower_typology_mode=0, global_rotation_mode=2, podium_floors=1, height_variation=.22, seed=106),
    'east_low': UrbGENConfig(bcr=.38, far=1.6, setback=2.0, min_width=7, min_footprint_per_tower=35, max_footprint_per_tower=120, min_tower_distance=3, tower_typology_mode=4, global_rotation_mode=0, podium_floors=1, height_variation=.12, seed=107),
    'south_center': UrbGENConfig(bcr=.46, far=3.0, setback=1.5, min_width=6, min_footprint_per_tower=30, max_footprint_per_tower=150, min_tower_distance=2.5, tower_typology_mode=5, global_rotation_mode=2, podium_floors=1, height_variation=.25, seed=108),
    'south_west': UrbGENConfig(bcr=.40, far=2.0, setback=2.0, min_width=7, min_footprint_per_tower=35, max_footprint_per_tower=130, min_tower_distance=3, tower_typology_mode=1, global_rotation_mode=0, podium_floors=1, height_variation=.15, seed=109),
    'central_west': UrbGENConfig(bcr=.48, far=3.4, setback=2.0, min_width=6, min_footprint_per_tower=30, max_footprint_per_tower=150, min_tower_distance=2.5, tower_typology_mode=6, global_rotation_mode=2, podium_floors=2, height_variation=.25, seed=110),
    'central_core': UrbGENConfig(bcr=.52, far=4.2, setback=2.0, min_width=6, min_footprint_per_tower=30, max_footprint_per_tower=160, min_tower_distance=2.5, tower_typology_mode=6, global_rotation_mode=2, podium_floors=2, height_variation=.35, seed=111),
}
# Échelle morphologique plus proche des références UrbGEN : plusieurs volumes par îlot,
# hauteurs limitées et distance réduite mais non nulle entre bâtiments.
configs = {k: replace(v, min_width=5.0, min_footprint_per_tower=24.0, min_tower_distance=2.0, max_building_height=70.0, enforce_height_regulation=True, height_regulation_mode=1) for k, v in configs.items()}

results = {}
for name, parcel in parcels.items():
    results[name] = generate_urbgen(parcel, configs[name])

# 3D view with height colour, closer to the original UrbGEN references.
fig = plt.figure(figsize=(15, 10), dpi=150)
ax = fig.add_subplot(111, projection='3d')
all_heights = [b.height for r in results.values() for b in r.model.buildings()]
hmin, hmax = min(all_heights), max(all_heights)
cmap = plt.cm.turbo

def color(h):
    return cmap((h-hmin) / max(1e-9, hmax-hmin))

def prism(poly, z0, z1):
    xy = list(poly.exterior.coords)[:-1]
    verts = [[(x,y,z0) for x,y in xy], [(x,y,z1) for x,y in xy]]
    for i in range(len(xy)):
        j = (i+1) % len(xy)
        verts.append([(xy[i][0],xy[i][1],z0),(xy[j][0],xy[j][1],z0),(xy[j][0],xy[j][1],z1),(xy[i][0],xy[i][1],z1)])
    return verts

# parcel outlines / streets
for name, parcel in parcels.items():
    xy = list(parcel.exterior.coords)
    ax.plot([p[0] for p in xy], [p[1] for p in xy], [0]*len(xy), color='#AAAAAA', linewidth=1)
for result in results.values():
    for b in result.model.buildings():
        if b.height <= 0: continue
        geom = b.footprint
        if geom.geom_type == 'MultiPolygon':
            polys = list(geom.geoms)
        else: polys = [geom]
        for poly in polys:
            ax.add_collection3d(Poly3DCollection(prism(poly, b.ground_z, b.ground_z+b.height), facecolors=color(b.height), edgecolors='#333333', linewidths=.25, alpha=.92))

ax.set_xlim(0,150); ax.set_ylim(0,115); ax.set_zlim(0, hmax*1.08)
ax.set_box_aspect((150,115,hmax*0.85)); ax.view_init(elev=42, azim=-58)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('height [m]')
ax.set_title(f'UrbGEN / district multi-îlots — {sum(r.model.building_count() for r in results.values())} bâtiments')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=hmin, vmax=hmax)); sm.set_array([])
fig.colorbar(sm, ax=ax, shrink=.55, pad=.08, label='Hauteur [m]')
fig.tight_layout()
out = OUT / 'urbgen_realistic_district_3d.png'; fig.savefig(out, bbox_inches='tight'); plt.close(fig)

metrics = {'parcel_count': len(results), 'building_count': sum(r.model.building_count() for r in results.values()), 'height_min': hmin, 'height_max': hmax, 'parcels': {k: {'building_count': v.model.building_count(), 'actual_bcr': v.actual_bcr, 'actual_far': v.actual_far, 'typologies': v.tower_typologies} for k,v in results.items()}}
(OUT / 'urbgen_realistic_district_3d.json').write_text(json.dumps(metrics, indent=2, default=float))
print(out)
print(json.dumps(metrics, indent=2, default=float))
