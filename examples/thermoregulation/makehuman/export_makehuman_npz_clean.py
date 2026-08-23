import os
from pathlib import Path
import argparse
import json
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=Path(os.getenv('MAKEHUMAN_BASE_NPZ', '/usr/share/makehuman-community/data/3dobjs/base.npz')))
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--scale', type=float, default=0.1)
parser.add_argument('--fill-holes', action='store_true')
parser.add_argument('--merge-digits', type=int, default=8)
parser.add_argument('--group-id', type=int, default=0, help='Groupe MakeHuman à exporter; 0 correspond à body')
parser.add_argument('--all-groups', action='store_true', help='Exporter tous les groupes au lieu du seul groupe body')
args = parser.parse_args()

data = np.load(args.input, allow_pickle=True)
vertices = args.scale * np.asarray(data['coord'], dtype=float)
all_raw = np.asarray(data['fvert'], dtype=np.int64)
all_groups = np.asarray(data['group'], dtype=np.int64)
selected = np.ones(len(all_raw), dtype=bool) if args.all_groups else (all_groups == args.group_id)
raw = all_raw[selected]
if raw.ndim != 2 or raw.shape[1] < 3:
    raise ValueError(f'Unexpected fvert shape: {raw.shape}')
triangles = []
for face in raw:
    face = face[:4]
    face = face[face >= 0]
    if len(face) == 3:
        candidates = [face]
    elif len(face) == 4:
        candidates = [face[[0, 1, 2]], face[[0, 2, 3]]]
    else:
        continue
    for tri in candidates:
        if len(set(map(int, tri))) == 3 and np.all(tri < len(vertices)):
            triangles.append(tri)
faces = np.asarray(triangles, dtype=np.int64)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.merge_vertices(digits_vertex=args.merge_digits)
if hasattr(mesh, 'nondegenerate_faces'):
    mesh.update_faces(mesh.nondegenerate_faces())
mesh.remove_unreferenced_vertices()
trimesh.repair.fix_winding(mesh)
if args.fill_holes:
    trimesh.repair.fill_holes(mesh)
    mesh.remove_unreferenced_vertices()
args.output.parent.mkdir(parents=True, exist_ok=True)
mesh.export(args.output)
components = mesh.split(only_watertight=False)
report = {
    'input': str(args.input),
    'output': str(args.output),
    'scale': args.scale,
    'raw_faces': int(len(raw)),
    'raw_faces_all_groups': int(len(all_raw)),
    'selected_group_id': None if args.all_groups else int(args.group_id),
    'triangles_exported': int(len(mesh.faces)),
    'vertices_exported': int(len(mesh.vertices)),
    'area_m2': float(mesh.area),
    'volume_m3': float(mesh.volume),
    'watertight': bool(mesh.is_watertight),
    'winding_consistent': bool(mesh.is_winding_consistent),
    'components': int(len(components)),
    'component_face_counts_top20': sorted((len(c.faces) for c in components), reverse=True)[:20],
    'bounds_min': mesh.bounds[0].tolist(),
    'bounds_max': mesh.bounds[1].tolist(),
}
report_path = args.output.with_suffix('.quality.json')
report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
