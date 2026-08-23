import os
from pathlib import Path
import argparse
import json
import numpy as np
import meshio

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=Path(os.getenv('MAKEHUMAN_BASE_NPZ', '/usr/share/makehuman-community/data/3dobjs/base.npz')))
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('--scale', type=float, default=0.1)
parser.add_argument('--group-id', type=int, default=0, help='Groupe MakeHuman à exporter; 0 correspond à body')
parser.add_argument('--all-groups', action='store_true', help='Exporter tous les groupes au lieu du seul groupe body')
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
data = np.load(args.input, allow_pickle=True)
points = args.scale * np.asarray(data['coord'], dtype=float)
all_quads = np.asarray(data['fvert'], dtype=np.int64)
all_groups = np.asarray(data['group'], dtype=np.int64)
if args.all_groups:
    selected = np.ones(len(all_quads), dtype=bool)
else:
    selected = all_groups == args.group_id
quads = all_quads[selected]
groups = all_groups[selected]
original_face_ids = np.flatnonzero(selected).astype(np.int64)
triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=0)
triangle_group = np.repeat(groups, 2)
triangle_face_id = np.repeat(original_face_ids, 2)
valid = (
    np.all(triangles >= 0, axis=1)
    & np.all(triangles < len(points), axis=1)
    & (triangles[:, 0] != triangles[:, 1])
    & (triangles[:, 1] != triangles[:, 2])
    & (triangles[:, 0] != triangles[:, 2])
)
triangles = triangles[valid]
triangle_group = triangle_group[valid]
triangle_face_id = triangle_face_id[valid]
mesh = meshio.Mesh(
    points=points,
    cells=[('triangle', triangles)],
    cell_data={
        'makehuman_group': [triangle_group],
        'makehuman_face_id': [triangle_face_id],
    },
)
meshio.write(args.out_dir / 'human_body.vtk', mesh)
meshio.write(args.out_dir / 'human_body.obj', mesh)
meshio.write(args.out_dir / 'human_body.stl', mesh)
report = {
    'input': str(args.input),
    'scale': args.scale,
    'input_quads': int(len(quads)),
    'input_quads_all_groups': int(len(all_quads)),
    'selected_group_id': None if args.all_groups else int(args.group_id),
    'output_triangles': int(len(triangles)),
    'points': int(len(points)),
    'groups': int(len(np.unique(triangle_group))),
    'area_m2': None,
    'note': 'meshio preserves cell_data in VTK; OBJ/STL are geometry-only exchange formats',
}
(args.out_dir / 'meshio_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
