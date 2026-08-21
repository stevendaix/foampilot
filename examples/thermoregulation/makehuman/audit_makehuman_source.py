from __future__ import annotations
from pathlib import Path
import json
import numpy as np

source = Path('/usr/share/makehuman-community/data/3dobjs/base.npz')
out = Path('/home/ubuntu/foampilot/examples/thermoregulation/validation/makehuman_source_audit.json')
data = np.load(source, allow_pickle=True)
result = {'source': str(source), 'keys': list(data.files)}
for key in data.files:
    arr = data[key]
    result.setdefault('arrays', {})[key] = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}
coord = np.asarray(data['coord'], dtype=float)
fvert = np.asarray(data['fvert'], dtype=np.int64)
group = np.asarray(data['group'], dtype=np.int64)
result['coord_min_raw'] = coord.min(axis=0).tolist()
result['coord_max_raw'] = coord.max(axis=0).tolist()
result['coord_span_raw'] = (coord.max(axis=0)-coord.min(axis=0)).tolist()
result['vertex_count'] = int(len(coord))
result['quad_count'] = int(len(fvert))
result['group_count'] = int(len(np.unique(group)))
result['group_values'] = np.unique(group).tolist()
result['invalid_index_count'] = int((~((fvert >= 0).all(1) & (fvert < len(coord)).all(1))).sum())
result['repeated_vertex_quad_count'] = int(((fvert[:,0] == fvert[:,1]) | (fvert[:,0] == fvert[:,2]) | (fvert[:,0] == fvert[:,3]) | (fvert[:,1] == fvert[:,2]) | (fvert[:,1] == fvert[:,3]) | (fvert[:,2] == fvert[:,3])).sum())
tri = np.concatenate((fvert[:, [0,1,2]], fvert[:, [0,2,3]]), axis=0)
valid = (tri >= 0).all(1) & (tri < len(coord)).all(1)
t = coord[tri[valid]] * 0.1
cross = np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0])
area = 0.5*np.linalg.norm(cross, axis=1)
result['triangles_after_split'] = int(len(t))
result['degenerate_triangle_count'] = int((area <= 1e-14).sum())
result['area_m2'] = float(area.sum())
result['bounding_box_m'] = {'min': (coord.min(0)*0.1).tolist(), 'max': (coord.max(0)*0.1).tolist()}
try:
    import trimesh
    mesh = trimesh.Trimesh(vertices=coord*0.1, faces=tri[valid], process=False)
    result['trimesh'] = {'watertight': bool(mesh.is_watertight), 'winding_consistent': bool(mesh.is_winding_consistent), 'components': int(len(mesh.split(only_watertight=False))), 'volume_signed_m3': float(mesh.volume)}
except Exception as exc:
    result['trimesh_error'] = repr(exc)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
