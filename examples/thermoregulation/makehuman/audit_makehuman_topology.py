from __future__ import annotations
from pathlib import Path
import json
import numpy as np

source = Path('/usr/share/makehuman-community/data/3dobjs/base.npz')
out = Path('/home/ubuntu/foampilot/examples/thermoregulation/validation/makehuman_topology_audit.json')
d = np.load(source, allow_pickle=True)
V = np.asarray(d['coord'], dtype=float) * 0.1
Q = np.asarray(d['fvert'], dtype=np.int64)
G = np.asarray(d['group'], dtype=np.int64)
F = np.concatenate((Q[:,[0,1,2]], Q[:,[0,2,3]], Q[:,[0,3,0]]), axis=0)[:0]  # explicit below
F = np.vstack((Q[:,[0,1,2]], Q[:,[0,2,3]]))
FG = np.repeat(G, 2)
# undirected edge multiplicities
edges = np.sort(np.vstack((F[:,[0,1]], F[:,[1,2]], F[:,[2,0]])), axis=1)
uniq, counts = np.unique(edges, axis=0, return_counts=True)
boundary = uniq[counts == 1]
nonmanifold = uniq[counts > 2]
# face adjacency through shared edges
edge_to_faces = {}
for fi, tri in enumerate(F):
    for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
        e = tuple(sorted((int(a),int(b))))
        edge_to_faces.setdefault(e, []).append(fi)
parent = np.arange(len(F))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a,b):
    a,b=find(a),find(b)
    if a!=b: parent[b]=a
for fs in edge_to_faces.values():
    for j in fs[1:]: union(fs[0],j)
roots, comp_sizes = np.unique([find(i) for i in range(len(F))], return_counts=True)
# signed volume per connected face component and group summaries
result = {
    'vertices': int(len(V)), 'triangles': int(len(F)), 'source_quads': int(len(Q)),
    'unique_edges': int(len(uniq)), 'boundary_edges': int(len(boundary)),
    'nonmanifold_edges': int(len(nonmanifold)), 'face_components': int(len(roots)),
    'component_face_counts_top20': sorted(comp_sizes.tolist(), reverse=True)[:20],
    'group_count': int(len(np.unique(G))),
    'groups': {}
}
for gid in np.unique(G):
    m = FG == gid
    result['groups'][str(int(gid))] = {'triangles': int(m.sum()), 'area_m2': float(sum(0.5*np.linalg.norm(np.cross(V[t[1]]-V[t[0]], V[t[2]]-V[t[0]])) for t in F[m]))}
# representative boundary edge bounding boxes and edge incidence histogram
result['edge_incidence_histogram'] = {str(int(k)): int((counts == k).sum()) for k in np.unique(counts)}
result['boundary_edge_bbox_m'] = None if not len(boundary) else {'min': V[boundary].reshape(-1,3).min(0).tolist(), 'max': V[boundary].reshape(-1,3).max(0).tolist()}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
