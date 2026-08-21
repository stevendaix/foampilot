from pathlib import Path
import re
import numpy as np
case=Path('/home/ubuntu/foampilot/openfoam_runs/meshio_body_only_case')/'constant/polyMesh'
def body(name):
    s=(case/'sets'/name).read_text()
    m=re.search(r'\n\s*\d+\s*\n\s*\(\s*(.*?)\s*\)',s,re.S)
    return [int(x) for x in re.findall(r'-?\d+',m.group(1))] if m else []
def foam_points(path):
    rows=[]
    for line in path.read_text().splitlines():
        m=re.match(r'\s*\((-?\S+)\s+(-?\S+)\s+(-?\S+)\)', line)
        if m: rows.append([float(v) for v in m.groups()])
    return np.asarray(rows, dtype=float)
def foam_faces(path):
    out=[]
    for line in path.read_text().splitlines():
        m=re.match(r'\s*\d+\(([^()]*)\)', line)
        if m: out.append(list(map(int, re.findall(r'-?\d+', m.group(1)))))
    return out
P=foam_points(case/'points'); F=foam_faces(case/'faces')
for name in ['skewFaces','concaveFaces','warpedFaces']:
    ids=body(name); print(name,len(ids))
    for i in ids[:20]:
        c=P[F[i]].mean(0); print(' ',i,'centroid',c.tolist(),'npoints',len(F[i]))
for name in ['nonManifoldPoints']:
    ids=body(name); print(name,len(ids))
    for i in ids: print(' ',i,P[i].tolist())
