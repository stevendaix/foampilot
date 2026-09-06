from pathlib import Path
import numpy as np, pyvista as pv
SRC=Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/aorta_surface_patches.vtp'; OUT=Path(__file__).resolve().parents[2] / 'examples/medical_build/openfoam_case/constant/triSurface/aorta_multiregion.stl'
names={0:'outlet_0',1:'outlet_1',2:'outlet_2',3:'outlet_3',4:'inlet',5:'outlet_5',6:'outlet_6',7:'outlet_7',8:'outlet_8',9:'wall'}
m=pv.read(SRC).triangulate(); f=m.faces.reshape(-1,4)[:,1:]; ids=np.asarray(m.cell_data['PatchId'])
def normal(t):
 n=np.cross(t[1]-t[0],t[2]-t[0]); q=np.linalg.norm(n); return n/q if q else np.zeros(3)
allkeys=[tuple(sorted(map(int,t))) for t in f]
counts={k:allkeys.count(k) for k in set(allkeys)}
seen=set()
with OUT.open('w') as h:
 for pid in sorted(set(ids.tolist())):
  h.write(f'solid {names.get(int(pid),"patch"+str(pid))}\n')
  for tri in f[ids==pid]:
   key=tuple(sorted(map(int,tri)))
   if counts.get(key,1)>1: continue
   if key in seen: continue
   seen.add(key)
   t=np.asarray(m.points[tri]); n=normal(t); h.write(f' facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n  outer loop\n')
   for p in t: h.write(f'   vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n')
   h.write('  endloop\n endfacet\n')
  h.write(f'endsolid {names.get(int(pid),"patch"+str(pid))}\n')
print(OUT)
