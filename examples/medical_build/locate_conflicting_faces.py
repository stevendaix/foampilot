import numpy as np, pyvista as pv, json
from pathlib import Path
p=Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/aorta_surface_patches.vtp'; out=Path(__file__).resolve().parents[2] / 'examples/medical_build/outputs'
m=pv.read(p).triangulate(); f=m.faces.reshape(-1,4)[:,1:]; ids=np.asarray(m.cell_data['PatchId']); keys=np.sort(f,axis=1); groups={}
for i,k in enumerate(map(tuple,keys)):
 groups.setdefault(k,[]).append(i)
rows=[]
for k,ii in groups.items():
 if len(ii)>1:
  rows.append({'cells':ii,'patch_ids':[int(ids[i]) for i in ii],'points':np.asarray(m.points[list(k)]).tolist()})
res={'input':str(p),'duplicate_groups':rows,'count':len(rows)}
(out/'conflicting_faces_detail.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
