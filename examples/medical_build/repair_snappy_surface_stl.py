from pathlib import Path
import numpy as np, pyvista as pv
ROOT=Path('/home/ubuntu/foampilot_pr_repo/examples/medical_build/openfoam_case/constant/triSurface')
SRC=Path('/home/ubuntu/foampilot_pr_repo/examples/medical_build/outputs/complex_vmtk_reference_clean.stl')
OUT=ROOT/'aorta_surface_repaired.stl'
m=pv.read(SRC).extract_surface().triangulate().clean(tolerance=1e-8)
faces=m.faces.reshape(-1,4); tri=faces[:,1:]
# remove degenerate and duplicate triangles regardless of orientation
valid=np.all(np.diff(np.sort(tri,axis=1),axis=1)>0,axis=1)
tri=tri[valid]
keys=np.sort(tri,axis=1); _, keep=np.unique(keys,axis=0,return_index=True); tri=tri[np.sort(keep)]
arr=np.empty((len(tri),4),dtype=np.int64); arr[:,0]=3; arr[:,1:]=tri
out=pv.PolyData(m.points,arr.ravel())
out=out.extract_surface().triangulate().clean(tolerance=1e-8); out=out.fill_holes(20.0).triangulate().clean(tolerance=1e-8); out.save(OUT)
print({'input':str(SRC),'output':str(OUT),'points':out.n_points,'cells':out.n_cells,'volume':out.volume,'area':out.area,'open_edges':out.n_open_edges})
