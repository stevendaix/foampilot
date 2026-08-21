from pathlib import Path
import json, numpy as np, pyvista as pv
root=Path('/home/ubuntu/foampilot_pr_repo'); out=root/'examples/medical_build/outputs'; out.mkdir(exist_ok=True)
files=[
 root/'foampilot/test/vmtk_test_data/aorta-surface.vtp',
 root/'foampilot/test/vmtk_test_data/aorta-surface-branch-split.vtp',
 Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package/openfoam_surface_patches/aorta_surface_patches.vtp'),
 root/'examples/medical_build/outputs/complex_vmtk_reference_clean.stl',
 root/'examples/medical_build/openfoam_case/constant/triSurface/aorta_surface.stl',
 root/'examples/medical_build/openfoam_case/constant/triSurface/aorta_multiregion.stl',
 Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package/openfoam_surface_patches/case_motorbike_style/constant/triSurface/wall.stl'),
]
def audit(p):
 r={'file':str(p),'exists':p.exists()}
 if not p.exists(): return r
 try:
  m=pv.read(p).extract_surface().triangulate()
  f=m.faces.reshape(-1,4)[:,1:] if m.n_cells else np.empty((0,3),int)
  keys=np.sort(f,axis=1) if len(f) else f
  dup=int(len(f)-len(np.unique(keys,axis=0))) if len(f) else 0
  fe=m.extract_feature_edges(boundary_edges=True,non_manifold_edges=True,feature_edges=False,manifold_edges=False); r.update(points=int(m.n_points),cells=int(m.n_cells),bounds=[float(x) for x in m.bounds],volume=float(m.volume),area=float(m.area),open_edges=int(m.n_open_edges),nonmanifold_edges=int(fe.n_cells-int(m.n_open_edges)),duplicate_triangles=dup,cell_arrays=list(m.cell_data.keys()))
  if 'PatchId' in m.cell_data: r['patch_counts']={str(int(k)):int(v) for k,v in zip(*np.unique(np.asarray(m.cell_data['PatchId']),return_counts=True))}
 except Exception as e: r['error']=repr(e)
 return r
res={'audit':[audit(p) for p in files]}
(out/'exhaustive_geometry_audit.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
