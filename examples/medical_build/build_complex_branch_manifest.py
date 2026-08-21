from pathlib import Path
import json
import numpy as np
import pyvista as pv
ROOT=Path('/home/ubuntu/foampilot_pr_repo'); SRC=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package'); OUT=ROOT/'examples/medical_build/outputs'
CAPS=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package/openfoam_surface_patches/patch_manifest.json')
def main():
 cl=pv.read(SRC/'centerlines.vtp'); manifest=json.loads(CAPS.read_text()); cap_items=[(str(v['cap_id']),v) for v in manifest['classification']]
 rows=[]
 for i in range(cl.n_cells):
  part=cl.extract_cells([i]); p=np.asarray(part.points,float); a=p[0]; b=p[-1]; length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
  d0=[(float(np.linalg.norm(a-np.asarray(v['loop_center']))),k) for k,v in cap_items]; d1=[(float(np.linalg.norm(b-np.asarray(v['loop_center']))),k) for k,v in cap_items]
  rows.append({'cell_id':i,'n_points':len(p),'length':length,'start':a.tolist(),'end':b.tolist(),'nearest_start_cap':min(d0)[1],'nearest_start_distance':min(d0)[0],'nearest_end_cap':min(d1)[1],'nearest_end_distance':min(d1)[0]})
 result={'centerlines':str(SRC/'centerlines.vtp'),'surface_manifest':str(CAPS),'n_cells':cl.n_cells,'n_caps':len(cap_items),'cells':rows}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'complex_branch_cap_manifest.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
