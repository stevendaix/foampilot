from pathlib import Path
import json,numpy as np
from foampilot.geometry.medical_build.global_blockmesh import GlobalBlockMesh
ROOT=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package'); OUT=ROOT/'global_blockmesh_diagnostic'; OUT.mkdir(exist_ok=True)
data=json.loads((ROOT/'analysis_sections.json').read_text())
def sample(points,n=8):
 p=np.asarray(points,float); p=p[:-1] if len(p)>1 and np.linalg.norm(p[0]-p[-1])<1e-8 else p; q=np.vstack([p,p[0]]); seg=np.linalg.norm(np.diff(q,axis=0),axis=1); cum=np.r_[0,np.cumsum(seg)]; out=[]
 for d in np.linspace(0,cum[-1],n,endpoint=False):
  k=min(max(int(np.searchsorted(cum,d,side='right')-1),0),len(p)-1); u=(d-cum[k])/max(seg[k],1e-12); out.append(q[k]*(1-u)+q[k+1]*u)
 return np.asarray(out)
mesh=GlobalBlockMesh(tolerance=1e-6); skipped=[]
for b in data['branches']:
 sections=b['sections'][::5]
 for s0,s1 in zip(sections[:-1],sections[1:]):
  o0=sample(s0['phase_locked_points']); o1=sample(s1['phase_locked_points']); c0=np.asarray(s0['center']); c1=np.asarray(s1['center']); i0=c0+.35*(o0-c0); i1=c1+.35*(o1-c1)
  for k in range(8):
   j=(k+1)%8
   try: mesh.add_block([i0[k],o0[k],o0[j],i0[j],i1[k],o1[k],o1[j],i1[j]],label=f'branch_{b["branch_id"]}_sector_{k}')
   except ValueError as exc: skipped.append({'branch_id':b['branch_id'],'sector':k,'error':str(exc)})
validation=mesh.validate(require_connected=False); validation['connected_required_result']=mesh.validate(require_connected=True); validation['skipped_blocks']=skipped
(mesh.write(OUT/'diagnostic_global_blockMeshDict'))
(OUT/'connectivity_report.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(validation,indent=2))
