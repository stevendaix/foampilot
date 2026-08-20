from pathlib import Path
import json,time,traceback
import numpy as np
import classy_blocks as cb
ROOT=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package'); OUT=ROOT/'classy_blocks_benchmark'; OUT.mkdir(exist_ok=True)
data=json.loads((ROOT/'analysis_sections.json').read_text())
def sample_closed(points,n=8):
 p=np.asarray(points,float); p=p[:-1] if len(p)>1 and np.linalg.norm(p[0]-p[-1])<1e-8 else p; q=np.vstack([p,p[0]]); seg=np.linalg.norm(np.diff(q,axis=0),axis=1); cum=np.r_[0,np.cumsum(seg)]; out=[]
 for d in np.linspace(0,cum[-1],n,endpoint=False):
  k=min(max(int(np.searchsorted(cum,d,side='right')-1),0),len(p)-1); u=(d-cum[k])/max(seg[k],1e-12); out.append(q[k]*(1-u)+q[k+1]*u)
 return np.asarray(out)
def section_faces(sec,n=8,inner_scale=.35):
 outer=sample_closed(sec['phase_locked_points'],n); c=np.asarray(sec['center']); inner=c+inner_scale*(outer-c)
 return [cb.Face([inner[i],outer[i],outer[(i+1)%n],inner[(i+1)%n]],check_coplanar=True) for i in range(n)]
report={'branches':[],'parameters':{'stations':20,'radial_sectors':8,'inner_scale':.35,'cells_per_block':[2,2,4]}}
for b in data['branches']:
 start=time.perf_counter(); row={'branch_id':b['branch_id'],'ok':False,'n_sections_total':len(b['sections'])}
 try:
  sections=b['sections'][::max(1,len(b['sections'])//20)]; mesh=cb.Mesh(); nblocks=0
  for s0,s1 in zip(sections[:-1],sections[1:]):
   for lo,hi in zip(section_faces(s0),section_faces(s1)):
    op=cb.Loft(lo,hi); op.chop(0,count=2); op.chop(1,count=2); op.chop(2,count=4); mesh.add(op); nblocks+=1
  path=OUT/f'branch_{b["branch_id"]:02d}_ogrid_blockMeshDict'; debug=OUT/f'branch_{b["branch_id"]:02d}_ogrid_debug.vtk'; mesh.write(str(path),debug_path=str(debug))
  row.update(ok=True,sections_used=len(sections),blocks=nblocks,vertices=len(mesh.vertices),time_s=time.perf_counter()-start,blockMeshDict=str(path),debug_vtk=str(debug))
 except Exception as e: row.update(error=type(e).__name__+': '+str(e)[:500],traceback=traceback.format_exc(limit=3),time_s=time.perf_counter()-start)
 print(row,flush=True); report['branches'].append(row)
(OUT/'classy_blocks_complex_report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
print({'report':str(OUT/'classy_blocks_complex_report.json'),'valid':sum(bool(x['ok']) for x in report['branches']),'total':len(report['branches'])})
