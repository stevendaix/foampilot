from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np

def unit(x): x=np.asarray(x,float); return x/max(np.linalg.norm(x),1e-12)
def main():
 d=json.loads(Path(sys.argv[1]).read_text()); rows=[]
 for b in d['branches']:
  p=np.asarray(b['points'],float); secs=b['sections']; first=secs[0]; last=secs[-1]; t0=unit(np.asarray(first.get('tangent',first.get('direction',[0,0,1])))); t1=unit(np.asarray(last.get('tangent',last.get('direction',[0,0,1])))); c0=np.asarray(first['center'],float); c1=np.asarray(last['center'],float); prof=first.get('phase_locked_points') or first.get('points'); radius=float(np.mean(np.linalg.norm(np.asarray(prof)-c0,axis=1))); rows.append({'branch_id':int(b['branch_id']),'source_cap_id':b.get('source_cap_id'),'target_cap_id':b.get('target_cap_id'),'first_center':c0.tolist(),'last_center':c1.tolist(),'first_tangent':t0.tolist(),'last_tangent':t1.tolist(),'first_radius':radius,'last_radius':float(np.mean(np.linalg.norm(np.asarray(last.get('phase_locked_points') or last.get('points'))-c1,axis=1)))})
 print(json.dumps(rows,indent=2)); out=Path(sys.argv[2]); out.write_text(json.dumps({'branches':rows},indent=2))
 centers=np.array([r['first_center'] for r in rows]); print('first_center_bbox',centers.min(0).tolist(),centers.max(0).tolist());
 for i in range(len(rows)):
  for j in range(i+1,len(rows)):
   d=np.linalg.norm(centers[i]-centers[j]);
   if d<20: print('near_first',i,j,d)
if __name__=='__main__': main()
