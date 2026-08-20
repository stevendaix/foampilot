from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('branch_dir',type=Path); ap.add_argument('--pitch',type=float,default=.75); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); import trimesh
    files=sorted(args.branch_dir.glob('branch_*.stl')); grids=[]; allp=[]
    for f in files:
        m=trimesh.load_mesh(f,process=False); g=m.voxelized(args.pitch).fill(); p=np.asarray(g.points); grids.append((f.name,p)); allp.append(p)
    p=np.vstack(allp); origin=np.floor(p.min(0)/args.pitch)*args.pitch; shape=np.ceil((p.max(0)-origin)/args.pitch).astype(int)+3; counts=np.zeros(tuple(shape),np.uint16)
    branch_stats=[]
    for name,pts in grids:
        idx=np.rint((pts-origin)/args.pitch).astype(int); ok=np.all((idx>=0)&(idx<shape),1); idx=idx[ok]; counts[tuple(idx.T)]+=1; branch_stats.append({'branch':name,'voxels':int(len(idx)),'volume':float(len(idx)*args.pitch**3)})
    hist={str(i):int(np.sum(counts==i)) for i in range(0,len(files)+1) if np.any(counts==i)}
    union=int(np.sum(counts>0)); overlap=int(np.sum(counts>1)); result={'pitch':args.pitch,'grid_shape':shape.tolist(),'branches':branch_stats,'sum_branch_voxels':int(counts.sum()),'union_voxels':union,'overlap_voxels':overlap,'overlap_fraction_of_sum':float(overlap/counts.sum()),'union_volume':float(union*args.pitch**3),'sum_branch_volume':float(counts.sum()*args.pitch**3),'histogram':hist}
    args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
