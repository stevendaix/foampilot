from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('branch_dir',type=Path); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--pitch',type=float,default=0.5); ap.add_argument('--closing',type=int,default=0, help='binary closing iterations on the voxel union'); args=ap.parse_args(); t0=time.perf_counter(); import trimesh
 files=sorted(args.branch_dir.glob('branch_*.stl')); grids=[]; all_points=[]
 for p in files:
  mesh=trimesh.load_mesh(p,process=False); grid=mesh.voxelized(args.pitch).fill(); pts=np.asarray(grid.points,float); grids.append((p.name,grid,pts)); all_points.append(pts)
 points=np.vstack(all_points); origin=np.floor(points.min(axis=0)/args.pitch)*args.pitch; maxp=np.ceil(points.max(axis=0)/args.pitch)*args.pitch; shape=np.maximum(np.ceil((maxp-origin)/args.pitch).astype(int)+3,1); occ=np.zeros(tuple(shape),dtype=bool)
 for name,grid,pts in grids:
  idx=np.rint((pts-origin)/args.pitch).astype(int); valid=np.all((idx>=0)&(idx<np.asarray(shape)),axis=1); occ[tuple(idx[valid].T)]=True
 if args.closing > 0:
  from scipy import ndimage
  occ=ndimage.binary_closing(occ, iterations=args.closing)
 from trimesh.voxel import ops
 mesh=ops.matrix_to_marching_cubes(occ,pitch=args.pitch); mesh.apply_translation(origin); mesh.merge_vertices(); mesh.remove_unreferenced_vertices(); mesh.process(validate=True); mesh.export(args.output)
 report={'inputs':[x[0] for x in grids],'pitch':args.pitch,'closing':args.closing,'grid_shape':list(map(int,shape)),'occupied_voxels':int(occ.sum()),'vertices':int(len(mesh.vertices)),'faces':int(len(mesh.faces)),'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'components':len(mesh.split(only_watertight=False)) if hasattr(mesh,'split') else None,'volume':float(mesh.volume),'elapsed_seconds':round(time.perf_counter()-t0,6)}; args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
