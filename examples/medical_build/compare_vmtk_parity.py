from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
 import vtk
 p=Path(path); r=vtk.vtkXMLPolyDataReader() if p.suffix=='.vtp' else vtk.vtkSTLReader(); r.SetFileName(str(p)); r.Update(); return r.GetOutput()
def arrays(a,b):
 out={}
 names=sorted({a.GetPointData().GetArrayName(i) for i in range(a.GetPointData().GetNumberOfArrays())}|{b.GetPointData().GetArrayName(i) for i in range(b.GetPointData().GetNumberOfArrays())})
 for n in names:
  x=a.GetPointData().GetArray(n); y=b.GetPointData().GetArray(n); same=False; maxerr=None
  if x and y and x.GetNumberOfTuples()==y.GetNumberOfTuples() and x.GetNumberOfComponents()==y.GetNumberOfComponents():
   xa=np.array([x.GetTuple(i) for i in range(x.GetNumberOfTuples())]); ya=np.array([y.GetTuple(i) for i in range(y.GetNumberOfTuples())]); maxerr=float(np.max(np.abs(xa-ya))) if xa.size else 0.; same=bool(np.array_equal(xa,ya))
  out[n]={'present_a':bool(x),'present_b':bool(y),'exact':same,'max_abs_error':maxerr}
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('official',type=Path); ap.add_argument('local',type=Path); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); result={'files':{},'status':'exact_or_equivalent'}
 for name in ['aorta-surface.vtp','aorta-centerline.vtp','aorta-centerline-branches.vtp','aorta-surface.stl']:
  a=args.official/name; b=args.local/name; item={'official_exists':a.exists(),'local_exists':b.exists()}
  if a.exists() and b.exists():
   item['sha256_equal']=sha(a)==sha(b); da=read(a); db=read(b); item.update({'points_equal':da.GetNumberOfPoints()==db.GetNumberOfPoints(),'cells_equal':da.GetNumberOfCells()==db.GetNumberOfCells(),'bounds_max_abs_error':float(np.max(np.abs(np.asarray(da.GetBounds())-np.asarray(db.GetBounds())))),'point_arrays':arrays(da,db) if a.suffix=='.vtp' else {}})
  result['files'][name]=item
 result['status']='exact_files' if all(v.get('sha256_equal',False) for v in result['files'].values() if v.get('official_exists') and v.get('local_exists')) else 'not_exact'
 args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
