from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--stride',type=int,default=12); args=ap.parse_args(); t0=time.perf_counter(); import vtk
 def read(p): r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(p)); r.Update(); return r.GetOutput()
 surface=read(args.root/'aorta-surface.vtp'); cl=read(args.root/'aorta-centerline-branches.vtp'); out=args.output; out.mkdir(parents=True,exist_ok=True); branches=[]
 for ci in range(cl.GetNumberOfCells()):
  cell=cl.GetCell(ci); ids=[cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]; sections=[]
  for k in range(0,len(ids),max(1,args.stride)):
   pid=ids[k]; c=np.array(cl.GetPoint(pid),float); prev=np.array(cl.GetPoint(ids[max(0,k-1)]),float); nxt=np.array(cl.GetPoint(ids[min(len(ids)-1,k+1)]),float); t=nxt-prev; t=t/max(np.linalg.norm(t),1e-12); plane=vtk.vtkPlane(); plane.SetOrigin(*c); plane.SetNormal(*t); cutter=vtk.vtkCutter(); cutter.SetInputData(surface); cutter.SetCutFunction(plane); cutter.Update(); stripper=vtk.vtkStripper(); stripper.SetInputConnection(cutter.GetOutputPort()); stripper.JoinContiguousSegmentsOn(); stripper.Update(); poly=stripper.GetOutput(); best=[]
   for q in range(poly.GetNumberOfCells()):
    line=poly.GetCell(q); arr=np.array([poly.GetPoint(line.GetPointId(j)) for j in range(line.GetNumberOfPoints())],float); best.append(arr)
   if best:
    contour=max(best,key=len); sections.append({'point_id':int(pid),'center':c.tolist(),'tangent':t.tolist(),'points':contour.tolist(),'closed':bool(np.linalg.norm(contour[0]-contour[-1])<1e-5),'point_count':len(contour)})
  branches.append({'branch_id':ci,'centerline_point_ids':ids,'sections':sections})
 report={'surface_points':surface.GetNumberOfPoints(),'surface_cells':surface.GetNumberOfCells(),'centerline_points':cl.GetNumberOfPoints(),'branch_count':cl.GetNumberOfCells(),'branches':branches,'elapsed_seconds':round(time.perf_counter()-t0,6)}; (out/'vmtk_real_sections.json').write_text(json.dumps(report,indent=2)); print(json.dumps({'branch_count':len(branches),'sections':sum(len(b['sections']) for b in branches),'closed_sections':sum(sum(int(s['closed']) for s in b['sections']) for b in branches),'elapsed_seconds':report['elapsed_seconds']},indent=2))
if __name__=='__main__': main()
