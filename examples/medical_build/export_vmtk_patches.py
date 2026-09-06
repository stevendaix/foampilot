from __future__ import annotations
import argparse,json
from pathlib import Path
import vtk
from vmtk import vtkvmtk

def read(path):
    r=vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update(); return r.GetOutput()

def subset(poly, ids):
    points=vtk.vtkPoints(); points.DeepCopy(poly.GetPoints()); cells=vtk.vtkCellArray()
    for cid in ids:
        cell=poly.GetCell(cid); cells.InsertNextCell(cell)
    out=vtk.vtkPolyData(); out.SetPoints(points); out.SetPolys(cells); return out

def area(poly):
    t=vtk.vtkTriangleFilter(); t.SetInputData(poly); t.Update(); m=vtk.vtkMassProperties(); m.SetInputData(t.GetOutput()); m.Update(); return float(m.GetSurfaceArea())

def write(poly,path):
    w=vtk.vtkSTLWriter(); w.SetFileName(str(path)); w.SetInputData(poly); w.Write()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('output',type=Path); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    raw=read(args.input); cap=vtkvmtk.vtkvmtkCapPolyData(); cap.SetInputData(raw); cap.SetCellEntityIdsArrayName('CapIds'); cap.Update(); p=cap.GetOutput(); arr=p.GetCellData().GetArray('CapIds')
    groups={}
    for i in range(p.GetNumberOfCells()): groups.setdefault(int(arr.GetTuple1(i)),[]).append(i)
    stats=[]
    for gid,cids in sorted(groups.items()):
        sub=subset(p,cids); a=area(sub); stats.append({'cap_id':gid,'cells':len(cids),'area':a,'polydata':sub})
    caps=[x for x in stats if x['cap_id']!=1]; caps.sort(key=lambda x:x['area'],reverse=True)
    wall=next(x for x in stats if x['cap_id']==1); write(wall['polydata'],args.output/'wall_open.stl')
    patch_stats=[]
    for i,x in enumerate(caps):
        name='inlet' if i==0 else f'outlet_{i-1}'; write(x['polydata'],args.output/f'{name}.stl'); patch_stats.append({'name':name,'cap_id':x['cap_id'],'area':x['area'],'cells':x['cells']})
    (args.output/'patch_report.json').write_text(json.dumps({'input':str(args.input),'patches':patch_stats,'wall_cells':wall['cells'],'wall_area':area(wall['polydata'])},indent=2)); print(json.dumps({'patches':patch_stats,'wall_cells':wall['cells'],'wall_area':area(wall['polydata'])},indent=2))
if __name__=='__main__': main()
