from pathlib import Path
import json, shutil
import numpy as np
import vtk

root=Path(__file__).resolve().parents[2]
src=root/'medical_build_complex_source_cap4'
out=root/'complex_analysis_raw_package'
out.mkdir(parents=True, exist_ok=True)

def read_poly(path):
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); p=vtk.vtkPolyData(); p.DeepCopy(r.GetOutput()); return p

poly=read_poly(src/'medical_build_centerlines.vtp')
# Preserve exact VTK output and export each line as a numerical branch array.
shutil.copy2(src/'medical_build_centerlines.vtp', out/'centerlines.vtp')
lines=poly.GetLines(); lines.InitTraversal(); ids=vtk.vtkIdList(); branches=[]; i=0
while lines.GetNextCell(ids):
    pts=np.array([poly.GetPoint(ids.GetId(j)) for j in range(ids.GetNumberOfIds())],dtype=float)
    record={'branch_id':i,'n_points':int(len(pts)),'length':float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()),'start':pts[0].tolist(),'end':pts[-1].tolist()}
    arrays={}
    for k in range(poly.GetPointData().GetNumberOfArrays()):
        name=poly.GetPointData().GetArrayName(k); arr=poly.GetPointData().GetArray(name)
        values=np.array([arr.GetTuple(ids.GetId(j)) for j in range(ids.GetNumberOfIds())],dtype=float)
        arrays[name]=values
    np.savez_compressed(out/f'branch_{i:02d}.npz',points=pts,**arrays)
    branches.append(record); i+=1

for name in ('medical_build_vmtk_diagnostics.json','topology.json','preprocess_quality.json','summary.json'):
    p=src/name
    if p.exists(): shutil.copy2(p,out/name)
metadata={'source_campaign':str(src),'points':int(poly.GetNumberOfPoints()),'lines':int(poly.GetNumberOfLines()),'point_arrays':[poly.GetPointData().GetArrayName(k) for k in range(poly.GetPointData().GetNumberOfArrays())],'branches':branches,'files':sorted(p.name for p in out.iterdir())}
(out/'raw_inventory.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
