from pathlib import Path
import json
import numpy as np
import vtk

ROOT = Path('/home/ubuntu/foampilot_pr_repo')
SURFACE = ROOT/'foampilot/test/vmtk_test_data/aorta-surface.vtp'
CENTERLINES = ROOT/'foampilot/test/vmtk_test_data/aorta-centerline-branches.vtp'
OUTPUTS = [
 ROOT/'examples/medical_build/outputs/vmtk_like_polyball_aorta.vtp',
 ROOT/'examples/medical_build/outputs/reconstructed_local_deformation/reference/aorta_sections_combined.stl',
]

def read(path):
 r = vtk.vtkXMLPolyDataReader() if path.suffix == '.vtp' else vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update(); return r.GetOutput()

def bounds(p):
 b=p.GetBounds(); return [b[1]-b[0],b[3]-b[2],b[5]-b[4]]

def endpoints(d):
 out=[]
 for ci in range(d.GetNumberOfCells()):
  cell=d.GetCell(ci); ids=[cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
  out.append({'branch_id':ci,'point_count':len(ids),'start':list(d.GetPoint(ids[0])),'end':list(d.GetPoint(ids[-1])),'length':sum(np.linalg.norm(np.asarray(d.GetPoint(ids[j]))-np.asarray(d.GetPoint(ids[j-1]))) for j in range(1,len(ids)))})
 return out

surface=read(SURFACE); cl=read(CENTERLINES)
result={'reference_surface':{'bounds':list(surface.GetBounds()),'dimensions':bounds(surface)},'centerlines':{'bounds':list(cl.GetBounds()),'dimensions':bounds(cl),'branches':endpoints(cl)},'outputs':[]}
for p in OUTPUTS:
 if p.exists():
  d=read(p); result['outputs'].append({'file':str(p),'bounds':list(d.GetBounds()),'dimensions':bounds(d),'points':d.GetNumberOfPoints(),'cells':d.GetNumberOfCells()})
out=ROOT/'examples/medical_build/outputs/missing_segments_audit.json'; out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
