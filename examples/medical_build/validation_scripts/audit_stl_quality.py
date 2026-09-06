from pathlib import Path
import json, numpy as np
import vtk, trimesh
from pymeshfix import _meshfix

ROOT=Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/case_motorbike_style/constant/triSurface'

def vtk_metrics(path):
 r=vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update(); p=vtk.vtkPolyData(); p.DeepCopy(r.GetOutput())
 clean=vtk.vtkCleanPolyData(); clean.SetInputData(p); clean.Update(); c=clean.GetOutput()
 fe=vtk.vtkFeatureEdges(); fe.SetInputData(c); fe.BoundaryEdgesOn(); fe.NonManifoldEdgesOn(); fe.FeatureEdgesOff(); fe.ManifoldEdgesOff(); fe.Update()
 # independent edge count using vtkExtractEdges and point-id incidence
 edges=vtk.vtkExtractEdges(); edges.SetInputData(c); edges.Update(); e=edges.GetOutput(); edge_counts={}
 for i in range(c.GetNumberOfCells()):
  ids=c.GetCell(i).GetPointIds(); n=ids.GetNumberOfIds()
  for j in range(n):
   a,b=ids.GetId(j),ids.GetId((j+1)%n); key=tuple(sorted((a,b))); edge_counts[key]=edge_counts.get(key,0)+1
 vals=list(edge_counts.values())
 return {'vtk_points':p.GetNumberOfPoints(),'vtk_triangles':p.GetNumberOfCells(),'clean_points':c.GetNumberOfPoints(),'boundary_edges':sum(v==1 for v in vals),'nonmanifold_edges':sum(v>2 for v in vals),'edge_histogram':{str(k):vals.count(k) for k in sorted(set(vals))},'bounds':list(p.GetBounds())}

def trimesh_metrics(path):
 m=trimesh.load_mesh(path, file_type='stl', process=False)
 if not isinstance(m,trimesh.Trimesh): m=trimesh.util.concatenate(tuple(m.geometry.values()))
 faces_sorted=np.sort(np.asarray(m.faces),axis=1); dup=int(len(faces_sorted)-len(np.unique(faces_sorted,axis=0))); deg=int(np.sum(np.asarray(m.area_faces)<=1e-14))
 return {'vertices':int(len(m.vertices)),'faces':int(len(m.faces)),'is_watertight':bool(m.is_watertight),'is_winding_consistent':bool(m.is_winding_consistent),'is_volume':bool(m.is_volume),'euler_number':int(m.euler_number),'duplicate_faces':dup,'degenerate_faces':deg,'signed_volume':float(m.volume),'area':float(m.area)}

report={'files':{}}
for path in sorted(ROOT.glob('*.stl')):
 try: report['files'][path.name]={'vtk':vtk_metrics(path),'trimesh':trimesh_metrics(path)}
 except Exception as exc: report['files'][path.name]={'error':f'{type(exc).__name__}: {exc}'}
report['summary']={'all_watertight':all(x.get('trimesh',{}).get('is_watertight',False) for x in report['files'].values()),'any_boundary_edges':any(x.get('vtk',{}).get('boundary_edges',0)>0 for x in report['files'].values()),'any_nonmanifold':any(x.get('vtk',{}).get('nonmanifold_edges',0)>0 for x in report['files'].values())}
out=ROOT.parent.parent/'stl_quality_audit.json'; out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
