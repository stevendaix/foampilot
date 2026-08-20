from pathlib import Path
import json
import numpy as np
import vtk

BASE=Path('/home/ubuntu/vmtk_audit_extract'); ROOT=BASE/'complex_analysis_raw_package'; OUT=ROOT/'openfoam_surface_patches'; TRI=OUT/'constant'/'triSurface'; TRI.mkdir(parents=True,exist_ok=True)

def read_poly(path):
 r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); p=vtk.vtkPolyData(); p.DeepCopy(r.GetOutput()); return p

def write_poly(poly,path):
 w=vtk.vtkSTLWriter(); w.SetFileName(str(path)); w.SetInputData(poly); w.SetFileTypeToBinary(); w.Write()

def subset(src,ids):
 used=sorted({src.GetCell(cid).GetPointId(j) for cid in ids for j in range(src.GetCell(cid).GetNumberOfPoints())}); rem={old:i for i,old in enumerate(used)}; pts=vtk.vtkPoints(); [pts.InsertNextPoint(src.GetPoint(i)) for i in used]; polys=vtk.vtkCellArray()
 for cid in ids:
  cell=src.GetCell(cid); pids=cell.GetPointIds(); polys.InsertNextCell(pids.GetNumberOfIds()); [polys.InsertCellPoint(rem[pids.GetId(j)]) for j in range(pids.GetNumberOfIds())]
 out=vtk.vtkPolyData(); out.SetPoints(pts); out.SetPolys(polys); return out

surface=read_poly(BASE/'medical_build_complex_source_cap4'/'capped_surface.vtp')
centers=vtk.vtkCellCenters(); centers.SetInputData(surface); centers.Update(); ca=np.asarray([centers.GetOutput().GetPoint(i) for i in range(surface.GetNumberOfCells())])
normals=vtk.vtkPolyDataNormals(); normals.SetInputData(surface); normals.ComputeCellNormalsOn(); normals.SplittingOff(); normals.Update(); na=np.asarray([normals.GetOutput().GetCellData().GetNormals().GetTuple(i) for i in range(surface.GetNumberOfCells())])
data=json.loads((ROOT/'analysis_sections.json').read_text()); loops=json.loads((BASE/'boundary_loops.json').read_text())['loops']
# Match each physical cap loop to the nearest centerline endpoint.
endpoints=[]
for b in data['branches']:
 endpoints.append((b['source_cap_id'],np.asarray(b['points'][0],float),'source',b['branch_id']))
 endpoints.append((b['target_cap_id'],np.asarray(b['points'][-1],float),'target',b['branch_id']))
cap_info={}
for loop in loops:
 c=np.asarray(loop['center']); cap_id,endpoint,role,bid=min(endpoints,key=lambda x:np.linalg.norm(x[1]-c)); cap_info[int(cap_id)]={'center':c,'normal':np.asarray(loop['normal'],float),'radius':max(np.sqrt(float(loop['area_proxy'])/np.pi),0.5),'role':role,'branch_id':bid,'endpoint':endpoint}
assigned={}; stats=[]
for cap_id,info in sorted(cap_info.items()):
 dist=np.linalg.norm(ca-info['center'][None,:],axis=1); align=np.abs(na@info['normal']); threshold=max(2.2*info['radius'],2.5); ids=np.where((dist<=threshold)&(align>=0.70))[0].tolist()
 for cid in ids:
  if cid not in assigned or dist[cid]<assigned[cid][0]: assigned[cid]=(float(dist[cid]),cap_id)
 stats.append({'cap_id':cap_id,'role':info['role'],'loop_center':info['center'].tolist(),'loop_radius':info['radius'],'threshold':threshold,'candidate_cells':len(ids)})
source_cap=data['branches'][0]['source_cap_id']; patch_cells={('inlet' if cap_id==source_cap else f'outlet_{cap_id}'):[] for cap_id in cap_info}
for cid,(_,cap_id) in assigned.items(): patch_cells['inlet' if cap_id==source_cap else f'outlet_{cap_id}'].append(cid)
patch_cells['wall']=[cid for cid in range(surface.GetNumberOfCells()) if cid not in assigned]
manifest={'source':'medical_build_complex_source_cap4/capped_surface.vtp','n_source_cells':surface.GetNumberOfCells(),'patches':{},'classification':stats}
for name,ids in patch_cells.items():
 poly=subset(surface,ids); path=TRI/(name+'.stl'); write_poly(poly,path); manifest['patches'][name]={'cells':len(ids),'points':poly.GetNumberOfPoints(),'stl':str(path)}
label=vtk.vtkIntArray(); label.SetName('PatchId'); label.SetNumberOfTuples(surface.GetNumberOfCells()); names=['wall']+sorted(k for k in patch_cells if k!='wall'); idmap={n:i for i,n in enumerate(names)}
for i in range(surface.GetNumberOfCells()): label.SetValue(i,idmap['wall'])
for n,ids in patch_cells.items():
 for i in ids: label.SetValue(i,idmap[n])
surface.GetCellData().AddArray(label); vw=vtk.vtkXMLPolyDataWriter(); vw.SetFileName(str(OUT/'aorta_surface_patches.vtp')); vw.SetInputData(surface); vw.Write()
(OUT/'patch_manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
