from pathlib import Path
import json, numpy as np, vtk
BASE=Path(__file__).resolve().parents[2]; SRC=BASE/'medical_build_complex_source_cap4'/'medical_build_input.vtp'; ROOT=BASE/'complex_analysis_raw_package'; OUT=ROOT/'openfoam_surface_patches_clean'; TRI=OUT/'constant'/'triSurface'; TRI.mkdir(parents=True,exist_ok=True)
def read_vtp(path):
 r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); p=vtk.vtkPolyData(); p.DeepCopy(r.GetOutput()); return p
def write_stl(p,path):
 w=vtk.vtkSTLWriter(); w.SetFileName(str(path)); w.SetInputData(p); w.SetFileTypeToBinary(); w.Write()
def loop_polydata(src):
 fe=vtk.vtkFeatureEdges(); fe.SetInputData(src); fe.BoundaryEdgesOn(); fe.FeatureEdgesOff(); fe.NonManifoldEdgesOff(); fe.ManifoldEdgesOff(); fe.Update()
 st=vtk.vtkStripper(); st.SetInputConnection(fe.GetOutputPort()); st.JoinContiguousSegmentsOn(); st.Update(); q=st.GetOutput(); return q
src=read_vtp(SRC); loops=loop_polydata(src); lines=loops.GetLines(); lines.InitTraversal(); ids=vtk.vtkIdList(); cap_polys=[]; cap_meta=[]
while lines.GetNextCell(ids):
 if ids.GetNumberOfIds()<3: continue
 # Each stripper line is ordered; close it explicitly for contour triangulation.
 poly=vtk.vtkPolyData(); pts=vtk.vtkPoints(); line=vtk.vtkPolyLine(); n=ids.GetNumberOfIds(); line.GetPointIds().SetNumberOfIds(n+1)
 for i in range(n): pts.InsertNextPoint(loops.GetPoint(ids.GetId(i))); line.GetPointIds().SetId(i,i)
 pts.InsertNextPoint(loops.GetPoint(ids.GetId(0))); line.GetPointIds().SetId(n,n)
 poly.SetPoints(pts); lc=vtk.vtkCellArray(); lc.InsertNextCell(line); poly.SetLines(lc)
 tri=vtk.vtkContourTriangulator(); tri.SetInputData(poly); tri.Update(); cap=vtk.vtkPolyData(); cap.DeepCopy(tri.GetOutput())
 if cap.GetNumberOfCells()==0: raise RuntimeError(f'failed triangulation for loop n={n}')
 cap_polys.append(cap); ca=np.asarray([cap.GetPoint(i) for i in range(cap.GetNumberOfPoints())]); cap_meta.append({'n_boundary_points':n,'n_triangles':cap.GetNumberOfCells(),'center':ca.mean(axis=0).tolist()})
# Original open surface is exactly the wall; caps are the triangulated original boundaries.
write_stl(src,TRI/'wall.stl')
# match caps to endpoint cap ids using the stored loop centers
analysis=json.loads((ROOT/'analysis_sections.json').read_text()); endpoints=[]
for b in analysis['branches']:
 endpoints += [(b['source_cap_id'],np.asarray(b['points'][0]),'inlet'),(b['target_cap_id'],np.asarray(b['points'][-1]),f'outlet_{b["target_cap_id"]}')]
used=set(); patch_info={}
for cap,meta in zip(cap_polys,cap_meta):
 c=np.asarray(meta['center']); cap_id,ep,name=min((x for x in endpoints if x[0] not in used),key=lambda x:np.linalg.norm(x[1]-c)); used.add(cap_id); write_stl(cap,TRI/f'{name}.stl'); patch_info[name]={**meta,'cap_id':cap_id,'stl':str(TRI/f'{name}.stl')}
# combined surface: exact original wall + exact caps
app=vtk.vtkAppendPolyData(); app.AddInputData(src)
for cap in cap_polys: app.AddInputData(cap)
app.Update(); clean=vtk.vtkCleanPolyData(); clean.SetInputConnection(app.GetOutputPort()); clean.Update(); combined=vtk.vtkPolyData(); combined.DeepCopy(clean.GetOutput()); write_stl(combined,OUT/'aorta_closed_combined.stl')
manifest={'source':str(SRC),'wall_is_original_open_surface':True,'n_boundary_loops':len(cap_polys),'patches':{'wall':{'triangles':src.GetNumberOfCells(),'stl':str(TRI/'wall.stl')},**{k:v for k,v in patch_info.items()}},'combined':{'triangles':combined.GetNumberOfCells(),'stl':str(OUT/'aorta_closed_combined.stl')}}
(OUT/'clean_patch_manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
