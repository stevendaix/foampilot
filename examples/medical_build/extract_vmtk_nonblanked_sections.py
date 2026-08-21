from pathlib import Path
import json
import numpy as np
import vtk

ROOT=Path('/home/ubuntu/foampilot_pr_repo')
NPZ_ROOT=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package')
SURFACE=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package/openfoam_surface_patches/aorta_surface_patches.vtp')
OUT=ROOT/'examples/medical_build/outputs'
ACTIVE=[2,6,7]

def tangent(points,i):
 t=np.zeros(3); w=0
 if i>0:
  d=points[i]-points[i-1]; n=np.linalg.norm(d)
  if n>1e-12:t+=d/n; w+=1
 if i+1<len(points):
  d=points[i+1]-points[i]; n=np.linalg.norm(d)
  if n>1e-12:t+=d/n; w+=1
 t/=max(w,1); t/=max(np.linalg.norm(t),1e-12); return t

def cut(surface,origin,normal):
 plane=vtk.vtkPlane(); plane.SetOrigin(*origin); plane.SetNormal(*normal)
 cutter=vtk.vtkCutter(); cutter.SetCutFunction(plane); cutter.SetInputData(surface); cutter.GenerateCutScalarsOn(); cutter.SetValue(0,0.0); cutter.Update()
 cleaner=vtk.vtkCleanPolyData(); cleaner.SetInputConnection(cutter.GetOutputPort()); cleaner.Update()
 if cleaner.GetOutput().GetNumberOfPoints()==0:return None
 conn=vtk.vtkPolyDataConnectivityFilter(); conn.SetInputConnection(cleaner.GetOutputPort()); conn.SetExtractionModeToClosestPointRegion(); conn.SetClosestPoint(*origin); conn.Update()
 data=conn.GetOutput(); data.BuildCells(); data.BuildLinks()
 if data.GetNumberOfCells()==0:return None
 endpoint=[]
 for pid in range(data.GetNumberOfPoints()):
  ids=vtk.vtkIdList(); data.GetPointCells(pid,ids)
  if ids.GetNumberOfIds()==1:endpoint.append(pid)
 first=endpoint[0] if endpoint else data.GetCell(0).GetPointId(0)
 sequence=[first]; first_cell=-1; point_id=first; closed=False; done=False
 while not done:
  ids=vtk.vtkIdList(); data.GetPointCells(point_id,ids); nc=ids.GetNumberOfIds()
  if nc==0:break
  chosen=-1
  for j in range(nc):
   cid=ids.GetId(j)
   if cid!=first_cell: chosen=cid; break
  if chosen<0: chosen=ids.GetId(0)
  cell=data.GetCell(chosen); cell_ids=cell.GetPointIds(); next_id=-1
  for j in range(cell_ids.GetNumberOfIds()-1):
   a=cell_ids.GetId(j); b=cell_ids.GetId(j+1)
   if a==point_id: next_id=b; break
   if b==point_id: next_id=a; break
  if next_id<0:break
  first_cell=chosen; point_id=next_id
  if point_id==first:
   closed=True; done=True
  else: sequence.append(point_id)
  if len(sequence)>data.GetNumberOfPoints()+2:break
 pts=np.array([data.GetPoint(pid) for pid in sequence])
 if len(pts)<3:return None
 return float(np.linalg.norm(pts.mean(axis=0)-origin)),pts,closed

def area2d(points,origin,normal):
 n=normal/np.linalg.norm(normal); ref=np.array([1.,0,0]) if abs(n[0])<.8 else np.array([0.,1,0]); u=ref-np.dot(ref,n)*n; u/=np.linalg.norm(u); v=np.cross(n,u); q=points-origin; xy=np.column_stack((q@u,q@v)); x=xy[:,0]; y=xy[:,1]; return .5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))

def main():
 r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(SURFACE)); r.Update(); surface=r.GetOutput(); rows=[]; all_sections=[]; counts={}
 for bid in ACTIVE:
  d=np.load(NPZ_ROOT/f'branch_{bid:02d}.npz',allow_pickle=True); points=np.asarray(d['points']); radii=np.asarray(d['MaximumInscribedSphereRadius']).reshape(-1); bcount={'valid':0,'open':0,'missing':0}
  for i,p in enumerate(points):
   t=tangent(points,i); result=cut(surface,p,t); row={'branch_id':bid,'point_id':i,'center':p.tolist(),'tangent':t.tolist(),'expected_radius':float(radii[i])}
   if result is None: row.update({'status':'MISSING','closed':False,'points':[]}); bcount['missing']+=1
   else:
    dist, contour, closed=result; a=area2d(contour,p,t); rad=np.linalg.norm(contour-p,axis=1); status='VALID' if closed and dist<max(2*radii[i],2.0) else 'OPEN_OR_AMBIGUOUS'; row.update({'status':status,'closed':bool(closed),'centroid_distance':dist,'area':a,'radius_median':float(np.median(rad)),'radius_max':float(np.max(rad)),'points':contour.tolist()}); bcount['valid' if status=='VALID' else 'open']+=1
   rows.append(row); all_sections.append(row)
  counts[str(bid)]=bcount
 report={'surface':str(SURFACE),'active_groups':ACTIVE,'counts':counts,'sections':all_sections}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'complex_vmtk_nonblanked_sections.json').write_text(json.dumps(report,indent=2))
 def make_polydata(selected, as_lines):
  pnts=vtk.vtkPoints(); cells=vtk.vtkCellArray(); br=vtk.vtkIntArray(); br.SetName('BranchId'); ar=vtk.vtkDoubleArray(); ar.SetName('SectionArea')
  for row in selected:
   if not row['points']: continue
   ids=vtk.vtkIdList()
   for p in row['points']: ids.InsertNextId(pnts.InsertNextPoint(*p))
   cells.InsertNextCell(ids); br.InsertNextValue(row['branch_id']); ar.InsertNextValue(row.get('area',0))
  poly=vtk.vtkPolyData(); poly.SetPoints(pnts); (poly.SetLines if as_lines else poly.SetPolys)(cells); poly.GetCellData().AddArray(br); poly.GetCellData().AddArray(ar); return poly
 closed_rows=[r for r in rows if r['status']=='VALID']
 open_rows=[r for r in rows if r['status']!='VALID']
 closed_out=make_polydata(closed_rows,False); open_out=make_polydata(open_rows,True)
 for poly,name in [(closed_out,'complex_vmtk_nonblanked_sections_closed.vtp'),(open_out,'complex_vmtk_nonblanked_sections_open.vtp')]:
  w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/name)); w.SetInputData(poly); w.Write()
 import pyvista as pv
 surface_pv=pv.read(SURFACE); closed_pv=pv.read(OUT/'complex_vmtk_nonblanked_sections_closed.vtp'); open_pv=pv.read(OUT/'complex_vmtk_nonblanked_sections_open.vtp')
 pl=pv.Plotter(off_screen=True,window_size=(1800,950)); pl.set_background('white'); pl.add_mesh(surface_pv,color='lightgray',opacity=.22,label='surface originale')
 if closed_pv.n_cells: pl.add_mesh(closed_pv,color='green',line_width=2,label='sections fermées valides')
 if open_pv.n_cells: pl.add_mesh(open_pv,color='red',line_width=4,label='sections ouvertes/ambiguës (lignes)')
 pl.add_legend(bcolor='white',face='rectangle'); pl.add_text('VMTK-like complex sections — non-blanked branches',font_size=15,color='black'); pl.camera_position='iso'; pl.show(screenshot=str(OUT/'complex_vmtk_nonblanked_sections.png'),auto_close=True)
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
