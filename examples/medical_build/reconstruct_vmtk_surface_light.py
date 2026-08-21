from pathlib import Path
import json, numpy as np, vtk, pyvista as pv
ROOT=Path('/home/ubuntu/foampilot_pr_repo'); OUT=ROOT/'examples/medical_build/outputs'; DATA=OUT/'complex_vmtk_nonblanked_sections.json'
def basis(n):
 n=np.asarray(n,float); n/=max(np.linalg.norm(n),1e-12); ref=np.array([1.,0,0]) if abs(n[0])<.8 else np.array([0.,1,0]); u=ref-n*np.dot(ref,n); u/=max(np.linalg.norm(u),1e-12); return u,np.cross(n,u)
def ring(row,N=24):
 p=np.asarray(row['points'],float); c=np.asarray(row['center'],float); t=np.asarray(row['tangent'],float); t/=max(np.linalg.norm(t),1e-12); p=p-((p-c)@t)[:,None]*t[None,:]; u,v=basis(t); a=np.arctan2((p-c)@v,(p-c)@u); order=np.argsort(a); p=p[order]; p=np.vstack([p,p[0]]); s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]; target=np.linspace(0,s[-1],N,endpoint=False); out=[]
 for z in target:
  j=min(np.searchsorted(s,z,side='right')-1,len(s)-2); q=(z-s[j])/max(s[j+1]-s[j],1e-12); out.append(p[j]*(1-q)+p[j+1]*q)
 return np.asarray(out)
def main():
 data=json.loads(DATA.read_text()); groups={}
 for r in data['sections']:
  if r['status']=='VALID' and r['closed']: groups.setdefault(int(r['branch_id']),[]).append(r)
 pts=vtk.vtkPoints(); polys=vtk.vtkCellArray(); stats={}
 for bid,rows in sorted(groups.items()):
  rows=sorted(rows,key=lambda x:x['point_id'])[::4]
  if len(rows)<2:continue
  rings=[ring(r) for r in rows]; base=[]
  for rr in rings: base.append([pts.InsertNextPoint(*p) for p in rr])
  for j in range(len(base)-1):
   for i in range(24):
    q=vtk.vtkIdList(); q.InsertNextId(base[j][i]); q.InsertNextId(base[j][(i+1)%24]); q.InsertNextId(base[j+1][(i+1)%24]); q.InsertNextId(base[j+1][i]); polys.InsertNextCell(q)
  stats[str(bid)]={'input_sections':len(groups[bid]),'used_sections':len(rows),'rings':24}
 out=vtk.vtkPolyData(); out.SetPoints(pts); out.SetPolys(polys); clean=vtk.vtkCleanPolyData(); clean.SetInputData(out); clean.Update(); tri=vtk.vtkTriangleFilter(); tri.SetInputConnection(clean.GetOutputPort()); tri.Update(); surf=tri.GetOutput()
 vtp=OUT/'complex_vmtk_light_branch_surface.vtp'; stl=OUT/'complex_vmtk_light_branch_surface.stl'; w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(vtp)); w.SetInputData(surf); w.Write(); sw=vtk.vtkSTLWriter(); sw.SetFileName(str(stl)); sw.SetInputData(surf); sw.Write()
 mesh=pv.read(stl); ref=pv.read('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package/openfoam_surface_patches/aorta_surface_patches.vtp'); pl=pv.Plotter(off_screen=True,window_size=(1800,950)); pl.set_background('white'); pl.add_mesh(ref,color='lightgray',opacity=.22,label='surface VMTK originale'); pl.add_mesh(mesh,color='royalblue',opacity=.88,label='surface reconstruite légère'); pl.add_legend(bcolor='white',face='rectangle'); pl.add_text('Reconstruction légère — sections fermées VMTK, Blanking-aware',font_size=15,color='black'); pl.camera_position='iso'; pl.show(screenshot=str(OUT/'complex_vmtk_light_branch_surface.png'),auto_close=True)
 (OUT/'complex_vmtk_light_branch_surface_report.json').write_text(json.dumps({'method':'ring-to-ring surface stitching','stride':4,'rings':24,'branches':stats,'note':'diagnostic surface; central bifurcation not reconstructed'},indent=2)); print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
