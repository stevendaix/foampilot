from pathlib import Path
import json, numpy as np
import build123d as bd
import pyvista as pv

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'examples/medical_build/outputs'
DATA=OUT/'complex_vmtk_nonblanked_sections.json'

def basis(n):
 n=np.asarray(n,float); n/=max(np.linalg.norm(n),1e-12); ref=np.array([1.,0,0]) if abs(n[0])<.8 else np.array([0.,1,0]); u=ref-n*np.dot(ref,n); u/=max(np.linalg.norm(u),1e-12); return u,np.cross(n,u)
def normalize_profile(points,center,tangent,N=48):
 p=np.asarray(points,float); c=np.asarray(center,float); t=np.asarray(tangent,float); t/=max(np.linalg.norm(t),1e-12); p=p-((p-c)@t)[:,None]*t[None,:]
 u,v=basis(t); q=p-c; ang=np.arctan2(q@v,q@u); rad=np.linalg.norm(q,axis=1); order=np.argsort(ang); ang=ang[order]; rad=rad[order]; p=p[order]
 # perimeter resampling using ordered contour; close parameter
 p=np.vstack([p,p[0]]); seg=np.linalg.norm(np.diff(p,axis=0),axis=1); s=np.r_[0,np.cumsum(seg)]; total=s[-1]
 if total<=1e-10: raise ValueError('degenerate contour')
 target=np.linspace(0,total,N,endpoint=False); out=[]
 for z in target:
  j=min(np.searchsorted(s,z,side='right')-1,len(seg)-1); a=(z-s[j])/max(seg[j],1e-12); out.append(p[j]*(1-a)+p[j+1]*a)
 return np.asarray(out)

def main():
 report=json.loads(DATA.read_text()); groups={}
 for r in report['sections']:
  if r.get('status')=='VALID' and r.get('closed') and len(r.get('points',[]))>=3: groups.setdefault(int(r['branch_id']),[]).append(r)
 solids=[]; stats={}
 for bid,rows in sorted(groups.items()):
  rows=sorted(rows,key=lambda r:r['point_id']); wires=[]
  for r in rows:
   pts=normalize_profile(r['points'],r['center'],r['tangent'],48); vs=[bd.Vector(*map(float,p)) for p in pts]; edges=[bd.Edge.make_line(vs[i],vs[(i+1)%len(vs)]) for i in range(len(vs))]; wires.append(bd.Wire(edges))
  if len(wires)>=2:
   solid=bd.Solid.make_loft(wires,ruled=False); solids.append(solid); stats[str(bid)]={'sections':len(wires),'valid':len(rows)}
 if not solids: raise RuntimeError('no valid branch loft')
 compound=bd.Compound.make_composite(solids); stl=OUT/'complex_vmtk_valid_branches_compound.stl'; step=OUT/'complex_vmtk_valid_branches_compound.step'; bd.export_stl(compound,str(stl)); bd.export_step(compound,str(step))
 mesh=pv.read(stl); ref=pv.read(str(Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/aorta_surface_patches.vtp'))
 pl=pv.Plotter(off_screen=True,window_size=(1800,950)); pl.set_background('white'); pl.add_mesh(ref,color='lightgray',opacity=.25,label='surface VMTK originale'); pl.add_mesh(mesh,color='royalblue',opacity=.9,label='lofts profils VMTK valides'); pl.add_legend(bcolor='white',face='rectangle'); pl.add_text('Reconstruction contrôlée — profils fermés, Blanking-aware',font_size=15,color='black'); pl.camera_position='iso'; pl.show(screenshot=str(OUT/'complex_vmtk_valid_branches_before_after.png'),auto_close=True)
 (OUT/'complex_vmtk_valid_branches_report.json').write_text(json.dumps({'input':str(DATA),'branches':stats,'stl':str(stl),'step':str(step),'note':'diagnostic compound: junction blanked not reconstructed'},indent=2)); print(json.dumps(stats,indent=2))
if __name__=='__main__': main()
