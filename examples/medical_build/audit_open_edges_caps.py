from pathlib import Path
import json, numpy as np, vtk, pyvista as pv
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'examples/medical_build/outputs'; STL=OUT/'complex_vmtk_reference_clean.stl'; MAN=Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/patch_manifest.json'
def main():
 m=pv.read(STL).triangulate().clean(); feat=vtk.vtkFeatureEdges(); feat.SetInputData(m); feat.BoundaryEdgesOn(); feat.FeatureEdgesOff(); feat.NonManifoldEdgesOff(); feat.ManifoldEdgesOff(); feat.Update(); e=pv.wrap(feat.GetOutput()); rows=[]
 caps=json.loads(MAN.read_text())['classification']; centers=np.array([c['loop_center'] for c in caps],float)
 for i in range(e.n_cells):
  p=np.asarray(e.extract_cells([i]).points); mid=p.mean(axis=0); d=np.linalg.norm(centers-mid,axis=1); rows.append({'edge_id':i,'n_points':len(p),'midpoint':mid.tolist(),'nearest_cap':int(np.argmin(d)),'distance_to_cap':float(d.min())})
 result={'stl':str(STL),'open_edge_cells':e.n_cells,'edges':rows,'bounds':list(map(float,m.bounds)),'volume':float(m.volume),'area':float(m.area)}; (OUT/'complex_open_edges_audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
