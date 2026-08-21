from __future__ import annotations
import argparse,json
from pathlib import Path
import trimesh

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('patch_dir',type=Path); ap.add_argument('--closed-reference',type=Path,required=True); ap.add_argument('--location',nargs=3,type=float,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    out={'patches':{},'location':list(map(float,args.location))}
    for p in sorted(args.patch_dir.glob('*.stl')):
        m=trimesh.load_mesh(p,process=False); out['patches'][p.name]={'vertices':len(m.vertices),'faces':len(m.faces),'watertight':bool(m.is_watertight),'components':len(m.split(only_watertight=False)),'boundary_edges':int(len(m.edges_boundary)) if hasattr(m,'edges_boundary') else None,'area':float(m.area),'volume':float(abs(m.volume))}
    ref=trimesh.load_mesh(args.closed_reference,process=False)\n    import vtk\n    pts=vtk.vtkPoints(); pts.InsertNextPoint(*map(float,args.location)); pd=vtk.vtkPolyData(); pd.SetPoints(pts)\n    rr=vtk.vtkSTLReader(); rr.SetFileName(str(args.closed_reference)); rr.Update(); sel=vtk.vtkSelectEnclosedPoints(); sel.SetInputData(pd); sel.SetSurfaceData(rr.GetOutput()); sel.CheckSurfaceOn(); sel.Update()\n    out['location_inside_closed_reference']=bool(sel.IsInside(0)); args.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
