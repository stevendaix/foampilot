from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import trimesh

def edge_report(faces):
    counts={}
    for tri in faces:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            key=tuple(sorted((int(a),int(b))))
            counts[key]=counts.get(key,0)+1
    vals=list(counts.values())
    return {"edges":len(vals),"boundary":sum(v==1 for v in vals),"nonmanifold":sum(v>2 for v in vals),"histogram":{str(k):vals.count(k) for k in sorted(set(vals))}}

def main():
    path=Path(sys.argv[1]); raw=trimesh.load_mesh(path,process=False); merged=trimesh.load_mesh(path,process=True)
    report={"file":str(path),"raw":{"vertices":len(raw.vertices),"faces":len(raw.faces),"watertight":bool(raw.is_watertight)},"merged":{"vertices":len(merged.vertices),"faces":len(merged.faces),"watertight":bool(merged.is_watertight),"winding_consistent":bool(merged.is_winding_consistent),"volume":float(merged.volume),"euler_number":int(merged.euler_number),"edge_report":edge_report(merged.faces)}}
    try:
        import vtk
        reader=vtk.vtkSTLReader(); reader.SetFileName(str(path)); reader.Update(); poly=reader.GetOutput(); clean=vtk.vtkCleanPolyData(); clean.SetInputData(poly); clean.Update(); boundary=vtk.vtkFeatureEdges(); boundary.SetInputData(clean.GetOutput()); boundary.BoundaryEdgesOn(); boundary.FeatureEdgesOff(); boundary.NonManifoldEdgesOff(); boundary.Update(); nonman=vtk.vtkFeatureEdges(); nonman.SetInputData(clean.GetOutput()); nonman.BoundaryEdgesOff(); nonman.FeatureEdgesOff(); nonman.NonManifoldEdgesOn(); nonman.Update(); report["vtk"]={"points":clean.GetOutput().GetNumberOfPoints(),"cells":clean.GetOutput().GetNumberOfCells(),"boundary_edges":boundary.GetOutput().GetNumberOfCells(),"nonmanifold_edges":nonman.GetOutput().GetNumberOfCells()}
    except ImportError: report["vtk"]="unavailable"
    print(json.dumps(report,indent=2));
    if report["merged"]["watertight"] is not True or report["merged"]["edge_report"]["boundary"] != 0 or report["merged"]["edge_report"]["nonmanifold"] != 0: raise SystemExit(2)
if __name__=="__main__": main()
