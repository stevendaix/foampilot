from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('stl',type=Path); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); import vtk
 reader=vtk.vtkSTLReader(); reader.SetFileName(str(args.stl)); reader.Update(); mesh=reader.GetOutput(); conn=vtk.vtkPolyDataConnectivityFilter(); conn.SetInputData(mesh); conn.SetExtractionModeToAllRegions(); conn.ColorRegionsOn(); conn.Update(); regions=conn.GetNumberOfExtractedRegions(); clean=vtk.vtkCleanPolyData(); clean.SetInputData(mesh); clean.PointMergingOn(); clean.SetTolerance(1e-5); clean.Update(); clean_mesh=clean.GetOutput(); boundary=vtk.vtkFeatureEdges(); boundary.SetInputData(clean_mesh); boundary.BoundaryEdgesOn(); boundary.FeatureEdgesOff(); boundary.NonManifoldEdgesOff(); boundary.Update(); nonmanifold=vtk.vtkFeatureEdges(); nonmanifold.SetInputData(clean_mesh); nonmanifold.BoundaryEdgesOff(); nonmanifold.FeatureEdgesOff(); nonmanifold.NonManifoldEdgesOn(); nonmanifold.Update(); report={'path':str(args.stl),'points':clean_mesh.GetNumberOfPoints(),'cells':clean_mesh.GetNumberOfCells(),'components':regions,'boundary_edges':boundary.GetOutput().GetNumberOfCells(),'nonmanifold_edges':nonmanifold.GetOutput().GetNumberOfCells()}; args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
