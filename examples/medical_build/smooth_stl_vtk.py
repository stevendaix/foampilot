from __future__ import annotations
import argparse
from pathlib import Path
import vtk

def read(path):
    r=vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update(); return r.GetOutput()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('output',type=Path); ap.add_argument('--iterations',type=int,default=15); ap.add_argument('--passband',type=float,default=.1); args=ap.parse_args()
    tri=vtk.vtkTriangleFilter(); tri.SetInputData(read(args.input)); tri.Update(); clean=vtk.vtkCleanPolyData(); clean.SetInputConnection(tri.GetOutputPort()); clean.Update()
    smooth=vtk.vtkWindowedSincPolyDataFilter(); smooth.SetInputConnection(clean.GetOutputPort()); smooth.SetNumberOfIterations(args.iterations); smooth.SetPassBand(args.passband); smooth.FeatureEdgeSmoothingOff(); smooth.BoundarySmoothingOff(); smooth.NonManifoldSmoothingOff(); smooth.NormalizeCoordinatesOn(); smooth.Update()
    out=vtk.vtkSTLWriter(); out.SetFileName(str(args.output)); out.SetInputConnection(smooth.GetOutputPort()); out.Write(); print({'input_points':clean.GetOutput().GetNumberOfPoints(),'output_points':smooth.GetOutput().GetNumberOfPoints(),'output_cells':smooth.GetOutput().GetNumberOfCells()})
if __name__=='__main__': main()
