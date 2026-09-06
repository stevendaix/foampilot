from __future__ import annotations
import argparse
from pathlib import Path
import vtk

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path',type=Path); args=ap.parse_args()
    r=vtk.vtkSTLReader(); r.SetFileName(str(args.path)); r.Update(); p=r.GetOutput()
    e=vtk.vtkFeatureEdges(); e.SetInputData(p); e.BoundaryEdgesOn(); e.FeatureEdgesOff(); e.NonManifoldEdgesOff(); e.ManifoldEdgesOff(); e.Update()
    conn=vtk.vtkPolyDataConnectivityFilter(); conn.SetInputData(e.GetOutput()); conn.SetExtractionModeToAllRegions(); conn.Update()
    print({'points':p.GetNumberOfPoints(),'cells':p.GetNumberOfCells(),'boundary_edge_cells':e.GetOutput().GetNumberOfCells(),'boundary_components':conn.GetNumberOfExtractedRegions(),'bounds':p.GetBounds()})
if __name__=='__main__': main()
