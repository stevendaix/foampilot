from __future__ import annotations
import argparse
from pathlib import Path
import vtk

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('file',type=Path); args=ap.parse_args()
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(args.file)); r.Update(); p=r.GetOutput()
    print({'file':str(args.file),'points':p.GetNumberOfPoints(),'cells':p.GetNumberOfCells(),'verts':p.GetNumberOfVerts(),'lines':p.GetNumberOfLines(),'polys':p.GetNumberOfPolys()})
    for assoc,name in [(p.GetPointData(),'point'),(p.GetCellData(),'cell')]:
        print(name, [(assoc.GetArray(i).GetName(), assoc.GetArray(i).GetNumberOfTuples(), assoc.GetArray(i).GetDataTypeAsString()) for i in range(assoc.GetNumberOfArrays())])
if __name__=='__main__': main()
