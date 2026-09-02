from __future__ import annotations
import argparse
from pathlib import Path
import vtk
from vmtk import vtkvmtk

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('output',type=Path); args=ap.parse_args()
    r=vtk.vtkSTLReader(); r.SetFileName(str(args.input)); r.Update()
    cap=vtkvmtk.vtkvmtkCapPolyData(); cap.SetInputData(r.GetOutput()); cap.SetCellEntityIdsArrayName('CapIds'); cap.Update()
    out=vtk.vtkXMLPolyDataWriter(); out.SetFileName(str(args.output)); out.SetInputData(cap.GetOutput()); out.Write()
    arr=cap.GetOutput().GetCellData().GetArray('CapIds'); hist={}
    if arr:
        for i in range(arr.GetNumberOfTuples()): hist[str(int(arr.GetTuple1(i)))]=hist.get(str(int(arr.GetTuple1(i))),0)+1
    print({'points':cap.GetOutput().GetNumberOfPoints(),'cells':cap.GetOutput().GetNumberOfCells(),'cap_id_histogram':hist})
if __name__=='__main__': main()
