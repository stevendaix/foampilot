from __future__ import annotations
import argparse
from pathlib import Path
from vmtk import vmtksurfacereader, vmtkcenterlinesections, vmtksurfacewriter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--surface',type=Path,required=True); ap.add_argument('--centerlines',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--centerline-output',type=Path,required=True); args=ap.parse_args()
    sr=vmtksurfacereader.vmtkSurfaceReader(); sr.InputFileName=str(args.surface); sr.Format='stl'; sr.Execute()
    cr=vmtksurfacereader.vmtkSurfaceReader(); cr.InputFileName=str(args.centerlines); cr.Format='vtkxml'; cr.Execute()
    cs=vmtkcenterlinesections.vmtkCenterlineSections(); cs.Surface=sr.Surface; cs.Centerlines=cr.Surface; cs.Execute()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.centerline_output.parent.mkdir(parents=True,exist_ok=True)
    sw=vmtksurfacewriter.vmtkSurfaceWriter(); sw.Surface=cs.CenterlineSections; sw.OutputFileName=str(args.output); sw.Format='vtkxml'; sw.Execute()
    cw=vmtksurfacewriter.vmtkSurfaceWriter(); cw.Surface=cs.Centerlines; cw.OutputFileName=str(args.centerline_output); cw.Format='vtkxml'; cw.Execute()
    print({'input_surface_points':sr.Surface.GetNumberOfPoints(),'input_surface_cells':sr.Surface.GetNumberOfCells(),'input_centerline_points':cr.Surface.GetNumberOfPoints(),'input_centerline_cells':cr.Surface.GetNumberOfCells(),'section_points':cs.CenterlineSections.GetNumberOfPoints(),'section_cells':cs.CenterlineSections.GetNumberOfCells(),'output':str(args.output)})
if __name__=='__main__': main()
