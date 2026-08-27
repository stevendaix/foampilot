from __future__ import annotations
import argparse
from pathlib import Path
from vmtk import pypes, vmtksurfacereader, vmtksurfacewriter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('output',type=Path); args=ap.parse_args()
    reader=vmtksurfacereader.vmtkSurfaceReader(); reader.InputFileName=str(args.input); reader.Format='stl'; reader.Execute()
    if reader.Surface is None or reader.Surface.GetNumberOfPoints()==0: raise RuntimeError('VMTK reader produced an empty surface')
    writer=vmtksurfacewriter.vmtkSurfaceWriter(); writer.Surface=reader.Surface; writer.OutputFileName=str(args.output); writer.Format='stl'; writer.Execute()
    print({'input':str(args.input),'output':str(args.output),'points':reader.Surface.GetNumberOfPoints(),'cells':reader.Surface.GetNumberOfCells()})
if __name__=='__main__': main()
