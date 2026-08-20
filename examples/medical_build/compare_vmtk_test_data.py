from __future__ import annotations
import json,sys
from pathlib import Path

def info(path):
 import vtk
 p=Path(path); out={'path':str(p),'bytes':p.stat().st_size}
 if p.suffix=='.vtp': r=vtk.vtkXMLPolyDataReader()
 elif p.suffix=='.stl': r=vtk.vtkSTLReader()
 else: return out
 r.SetFileName(str(p)); r.Update(); d=r.GetOutput(); out.update({'points':d.GetNumberOfPoints(),'cells':d.GetNumberOfCells(),'bounds':list(d.GetBounds()),'point_arrays':[d.GetPointData().GetArrayName(i) for i in range(d.GetPointData().GetNumberOfArrays())],'cell_arrays':[d.GetCellData().GetArrayName(i) for i in range(d.GetCellData().GetNumberOfArrays())]}); return out

def main():
 paths=[Path(x) for x in sys.argv[1:]]; result=[info(p) for p in paths]; print(json.dumps(result,indent=2)); Path('/tmp/vmtk_aorta_data_comparison.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
