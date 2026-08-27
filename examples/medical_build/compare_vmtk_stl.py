from __future__ import annotations
import argparse,json
from pathlib import Path
import vtk
from vmtk import vtkvmtk

def read_polydata(path):
    path=Path(path); ext=path.suffix.lower()
    if ext=='.stl': r=vtk.vtkSTLReader()
    elif ext=='.vtp': r=vtk.vtkXMLPolyDataReader()
    else: raise ValueError(f'unsupported extension: {ext}')
    r.SetFileName(str(path)); r.Update(); return r.GetOutput()

def stats(poly):
    tri=vtk.vtkTriangleFilter(); tri.SetInputData(poly); tri.Update(); clean=vtk.vtkCleanPolyData(); clean.SetInputConnection(tri.GetOutputPort()); clean.Update(); p=clean.GetOutput()
    mass=vtk.vtkMassProperties(); mass.SetInputData(p); mass.Update()
    feature=vtk.vtkFeatureEdges(); feature.SetInputData(p); feature.BoundaryEdgesOn(); feature.FeatureEdgesOff(); feature.NonManifoldEdgesOff(); feature.ManifoldEdgesOff(); feature.Update()
    conn=vtk.vtkPolyDataConnectivityFilter(); conn.SetInputData(p); conn.SetExtractionModeToAllRegions(); conn.Update()
    return {'points':p.GetNumberOfPoints(),'cells':p.GetNumberOfCells(),'area':mass.GetSurfaceArea(),'volume':mass.GetVolume(),'boundary_edge_cells':feature.GetOutput().GetNumberOfCells(),'components':conn.GetNumberOfExtractedRegions(),'bounds':p.GetBounds()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('reference',type=Path); ap.add_argument('candidate',type=Path); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    ref=stats(read_polydata(args.reference)); cand=stats(read_polydata(args.candidate))
    result={'reference':{'file':str(args.reference),**ref},'candidate':{'file':str(args.candidate),**cand},'relative_volume_error':cand['volume']/ref['volume']-1 if ref['volume'] else None,'relative_area_error':cand['area']/ref['area']-1 if ref['area'] else None,'vtk_version':vtk.vtkVersion.GetVTKVersion(),'vmtk_module':vtkvmtk.__file__}
    args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
