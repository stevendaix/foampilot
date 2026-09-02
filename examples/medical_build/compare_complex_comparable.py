from pathlib import Path
import json
import vtk
import pyvista as pv

ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/str(Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/aorta_surface_patches.vtp')
# Path is absolute after ROOT / correction below.
REF=Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/aorta_surface_patches.vtp'
BRANCH_DIR=ROOT/'examples/medical_build/case_complex/exports_complex/manual_stl'
OUT=ROOT/'examples/medical_build/outputs'

def read(path):
    if path.suffix=='.vtp': r=vtk.vtkXMLPolyDataReader()
    else: r=vtk.vtkSTLReader()
    r.SetFileName(str(path)); r.Update(); return r.GetOutput()

def clean(poly):
    tri=vtk.vtkTriangleFilter(); tri.SetInputData(poly); tri.Update()
    cl=vtk.vtkCleanPolyData(); cl.SetInputConnection(tri.GetOutputPort()); cl.Update(); return cl.GetOutput()

def stats(poly):
    p=clean(poly); m=vtk.vtkMassProperties(); m.SetInputData(p); m.Update()
    f=vtk.vtkFeatureEdges(); f.SetInputData(p); f.BoundaryEdgesOn(); f.NonManifoldEdgesOn(); f.FeatureEdgesOff(); f.ManifoldEdgesOff(); f.Update()
    c=vtk.vtkPolyDataConnectivityFilter(); c.SetInputData(p); c.SetExtractionModeToAllRegions(); c.Update()
    return {'points':p.GetNumberOfPoints(),'cells':p.GetNumberOfCells(),'bounds':list(p.GetBounds()),'dimensions':[p.GetBounds()[1]-p.GetBounds()[0],p.GetBounds()[3]-p.GetBounds()[2],p.GetBounds()[5]-p.GetBounds()[4]],'area':m.GetSurfaceArea(),'volume':m.GetVolume(),'boundary_or_nonmanifold_edges':f.GetOutput().GetNumberOfCells(),'components':c.GetNumberOfExtractedRegions()}

def main():
    ref=read(REF); branches=[]; append=vtk.vtkAppendPolyData()
    for path in sorted(BRANCH_DIR.glob('branch_*.stl')):
        p=read(path); branches.append({'file':str(path),'stats':stats(p)}); append.AddInputData(p)
    append.Update(); candidate=append.GetOutput()
    result={'reference':{'file':str(REF),**stats(ref)},'candidate_append':{'files':[b['file'] for b in branches],**stats(candidate)},'branches':branches}
    result['relative_volume_error']=result['candidate_append']['volume']/result['reference']['volume']-1 if result['reference']['volume'] else None
    result['relative_area_error']=result['candidate_append']['area']/result['reference']['area']-1 if result['reference']['area'] else None
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'complex_comparable_comparison.json').write_text(json.dumps(result,indent=2))
    refpv=pv.read(REF); candpv=pv.MultiBlock([pv.read(p['file']) for p in branches])
    pl=pv.Plotter(off_screen=True,window_size=(1600,1000)); pl.set_background('white')
    pl.add_mesh(refpv,color='lightgray',opacity=.25,label='reference patches')
    pl.add_mesh(candpv,color='red',opacity=.35,show_edges=False,label='manual branches')
    pl.add_legend(bcolor='white',face='rectangle'); pl.add_text('Comparable complex case: same coordinates and same eight branch files',font_size=14,color='black'); pl.camera_position='iso'; pl.show(screenshot=str(OUT/'complex_comparable_overlay.png'),auto_close=True)
    print(json.dumps({'reference_volume':result['reference']['volume'],'candidate_volume':result['candidate_append']['volume'],'relative_volume_error':result['relative_volume_error'],'reference_components':result['reference']['components'],'candidate_components':result['candidate_append']['components']},indent=2))

if __name__=='__main__': main()
