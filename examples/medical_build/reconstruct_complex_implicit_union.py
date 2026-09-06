from pathlib import Path
import json
import vtk
import pyvista as pv

ROOT=Path(__file__).resolve().parents[2]
REF=Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/aorta_surface_patches.vtp'
BRANCH_DIR=ROOT/'examples/medical_build/case_complex/exports_complex/manual_stl'
OUT=ROOT/'examples/medical_build/outputs'
SPACING=0.75

def read(path):
 r=vtk.vtkXMLPolyDataReader() if path.suffix=='.vtp' else vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update(); return r.GetOutput()

def stats(p):
 tri=vtk.vtkTriangleFilter(); tri.SetInputData(p); tri.Update(); clean=vtk.vtkCleanPolyData(); clean.SetInputConnection(tri.GetOutputPort()); clean.Update(); q=clean.GetOutput(); m=vtk.vtkMassProperties(); m.SetInputData(q); m.Update(); return {'points':q.GetNumberOfPoints(),'cells':q.GetNumberOfCells(),'bounds':list(q.GetBounds()),'area':m.GetSurfaceArea(),'volume':m.GetVolume()}

def main():
 ref=read(REF); rb=ref.GetBounds(); margin=3.0
 implicit=vtk.vtkImplicitBoolean(); implicit.SetOperationTypeToUnion()
 files=sorted(BRANCH_DIR.glob('branch_*.stl'))
 for f in files:
  poly=read(f); dist=vtk.vtkImplicitPolyDataDistance(); dist.SetInput(poly); implicit.AddFunction(dist)
 sample=vtk.vtkSampleFunction(); sample.SetImplicitFunction(implicit); sample.SetModelBounds(rb[0]-margin,rb[1]+margin,rb[2]-margin,rb[3]+margin,rb[4]-margin,rb[5]+margin); sample.SetSampleDimensions(max(2,int((rb[1]-rb[0]+2*margin)/SPACING)),max(2,int((rb[3]-rb[2]+2*margin)/SPACING)),max(2,int((rb[5]-rb[4]+2*margin)/SPACING))); sample.ComputeNormalsOff(); sample.Update()
 mc=vtk.vtkMarchingCubes(); mc.SetInputConnection(sample.GetOutputPort()); mc.SetValue(0,0.0); mc.Update(); smooth=vtk.vtkWindowedSincPolyDataFilter(); smooth.SetInputConnection(mc.GetOutputPort()); smooth.SetNumberOfIterations(10); smooth.SetPassBand(0.1); smooth.BoundarySmoothingOff(); smooth.FeatureEdgeSmoothingOff(); smooth.Update(); out=smooth.GetOutput()
 st={'reference':stats(ref),'improved_union':stats(out),'spacing':SPACING,'files':[str(f) for f in files]}; st['relative_volume_error']=st['improved_union']['volume']/st['reference']['volume']-1
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'complex_implicit_union_comparison.json').write_text(json.dumps(st,indent=2))
 w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/'complex_implicit_union.vtp')); w.SetInputData(out); w.Write(); sw=vtk.vtkSTLWriter(); sw.SetFileName(str(OUT/'complex_implicit_union.stl')); sw.SetInputData(out); sw.Write()
 pl=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1800,850));
 for i,(mesh,title,color) in enumerate([(pv.read(REF),'Reference surface','lightgray'),(pv.read(OUT/'complex_implicit_union.vtp'),'Improved implicit union','tomato')]):
  pl.subplot(0,i); pl.set_background('white'); pl.add_mesh(mesh,color=color,opacity=.7); pl.add_text(title,font_size=15,color='black'); pl.camera_position='iso'
 pl.link_views(); pl.show(screenshot=str(OUT/'complex_implicit_union_before_after.png'),auto_close=True)
 print(json.dumps({'reference_volume':st['reference']['volume'],'improved_volume':st['improved_union']['volume'],'relative_volume_error':st['relative_volume_error'],'improved_points':st['improved_union']['points'],'improved_cells':st['improved_union']['cells']},indent=2))
if __name__=='__main__': main()
