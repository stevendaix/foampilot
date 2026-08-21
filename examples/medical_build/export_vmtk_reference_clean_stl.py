from pathlib import Path
import json, vtk, pyvista as pv
ROOT=Path('/home/ubuntu/foampilot_pr_repo'); OUT=ROOT/'examples/medical_build/outputs'
SRC=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package/openfoam_surface_patches/aorta_surface_patches.vtp')
def main():
 src=pv.read(SRC).extract_surface().triangulate().clean(tolerance=1e-8)
 poly=src.cast_to_unstructured_grid().extract_surface().triangulate().clean()
 v=vtk.vtkPolyData(); v.DeepCopy(poly)
 normals=vtk.vtkPolyDataNormals(); normals.SetInputData(v); normals.ConsistencyOn(); normals.AutoOrientNormalsOn(); normals.SplittingOff(); normals.Update()
 clean=vtk.vtkCleanPolyData(); clean.SetInputConnection(normals.GetOutputPort()); clean.Update()
 surf=clean.GetOutput(); vtp=OUT/'complex_vmtk_reference_clean.stl.vtp'; stl=OUT/'complex_vmtk_reference_clean.stl'; w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(vtp)); w.SetInputData(surf); w.Write(); sw=vtk.vtkSTLWriter(); sw.SetFileName(str(stl)); sw.SetInputData(surf); sw.SetFileTypeToBinary(); sw.Write()
 mesh=pv.read(stl); plot=pv.Plotter(off_screen=True,window_size=(1800,950)); plot.set_background('white'); plot.add_mesh(mesh,color='royalblue',opacity=.95,label='STL VMTK nettoyé'); plot.add_text('STL final — surface VMTK cappée, 8 sorties',font_size=16,color='black'); plot.add_legend(bcolor='white',face='rectangle'); plot.camera_position='iso'; plot.show(screenshot=str(OUT/'complex_vmtk_reference_clean.png'),auto_close=True)
 result={'input':str(SRC),'output_stl':str(stl),'output_vtp':str(vtp),'points':int(mesh.n_points),'cells':int(mesh.n_cells),'bounds':list(map(float,mesh.bounds)),'volume':float(mesh.volume),'area':float(mesh.area),'n_open_edges':int(mesh.n_open_edges),'n_connected_components':int(mesh.connectivity().cell_data['RegionId'].max()+1) if mesh.n_cells else 0}
 (OUT/'complex_vmtk_reference_clean_report.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
