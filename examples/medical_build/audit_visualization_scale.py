from pathlib import Path
import json
import vtk


def read(path):
    r = vtk.vtkXMLPolyDataReader() if path.suffix == '.vtp' else vtk.vtkSTLReader()
    r.SetFileName(str(path)); r.Update(); return r.GetOutput()


def stats(path):
    p = read(path)
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(p); tri.Update()
    mass = vtk.vtkMassProperties(); mass.SetInputData(tri.GetOutput()); mass.Update()
    b = p.GetBounds()
    return {'file': str(path), 'points': p.GetNumberOfPoints(), 'cells': p.GetNumberOfCells(), 'bounds': list(b), 'dimensions': [b[1]-b[0], b[3]-b[2], b[5]-b[4]], 'area': mass.GetSurfaceArea(), 'volume': mass.GetVolume()}

root = Path('/home/ubuntu/foampilot_pr_repo')
paths = [
 root/'foampilot/test/vmtk_test_data/aorta-surface.vtp',
 root/'examples/medical_build/outputs/vmtk_like_polyball_aorta.vtp',
 root/'examples/medical_build/outputs/vmtk_like_aorta_sections_filtered.vtp',
 root/'examples/medical_build/outputs/reconstructed_local_deformation/reference/aorta_sections_combined.stl',
]
result = {'surfaces': [stats(p) for p in paths if p.exists()]}
out = root/'examples/medical_build/outputs/visualization_scale_audit.json'
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
