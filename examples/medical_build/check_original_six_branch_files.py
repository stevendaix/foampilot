from pathlib import Path
import vtk
import numpy as np

root = Path(__file__).resolve().parents[2] / 'foampilot/test/vmtk_test_data'
for path in sorted(root.glob('aorta-centerline*.vtp')):
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); d = r.GetOutput()
    print(f'\n{path.name}: points={d.GetNumberOfPoints()} cells={d.GetNumberOfCells()}')
    for ci in range(d.GetNumberOfCells()):
        cell=d.GetCell(ci); ids=[cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        start=np.asarray(d.GetPoint(ids[0])); end=np.asarray(d.GetPoint(ids[-1]))
        length=sum(np.linalg.norm(np.asarray(d.GetPoint(ids[j]))-np.asarray(d.GetPoint(ids[j-1]))) for j in range(1,len(ids)))
        print(f'  cell {ci}: n={len(ids)} length={length:.3f} start={start.round(2).tolist()} end={end.round(2).tolist()}')
    print(' point arrays:', [d.GetPointData().GetArray(i).GetName() for i in range(d.GetPointData().GetNumberOfArrays())])
