import pyvista as pv
p=str(Path(__file__).resolve().parents[2] / 'case_complex/openfoam_surface_patches/aorta_surface_patches.vtp')
m=pv.read(p)
print(type(m).__name__, m.n_points, m.n_cells)
print('point_data', list(m.point_data.keys()))
print('cell_data', list(m.cell_data.keys()))
for k in m.cell_data:
 a=m.cell_data[k]
 print(k, a.dtype, a.shape, sorted(set(a.tolist()))[:30] if a.size < 100000 else 'large')
