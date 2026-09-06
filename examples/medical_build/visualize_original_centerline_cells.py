from pathlib import Path
import pyvista as pv

root=Path(__file__).resolve().parents[2]
path=root/'foampilot/test/vmtk_test_data/aorta-centerline-branches.vtp'
cl=pv.read(path)
plotter=pv.Plotter(off_screen=True, window_size=(1600,1000))
plotter.set_background('white')
colors=['red','orange','yellow','green','blue','purple']
for i in range(cl.n_cells):
    part=cl.extract_cells([i])
    plotter.add_mesh(part, color=colors[i], line_width=8, render_lines_as_tubes=True, label=f'cell {i}')
    pts=part.points
    if len(pts):
        plotter.add_point_labels(pts[[0,-1]], [f'{i}:start', f'{i}:end'], font_size=12, point_size=8, shape_color='white', text_color='black')
plotter.add_legend(bcolor='white', face='rectangle')
plotter.add_text('Original VMTK aorta-centerline-branches.vtp', font_size=16, color='black')
plotter.camera_position='iso'
plotter.show(screenshot=str(root/'examples/medical_build/outputs/original_centerline_cells.png'), auto_close=True)
