#!/usr/bin/env python
"""
Post-processing script for building_aero simulation.
This script is independent and can be used to debug visualization issues.
"""

from pathlib import Path
import numpy as np
import pyvista as pv

from foampilot import postprocess, utilities

# Set off-screen rendering with white background
pv.set_plot_theme("document")
pv.global_theme.background = 'white'
pv.global_theme.color = 'black'

# Define the working directory
current_path = Path.cwd() / 'quartier_gmsh'

# Analysis of residuals
log_file = current_path / "log.incompressibleFluid"
if log_file.exists():
    residuals_post = utilities.ResidualsPost(log_file)
    residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
    print("Résidus analysés")

# Post-processing with PyVista (skip VTK generation - already done)
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
# foam_post.foamToVTK()  # Skip - already exists

# Get all available time steps
time_steps = foam_post.get_all_time_steps()
print(f"Time steps disponibles: {time_steps}")

if not time_steps:
    print("Aucun time step trouvé!")
    exit(1)

# Load the last time step
latest_time_step = time_steps[-1]
print(f"Chargement du time step: {latest_time_step}")

structure = foam_post.load_time_step(latest_time_step)
cell_mesh = structure["cell"]
boundaries = structure["boundaries"]

print(f"Maillage chargé: {cell_mesh}")
print(f"N Cells: {cell_mesh.n_cells}")
print(f"N Points: {cell_mesh.n_points}")
print(f"Frontières: {list(boundaries.keys())}")

# Create visualization directory
viz_dir = current_path / "visualisations"
viz_dir.mkdir(exist_ok=True)

print("\n--- Test slicing ---")

# Get the mesh bounds
bounds = cell_mesh.bounds
print(f"Mesh bounds: x=[{bounds[0]:.1f}, {bounds[1]:.1f}], y=[{bounds[2]:.1f}, {bounds[3]:.1f}], z=[{bounds[4]:.1f}, {bounds[5]:.1f}]")

# The domain is rotated, so z is actually the vertical direction
# Try slicing at different z values (vertical direction)
# Note: mesh bounds are z=[-40, 40]
test_z_values = [0, 15, 20, 30]
for z_val in test_z_values:
    print(f"\nTrying slice at z={z_val}...")
    try:
        # Correct API: slice(normal='z', origin=(x, y, z))
        slice_mesh = cell_mesh.slice(normal='z', origin=(0, 0, z_val))
        print(f"  Slice has {slice_mesh.n_points} points, {slice_mesh.n_cells} cells")
        if slice_mesh.n_points > 0:
            print(f"  SUCCESS!")
            # Plot this slice with velocity magnitude
            pl = pv.Plotter(off_screen=True)
            # Compute velocity magnitude for coloring
            if 'U' in slice_mesh.point_data:
                U = slice_mesh.point_data['U']
                velocity_magnitude = np.linalg.norm(U, axis=1)
                slice_mesh.point_data['velocity_magnitude'] = velocity_magnitude
                pl.add_mesh(slice_mesh, scalars='velocity_magnitude', show_scalar_bar=True, 
                           opacity=1.0, cmap='viridis', show_edges=False)
            else:
                pl.add_mesh(slice_mesh, show_scalar_bar=True, opacity=1.0)
            pl.camera_position = 'xy'
            output_file = viz_dir / f"slice_z{z_val}.png"
            pl.screenshot(str(output_file))
            print(f"  Saved to {output_file}")
            pl.close()
    except Exception as e:
        print(f"  ERROR: {e}")

# Try x-normal slice (vertical plane in wind direction)
print("\n--- Testing x-normal slice ---")
try:
    slice_mesh = cell_mesh.slice(normal='x', origin=(0, 0, 0))
    print(f"Slice x has {slice_mesh.n_points} points")
    if slice_mesh.n_points > 0:
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(slice_mesh, scalars='U', show_scalar_bar=True)
        pl.camera_position = 'yz'
        pl.screenshot(str(viz_dir / "slice_x.png"))
        print("Saved slice_x.png")
        pl.close()
except Exception as e:
    print(f"ERROR: {e}")

# Try y-normal slice
print("\n--- Testing y-normal slice ---")
try:
    slice_mesh = cell_mesh.slice(normal='y', origin=(0, 0, 0))
    print(f"Slice y has {slice_mesh.n_points} points")
    if slice_mesh.n_points > 0:
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(slice_mesh, scalars='U', show_scalar_bar=True)
        pl.camera_position = 'xz'
        pl.screenshot(str(viz_dir / "slice_y.png"))
        print("Saved slice_y.png")
        pl.close()
except Exception as e:
    print(f"ERROR: {e}")

# Check available point data
print("\n--- Available arrays ---")
print(f"Point data: {list(cell_mesh.point_data.keys())}")
print(f"Cell data: {list(cell_mesh.cell_data.keys())}")

# Try contour plot
print("\n--- Testing contour plot ---")
try:
    pl_contour = pv.Plotter(off_screen=True)
    pl_contour.add_mesh(cell_mesh, scalars='p', show_scalar_bar=True)
    pl_contour.screenshot(str(viz_dir / "contour_pression.png"))
    print("Saved contour_pression.png")
    pl_contour.close()
except Exception as e:
    print(f"ERROR: {e}")

# Test vector plot
print("\n--- Testing vector plot ---")
try:
    pl_vectors = pv.Plotter(off_screen=True)
    cell_mesh.set_active_vectors('U')
    arrows = cell_mesh.glyph(orient='U', factor=0.0003)
    pl_vectors.add_mesh(arrows, color='blue')
    pl_vectors.screenshot(str(viz_dir / "vector_plot.png"))
    print("Saved vector_plot.png")
    pl_vectors.close()
except Exception as e:
    print(f"ERROR: {e}")

print("\n--- Test completed ---")
print(f"Visualizations saved to: {viz_dir}")
