#!/usr/bin/env python

# Import required libraries
from foampilot.solver import Solver
from foampilot import Meshing, commons, utilities, postprocess,latex_pdf
from foampilot.utilities.fluids_theory import FluidMechanics
import pyvista as pv
from pathlib import Path
from foampilot.utilities.manageunits import Quantity
import numpy as np
import classy_blocks as cb
import json
import pandas as pd

# Define the working directory for the simulation case
current_path = Path.cwd() / 'exemple2'

# List available fluids
print("Available fluids:")
available_fluids = FluidMechanics.get_available_fluids()
for name in available_fluids:
    print(f"- {name}")

# Create a FluidMechanics instance for water at room temperature and atmospheric pressure
fluid_mech = FluidMechanics(
    available_fluids['Water'],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
)

# Get fluid properties including kinematic viscosity
properties = fluid_mech.get_fluid_properties()
kinematic_viscosity = properties['kinematic_viscosity']
print(f"\nUsing fluid: Water")
print(f"Kinematic viscosity: {kinematic_viscosity}")

# Initialize the solver 
solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False

# Set the kinematic viscosity in the solver's constant directory

solver.constant.transportProperties.nu=kinematic_viscosity

# Generate system and constant directories with updated OpenFOAM configuration
system_dir = solver.system.write()
system_dir = solver.constant.write()

# Convert numerical schemes settings to Python dictionary for inspection
solver.system.fvSchemes.to_dict()

# Geometric parameters for the muffler design
pipe_radius = 0.05      # Radius of the inlet/outlet pipe
muffler_radius = 0.08   # Radius of the muffler chamber
ref_length = 0.1        # Reference length for geometry segments

# Mesh parameters
# Constant cell size for uniform mesh visualization
cell_size = 0.015

# List to store geometric shapes
# Indices in this list correspond to shapes in the original sketch
shapes = []

# 0: Create the first cylinder (inlet pipe)
# The cylinder is defined by its start point, end point, and a point defining its radius
shapes.append(cb.Cylinder([0, 0, 0], [3 * ref_length, 0, 0], [0, pipe_radius, 0]))
# Define axial meshing (along the cylinder's axis)
shapes[-1].chop_axial(start_size=cell_size)
# Define radial meshing (from center to outside)
shapes[-1].chop_radial(start_size=cell_size)
# Define tangential meshing (around the cylinder)
shapes[-1].chop_tangential(start_size=cell_size)
# Assign an 'inlet' patch (surface) to the starting face of the cylinder
shapes[-1].set_start_patch("inlet")

# 1: Chain a cylinder to the previous shape
# The 'chain' method creates a new shape that extends the previous one
shapes.append(cb.Cylinder.chain(shapes[-1], ref_length))
# Axial meshing for this new cylinder segment
shapes[-1].chop_axial(start_size=cell_size)

# 2: Create an extruded ring (start of the muffler)
# The 'expand' method creates an extruded ring by increasing the radius of the previous shape
shapes.append(cb.ExtrudedRing.expand(shapes[-1], muffler_radius - pipe_radius))
# Radial meshing for the extruded ring
shapes[-1].chop_radial(start_size=cell_size)

# 3: Chain an extruded ring (muffler body)
shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
# Axial meshing for this ring segment
shapes[-1].chop_axial(start_size=cell_size)

# 4: Chaînage d'un autre anneau extrudé (fin du silencieux)
shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
# Maillage axial pour ce segment d'anneau.
shapes[-1].chop_axial(start_size=cell_size)

# 5: Fill the extruded ring (return to a cylinder)
# The 'fill' method creates a cylinder that fills the inner space of the previous extruded ring
shapes.append(cb.Cylinder.fill(shapes[-1]))
# Radial meshing for the filling cylinder
shapes[-1].chop_radial(start_size=cell_size)

# 6: Create an elbow
# Calculate the elbow center based on the previous shape
elbow_center = shapes[-1].sketch_2.center + np.array([0, 2 * muffler_radius, 0])
# The 'Elbow.chain' method creates an elbow that extends the previous shape
# Parameters define the curve angle, rotation center, rotation axis, and elbow radius
shapes.append(
    cb.Elbow.chain(shapes[-1], np.pi / 2, elbow_center, [0, 0, 1], pipe_radius)
)
# Axial meshing for the elbow
shapes[-1].chop_axial(start_size=cell_size)
# Assign an 'outlet' patch to the end face of the elbow
shapes[-1].set_end_patch("outlet")

# Initialize the Mesh object
# This is the main object that will contain all shapes and generate the blockMeshDict
mesh = cb.Mesh()
# Add all created shapes to the mesh
for shape in shapes:
    mesh.add(shape)

# Définition d'un patch par défaut nommé 'walls' avec le type 'wall'.
# Cela s'applique à toutes les surfaces qui n'ont pas été explicitement définies avec un patch.
# Set default patch type for all unspecified boundaries
mesh.set_default_patch("walls", "wall")

# Write output files
# First argument is the path to OpenFOAM's blockMeshDict file
# Second argument is the path to a debug VTK file, useful for visualization
mesh.write(current_path / "system" / "blockMeshDict", current_path /"debug.vtk")

print("Successfully generated blockMeshDict and debug.vtk files in the case directory.")

# Initialize meshing object and run blockMesh utility

mesh = Meshing(current_path,mesher="blockMesh")
mesh.mesher.run(current_path / "city_block_cfd_domain.step")

# --- 3. Boundary Conditions Management ---

# Initialize boundary conditions
solver.boundary.initialize_boundary()

# Set inlet velocity boundary condition
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")),
    turbulence_intensity=0.05  # 5% turbulence intensity
)

# Set outlet pressure boundary condition
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)


solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)


# Write boundary condition files for all fields
solver.boundary.write_boundary_conditions()


print("Boundary condition files have been generated")

# Execute the simulation
solver.run_simulation()

# Post-process simulation residuals
residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
# Export residuals in multiple formats for analysis
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)

# Initialize post-processing instance
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
# Convert OpenFOAM results to VTK format for visualization
foam_post.foamToVTK()

# Get all available time steps from the simulation
time_steps = foam_post.get_all_time_steps()
print(f"Available time steps: {time_steps}")

# Load and analyze the results from the latest time step
if time_steps:
    latest_time_step = time_steps[-1]
    # Load mesh and boundary data
    structure = foam_post.load_time_step(latest_time_step)
    cell_mesh = structure["cell"]
    boundaries = structure["boundaries"]
    print(f"Main mesh loaded for time step {latest_time_step}: {cell_mesh}")
    print(f"Boundaries loaded: {list(boundaries.keys())}")

    print("\n--- Testing visualization features (image generation) ---")

        # Create a slice visualization
    print("Generating a slice plot...")
    foam_post.plot_slice(
        structure=structure,
        plane="z",
        scalars="U",
        opacity=0.25,
        path_filename=current_path / "slice_plot.png"
    )

    # Create a pressure contour visualization
    print("Generating a contour plot...")
    pl_contour = pv.Plotter(off_screen=True)
    pl_contour.add_mesh(cell_mesh, scalars='p', show_scalar_bar=True)
    foam_post.export_plot(pl_contour, current_path / "contour_plot.png")

    # Create a velocity vector field visualization
    print("Generating a vector plot...")
    pl_vectors = pv.Plotter(off_screen=True)
    cell_mesh.set_active_vectors('U')
    # Create arrow glyphs oriented by velocity field
    arrows = cell_mesh.glyph(orient='U', factor=0.0003)
    pl_vectors.add_mesh(arrows, color='blue')
    foam_post.export_plot(pl_vectors, current_path / "vector_plot.png")

    # Create a mesh wireframe visualization
    print("Generating a mesh style plot...")
    pl_mesh_style = pv.Plotter(off_screen=True)
    pl_mesh_style.add_mesh(cell_mesh, style='wireframe', show_edges=True, color='red')
    foam_post.export_plot(pl_mesh_style, current_path / "mesh_style_plot.png")

    print("\n--- Testing advanced flow analysis features ---")

    # Calculate Q-criterion for vortex identification
    print("Calculating Q-criterion...")
    mesh_with_q = foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
    if 'q_criterion' in mesh_with_q.point_data:
        print(f"Q-criterion calculated. Value range: {mesh_with_q.point_data['q_criterion'].min():.2e} to {mesh_with_q.point_data['q_criterion'].max():.2e}")
    else:
        print("Failed to calculate Q-criterion.")

    # Calculate vorticity field
    print("Calculating vorticity...")
    mesh_with_vorticity = foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")
    if 'vorticity' in mesh_with_vorticity.point_data:
        print(f"Vorticity calculated. Value range: {mesh_with_vorticity.point_data['vorticity'].min():.2e} to {mesh_with_vorticity.point_data['vorticity'].max():.2e}")
    else:
        print("Failed to calculate vorticity.")

    print("\n--- Performing statistical analysis ---")

    # Calculate mesh quality statistics
    print("Calculating mesh statistics...")
    mesh_stats = foam_post.get_mesh_statistics(cell_mesh)
    print(f"Mesh statistics: {mesh_stats}")

    # Calculate velocity field statistics in cell region
    print("Calculating statistics for 'cell' region and 'U' field...")
    cell_region_stats = foam_post.get_region_statistics(structure, "cell", "U")
    print(f"'Cell' region statistics for 'U': {cell_region_stats}")

    # Calculate pressure field statistics on boundary1 if it exists
    if "boundary1" in boundaries:
        print("Calculating statistics for 'boundary1' region and 'p' field...")
        boundary_region_stats = foam_post.get_region_statistics(structure, "boundary1", "p")
        print(f"'Boundary1' region statistics for 'p': {boundary_region_stats}")

    # Export cell data to CSV for external analysis
    print("Exporting 'cell' region data to CSV file...")
    foam_post.export_region_data_to_csv(structure, "cell", ["U", "p"], current_path / "cell_data.csv")

    # Compile and export all statistics to JSON
    print("Exporting statistics to JSON file...")
    all_stats = {
        "mesh_stats": mesh_stats,
        "cell_region_stats_U": cell_region_stats,
        "boundary1_region_stats_p": boundary_region_stats if "boundary1" in boundaries else "N/A"
    }
    foam_post.export_statistics_to_json(all_stats, current_path / "all_stats.json")

    # Create animation of the velocity field evolution
    print("Creating an animation...")
    foam_post.create_animation(scalars='U', filename= current_path / 'animation_test.gif', fps=5)

else:
    print("No time steps found, unable to test the class.")

    print("\nTest completed.")






current_path = Path.cwd() / "exemple2"

# Load statistics JSON
stats_file = current_path / "all_stats.json"
with open(stats_file, "r") as f:
    stats = json.load(f)

# Load cell data CSV
cell_csv = current_path / "cell_data.csv"
cell_df = pd.read_csv(cell_csv)

# Create LaTeX document
doc = latex_pdf.LatexDocument(
    title="Simulation Report: Muffler Flow Case",
    author="Automated Report",
    filename="simulation_report",
    output_dir=current_path
)

doc.add_title()
doc.add_toc()
doc.add_abstract("This report summarizes the results of the incompressible fluid simulation for the muffler case.")

# Sections
doc.add_section("Fluid Properties", f"Kinematic viscosity: {stats['cell_region_stats_U']['nu'] if 'nu' in stats['cell_region_stats_U'] else 'N/A'}")

# Mesh statistics
doc.add_section("Mesh Statistics", "Summary of mesh quality metrics:")
mesh_stats = stats.get("mesh_stats", {})
mesh_table_data = [[k, v] for k, v in mesh_stats.items()]
doc.add_table(
    mesh_table_data,
    headers=["Statistic", "Value"],
    caption="Mesh Quality Statistics"
)

# Cell region statistics
doc.add_section("Velocity Field Statistics (Cell Region)", "Statistics of the velocity field in the cell region.")
cell_stats = stats.get("cell_region_stats_U", {})
cell_table_data = [[k, v] for k, v in cell_stats.items()]
doc.add_table(
    cell_table_data,
    headers=["Statistic", "Value"],
    caption="Velocity Field ('U') Statistics"
)

# Boundary statistics if available
if "boundary1_region_stats_p" in stats and stats["boundary1_region_stats_p"] != "N/A":
    doc.add_section("Pressure Field Statistics (Boundary1)", "Statistics of the pressure field on boundary1.")

    boundary_stats = stats["boundary1_region_stats_p"]
    boundary_table_data = [[k, v] for k, v in boundary_stats.items()]
    doc.add_table(
        boundary_table_data,
        headers=["Statistic", "Value"],
        caption="Pressure Field ('p') Statistics"
    )

# Figures
doc.add_section("Visualizations", "Figures representing flow, pressure, velocity vectors, and mesh.")
for img_name in ["slice_plot.png", "contour_plot.png", "vector_plot.png", "mesh_style_plot.png"]:
    img_path = current_path / img_name
    if img_path.exists():
        doc.add_figure(str(img_path), caption=img_name.replace("_", " ").title(), width="0.7\\textwidth")

# Animation
# animation_file = current_path / "animation_test.gif"
# if animation_file.exists():
#     doc.add_section("Animation")
#     doc.add_figure(str(animation_file), caption="Velocity Field Evolution", width="0.7\\textwidth")

# Appendix
doc.add_appendix("Cell Data Export", f"The cell data has been exported to {cell_csv.name} for further analysis.")

# Generate PDF
doc.generate_document(output_format="pdf")
print("PDF report generated successfully.")
