#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Foampilot v4.0 Tutorial Template

This is the canonical reference for all tutorials and examples.
It demonstrates the proper FoamPilot API without any file manipulation.
"""
from pathlib import Path
from foampilot.solver import Solver
from foampilot.core.physics.fluids_theory import FluidMechanics
from foampilot.core.units.manageunits import ValueWithUnit
from foampilot.core.base.meshing import Meshing
from foampilot.core.postprocessing.openfoam_pyvista import FoamPostProcessing
from foampilot.core.reporting.latex_pdf import LatexDocument

# 1. CONFIGURATION & PHYSICS
case_path = Path(__file__).parent / "case_data"
fluid = FluidMechanics(
    'Water',
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)

# 2. SOLVER INITIALIZATION (API Fluide)
solver = Solver(case_path)
solver.compressible = False
solver.constant.transportProperties.nu = fluid.get_fluid_properties()['kinematic_viscosity']

# 3. MESHING
mesh_tool = Meshing(case_path, mesher="blockMesh")
mesh_tool.mesher.run()

# 4. BOUNDARY CONDITIONS
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"))
)
solver.boundary.write_boundary_conditions()

# 5. EXECUTION
solver.run_simulation()

# 6. POST-PROCESSING
post = FoamPostProcessing(case_path=case_path)
post.foamToVTK()
stats = post.get_region_statistics(
    post.load_time_step(post.get_all_time_steps()[-1]),
    "cell", "U"
)

# 7. REPORT
doc = LatexDocument(title="Rapport: Cavity", filename="report", output_dir=case_path)
doc.add_section("Statistiques", str(stats))
doc.generate_document(output_format="pdf")