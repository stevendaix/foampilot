from foampilot.utilities.make_human import HumanGeometry
from build123d import *
from build123d import exporters3d

import gmsh
import random
#!/usr/bin/env python
from pathlib import Path

# Import required libraries
from foampilot import Meshing, commons, postprocess,latex_pdf,ValueWithUnit,FluidMechanics,Solver

current_path = Path.cwd()

human = HumanGeometry(height=1.75, posture="standing")
part = human.build_human()


human.export_step_all(
        human_filename= "human_model_test.step",

    ) 

human.export_step_all(
        human_filename= "human_model.step",
        domain_filename= "domain.step",
        domain_size= [3.0, 3.0, 5.0],
    ) 


# mesh = Meshing(current_path,mesher="gmsh")
# mesh.mesher.load_geometry(current_path / "human_model.step")


# mesh.mesher.mesh_volume( lc_min= 0.001,
#                         lc_max= 5,
#                        )

# mesh.mesher.get_basic_mesh_stats()

# mesh.mesher.analyze_mesh_quality()

# mesh.mesher.get_volume_tags()

# mesh.mesher.get_face_tags()



# mesh.mesher.export_to_openfoam(run_gmshtofoam = True)

# mesh.mesher.finalize()