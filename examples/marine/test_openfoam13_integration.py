"""OpenFOAM 13 integration smoke test for the FoamPilot marine workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from foampilot.workflows.marine import (
    dtc_openfoam13_workflow,
    write_mrf_properties,
    write_openfoam13_rigid_body_mover,
    write_overset_dynamic_mesh,
)
from foampilot.workflows.openfoam import OpenFOAMWorkflow


workspace = Path.cwd() / "openfoam13-marine-smoke"
if workspace.exists():
    shutil.rmtree(workspace)
workspace.mkdir(parents=True)
tutorials = Path(os.environ["FOAM_TUTORIALS"])

# Validate the generic runner with a stock Foundation 13 tutorial and actual
# mesh/solver execution.
pitz_template = tutorials / "incompressibleFluid" / "pitzDailySteady"
pitz_case = workspace / "pitzDailySteady"
shutil.copytree(pitz_template, pitz_case)
subprocess.run(
    ["foamDictionary", "-entry", "endTime", "-set", "20", "system/controlDict"],
    cwd=pitz_case,
    check=True,
)
pitz_workflow = OpenFOAMWorkflow(pitz_case, "openfoam13-pitz-daily")
pitz_workflow.add_command("block-mesh", "blockMesh")
pitz_workflow.add_command("solve", "foamRun")
pitz_results = pitz_workflow.run()

# Build the mesh source prescribed by the official DTCHullMoving tutorial.
dtc_tutorial_root = tutorials / "incompressibleVoF"
dtc_mesh = workspace / "DTCHull"
dtc_case = workspace / "DTCHullMoving"
shutil.copytree(dtc_tutorial_root / "DTCHull", dtc_mesh)
shutil.copytree(dtc_tutorial_root / "DTCHullMoving", dtc_case)
shutil.copy2(
    tutorials / "resources" / "geometry" / "DTC-scaled.stl.gz",
    dtc_mesh / "constant" / "geometry" / "DTC-scaled.stl.gz",
)
for command in (
    ("surfaceFeatures",),
    ("blockMesh",),
    ("refineMesh",),
    ("snappyHexMesh",),
    ("renumberMesh", "-noFields"),
):
    subprocess.run(command, cwd=dtc_mesh, check=True)

# Replace the copied tutorial mover with FoamPilot's generated equivalent,
# shorten the run, then execute the complete native OpenFOAM 13 DTC workflow.
write_openfoam13_rigid_body_mover(dtc_case)
subprocess.run(
    ["foamDictionary", "-entry", "endTime", "-set", "0.001", "system/controlDict"],
    cwd=dtc_case,
    check=True,
)
dtc_results = dtc_openfoam13_workflow(dtc_case, mesh_source=Path("../DTCHull")).run()

# The legacy and Foundation-13 dictionary helpers are also accepted by the
# OpenFOAM dictionary parser.
marine_case = workspace / "marine-dictionaries"
mrf = write_mrf_properties(marine_case / "propeller", omega=100.0)
legacy_dynamic = write_overset_dynamic_mesh(marine_case / "legacy-overset")
foundation_dynamic = write_openfoam13_rigid_body_mover(marine_case / "foundation13")
for dictionary, entry in (
    (mrf, "MRF1"),
    (legacy_dynamic, "dynamicFvMesh"),
    (foundation_dynamic, "mover"),
):
    subprocess.run(
        ["foamDictionary", str(dictionary), "-entry", entry, "-value"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

for case, results, log_name in (
    (pitz_case, pitz_results, "02_solve.log"),
    (dtc_case, dtc_results, "04_solve.log"),
):
    solver_log = case / "logs" / log_name
    assert "End" in solver_log.read_text(encoding="utf-8"), f"foamRun did not finish in {case}"
    assert all(result.status == "completed" for result in results)

print("OPENFOAM13_MARINE_SMOKE_TEST=PASS")
print(f"workspace={workspace}")
print(f"dtc_log={dtc_case / 'logs' / '04_solve.log'}")
