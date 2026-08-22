"""OpenFOAM 13 integration smoke test for the FoamPilot marine workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from foampilot.workflows.marine import write_mrf_properties, write_overset_dynamic_mesh
from foampilot.workflows.openfoam import OpenFOAMWorkflow


workspace = Path.cwd() / "openfoam13-marine-smoke"
if workspace.exists():
    shutil.rmtree(workspace)
workspace.mkdir(parents=True)

# A stock OpenFOAM 13 tutorial validates command execution through the new
# workflow runner with a real mesh and solver, without relying on legacy cases.
tutorial = Path(os.environ["FOAM_TUTORIALS"]) / "incompressibleFluid" / "pitzDailySteady"
case = workspace / "pitzDailySteady"
shutil.copytree(tutorial, case)
subprocess.run(
    ["foamDictionary", "-entry", "endTime", "-set", "20", "system/controlDict"],
    cwd=case,
    check=True,
)
workflow = OpenFOAMWorkflow(case, "openfoam13-pitz-daily")
workflow.add_command("block-mesh", "blockMesh")
workflow.add_command("solve", "foamRun")
results = workflow.run()

# The marine helpers are validated by OpenFOAM's dictionary parser itself.
marine_case = workspace / "marine-dictionaries"
mrf = write_mrf_properties(marine_case / "propeller", omega=100.0)
dynamic = write_overset_dynamic_mesh(marine_case / "overset")
for dictionary, entry in ((mrf, "MRF1"), (dynamic, "dynamicFvMesh")):
    subprocess.run(
        ["foamDictionary", str(dictionary), "-entry", entry, "-value"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

solver_log = case / "logs" / "02_solve.log"
assert "End" in solver_log.read_text(encoding="utf-8"), "foamRun did not finish successfully"
assert all(result.status == "completed" for result in results)
print("OPENFOAM13_MARINE_SMOKE_TEST=PASS")
print(f"workspace={workspace}")
print(f"solver_log={workspace / 'pitzDailySteady' / 'logs' / '02_solve.log'}")
