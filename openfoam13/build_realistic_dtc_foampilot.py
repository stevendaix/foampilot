#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "foampilot"
sys.path.insert(0, str(REPO / "src"))

from foampilot.mesh.marine_motion import write_six_dof_dynamic_mesh_dict
from foampilot.solver.marine_case import MarineCaseConfig
from foampilot.tutorials.openfoam13 import validate_generated_case

_foam_tutorials = os.environ.get("FOAM_TUTORIALS")
if not _foam_tutorials:
    raise SystemExit("FOAM_TUTORIALS must point to the Foundation 13 tutorials")
SOURCE = Path(_foam_tutorials) / "incompressibleVoF" / "DTCHullWave"
TARGET = ROOT / "FoamPilotCases" / "DTCRealisticFoundation13"

if TARGET.exists():
    shutil.rmtree(TARGET)
shutil.copytree(SOURCE, TARGET, symlinks=True)
initial_fields = tuple((TARGET / "0").glob("*.orig"))
if initial_fields and not (TARGET / "0" / "U").exists():
    for source_field in initial_fields:
        shutil.copy2(source_field, TARGET / "0" / source_field.stem)

# FoamPilot owns the solver contract while the Foundation tutorial supplies the
# complete fluid domain, hull patch, wave setup and physically meaningful fields.
(TARGET / "constant" / "marineProperties").write_text(
    """FoamFile
{
    format ascii;
    class dictionary;
    object marineProperties;
}

mode dtc_moving;
solver incompressibleVoF;
meshBackend snappyHexMesh;
fluid waterAir;
freeSurfaceLevel 0.244;
"""
)
phase = TARGET / "constant" / "phaseProperties"
phase.write_text(phase.read_text().replace("sigma           0;", "sigma           0.072;"))

write_six_dof_dynamic_mesh_dict(
    TARGET, body_name="hull", patch_name="hull", mass=412.73,
    centre_of_mass=(0.0, 0.0, 0.0),
    inertia=(40.0, 0.0, 0.0, 921.0, 0.0, 921.0),
    transform_origin=(2.929541, 0.0, 0.2),
    inner_distance=0.3, outer_distance=1.0,
    translation_damper_coeff=8596.0,
    rotation_damper_coeff=11586.0,
)

control = TARGET / "system" / "controlDict"
text = control.read_text()
text = re.sub(r"application\s+[^;]+;", "application     marineFoam;", text, count=1)
if "application" not in text:
    marker = "}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"
    text = text.replace(marker, marker + "\napplication     marineFoam;", 1)
text = re.sub(r"endTime\s+[^;]+;", "endTime         0.0002;", text, count=1)
control.write_text(text)

# Keep Foundation's official force and rigid-body diagnostics unchanged.
# The custom inter-mesh constraint is intentionally not enabled in this
# baseline: this case is the physical full-domain validation; the separate
# DTC overset harness validates MarineOversetConstraint.
mesh_script = SOURCE / "Allmesh"
shutil.copy2(mesh_script, TARGET / "Allmesh.FoamPilot")

config = MarineCaseConfig.from_case(TARGET)
config.validate_files()
config.validate_foundation13()
validation = validate_generated_case(TARGET, is_vof=True)
if not validation.valid:
    raise SystemExit(f"Generated case validation failed: {validation}")
print(f"created {TARGET}")
print(f"mode={config.mode} solver={config.solver} baseline=DTCHullWave")
print("structural validation: valid=True")
