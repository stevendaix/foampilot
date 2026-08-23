# OpenFOAM 13 tutorials with FoamPilot

This directory contains the FoamPilot-side integration for the external
`shor-ty/OpenFOAMTutorials` repository. The source repository is treated as
an audit input only: generated OpenFOAM dictionaries are owned by FoamPilot
and are produced from Python.

## Workflow

The supported workflow is deterministic. First install OpenFOAM 13 using the
[official Ubuntu instructions](https://openfoam.org/download/13-ubuntu/), then
install FoamPilot in editable mode from the repository root. Next run the
manifest audit:

```bash
python audit_repository.py /path/to/OpenFOAMTutorials openfoam_tutorial_manifest.csv
```

The manifest records the two OpenFOAM families, whether a `run` script exists,
and whether a case refers to external STL, STEP, IGES or UNV data. The GitHub
repository intentionally omits many large geometry files, so a case classified
as requiring external geometry cannot be considered executable until those
inputs are supplied.

The first self-contained integration is the laminar channel:

```bash
cd 01_laminar_channel
python run.py
```

`run.py` uses `Solver`, `Meshing`, `ValueWithUnit`, the FoamPilot boundary API,
and `run_foampilot_case`. It generates `system/`, `constant/`, `0/`, and the
mesh, validates the generated case contract, and runs `foamRun` in the
OpenFOAM 13 environment. Incompressible cases must explicitly contain
`constant/transportProperties` with a `nu` entry.

## Acceptance criteria

A tutorial integration is accepted only when FoamPilot generates the complete
case, the generated files are inspected, the mesh utility succeeds, the
solver exits successfully, and the log contains a normal completion marker.
Cases depending on omitted geometry are reported as blocked rather than
silently treated as validated.
