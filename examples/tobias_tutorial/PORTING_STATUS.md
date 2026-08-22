# Tobias tutorial porting status

This register is intentionally conservative: **validated** means that the case-specific `run.py` has completed the intended OpenFOAM 13 workflow. A source archive, generated template, or syntax-only test is not sufficient.

| Status | Cases |
| --- | --- |
| Validated | `2d_rotational_axis_symmetric`, `pitot_tube`, `fluidic_oscillator`, `falling_droplets`, `magnus_effect`, `cell_zone_generation`, `meshing_pipe_45deg`, `meshing_pipe_90deg`, `2d_ami_ncc` |
| Pending | All other Tobias training cases in the source catalog |

The pending cases require the same sequence: obtain the complete archive where the GitHub repository omits geometry, create a dedicated directory, generate dictionaries and fields through FoamPilot, execute every documented mesh and solver stage with OpenFOAM 13, inspect the rendered files and logs, and write a case report. Cases that require Salome, DAKOTA, ParaView-only operations, unavailable OpenFOAM.com functionality, or multi-day production runs must record those constraints explicitly and must not be marked validated without a successful executable workflow.

The current PR is intentionally incremental. It contains the shared exact-dictionary writer and nine validated case ports, including the 2D arbitrary mesh interface/non-conformal coupling tutorial. The AMI/NCC runner includes case-local OpenFOAM 13 syntax adaptations and validates conversion, snappy meshing, baffle creation, non-conformal coupling, and a short incompressible calculation.
