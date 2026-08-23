# Tobias — Falling Droplets

## Objective

This case reproduces Tobias Holzmann’s Falling Droplets multiphase training case. It uses `interFoam`/`incompressibleVoF` to study droplets under gravity and an upward flow in a 2-D simplified domain, with the non-physical inlet condition documented by Tobias.

## FoamPilot workflow

`run.py` generates the dictionaries and initial fields through `OpenFOAMDictAddFile.write_raw`, converts `cad/meshSquare.unv` with `ideasUnvToFoam`, applies `changeDictionary`, initializes the droplets with `setFields`, and launches `foamRun -solver incompressibleVoF` through `Solver.run_command`. The source shell runner is not executed.

For OpenFOAM 13, the runner changes `div(phi,alpha)` to `Gauss interfaceCompression vanLeer 1` and removes the obsolete `cAlpha` entry. The source `endTime` of 0.12 s is shortened to 0.002 s for bounded validation.

## Validation

The case was executed successfully with OpenFOAM 13. `ideasUnvToFoam`, `changeDictionary`, `setFields`, and the VoF solver all returned success; the generated solver log contains completed alpha/MULES and pressure steps. This is a **validated short calculation**, not the full 0.12 s training campaign.

## Reference

[1]: https://holzmann-cfd.com/community/training-cases/falling-droplets — Tobias Holzmann, Falling Droplets.
