# codedFixedValue Vadier Test Case

Test case for `codedFixedValue` boundary conditions with a logarithmic wind
profile inlet in OpenFOAM 13, simulating airflow over a harbor/shallow-water
("le vadier") domain.

## How it works

`codedFixedValue` lets users write C++ code inline in the boundary condition
file. At runtime, OpenFOAM:

1. Reads the `code` block (`#{ ... #};` delimiters).
2. Substitutes it into the C++ template
   `codedFixedValueFvPatchFieldTemplate.C`.
3. Compiles a shared library via `wmake` (dynamic code compilation).
4. Loads and applies the boundary condition.

### Correct OpenFOAM API (validated in OF13)

- Use `this->patch().Cf()` to get face centres — **not** `pos()`.
- Access vector components via `.z()`, `.x()`, `.y()` (they're overloaded functions).
- Set values with `operator==()` on a `vectorField` or `scalarField` —
  **not** `result = ...`.
- Each `codedFixedValue` BC requires a `name` entry (the generated class name).

### Files

| File | Purpose |
|---|---|
| `system/blockMeshDict` | 100×50×20 m rectangular domain, hex mesh |
| `system/controlDict` | `foamRun` with `incompressibleFluid` solver |
| `system/fvSchemes` | Steady-state schemes + `meshWave wallDist` |
| `system/fvSolution` | SIMPLE with GAMG/smoothSolver for all fields |
| `constant/transportProperties` | Newtonian, nu = 1.5e-5 |
| `constant/turbulenceProperties` | k-omega SST |
| `0/U` | codedFixedValue inlet with log-wind profile |
| `0/k` | codedFixedValue inlet with TKE profile |
| `0/omega` | codedFixedValue inlet with specific dissipation |
| `0/p` | zeroGradient inlet, fixedValue outlet |
| `0/nut` | nutkWallFunction on walls |
| `constant/polyMesh/boundary` | INLET/OUTLET/WALLS/TOP/GROUND patches |

## Running

```bash
source /opt/openfoam13/etc/bashrc
foamRun -solver incompressibleFluid -case .
```

All three `codedFixedValue` BCs (`inletVelocityProfile`, `inletTkeProfile`,
`inletOmegaProfile`) compile and link on first run, then the SIMPLE loop
converges.

## Regenerating boundary files

```bash
cd /home/steven/foampilot/foampilot
PYTHONPATH=src python3 -c "
from pathlib import Path
from foampilot.base.openFOAMFile import OpenFOAMFile
# ... (see test_coded_fixed_value.py for full generation script)
"
```
