# FoamPilot v3 Architecture

## Principles

### 1. Version Agnosticism
An OpenFOAM version (Foundation 13, 14, OpenFOAM.com) is **not** a capability, workflow, or extension. It is a **runtime backend metadata**. It must never structure the codebase.

### 2. Workflows are Composition Scripts
A workflow (e.g., `muffler_simulation.py`) is an **imperative Python script** that imports and assembles generic capabilities. It must never be encapsulated in an opaque class.

### 3. Strict Dependency Flow
```
Workflows → Capabilities (core/) → Backend (openfoam/) → External Libraries
```

**Forbidden**: `core/` must never import `workflows/`, `examples/`, or version-specific modules.

## Directory Structure

```
foampilot/
├── core/                    # Generic, reusable capabilities
│   ├── dictionaries/         # FoamDict, BoundaryDict, CaseLayout
│   ├── geometry/             # CAD, surfaces, topology, VMTK
│   ├── meshing/             # Gmsh, blockMesh, snappyHexMesh
│   ├── boundaries/          # Boundary condition management
│   ├── postprocessing/      # VTK, residuals, stats, PyVista
│   ├── reporting/           # LaTeX, JSON, CSV reports
│   ├── units/              # ValueWithUnit, Pint
│   └── physics/            # FluidMechanics, OpenFOAM configs
│
├── openfoam/                # Execution backend
│   ├── backend/            # Version/distribution detection
│   ├── environment/         # bashrc, environment variables
│   ├── runner/             # subprocess, mpirun, decomposePar
│   ├── solvers/            # Solver registry and configs
│   └── extensions/         # C++ couplings (FSI, Cantera, Yade)
│
├── extensions/              # Domain-specific capabilities
│   ├── marine/             # Marine and offshore CFD
│   ├── urban/              # Urban canopy modeling
│   └── energy/             # Energy sector
│
├── workflows/               # Imperative composition scripts
│   ├── medical/            # Aorta, vessels, biomechanics
│   ├── marine/             # Muffler, propulsion
│   ├── urban/              # Neighborhood CFD
│   ├── energy/             # Heat exchangers
│   └── wind/               # Floating turbines
│
├── patches/                 # Version-specific bug fixes
│   └── openfoam/
│       └── foundation/
│           └── 13/         # OpenFOAM 13 patches
│
└── tools/
    └── audit/
        └── check_architecture.py  # CI architectural validation
```

## Migration Notes

### Shim Pattern (Strangler Fig)
When moving modules, the old location becomes a wrapper that imports the new location and raises a `DeprecationWarning`. This ensures backward compatibility during transition.

### What Goes Where

| Content | Destination |
|---------|-------------|
| Generic capability | `core/` |
| Version-specific bug fix | `patches/openfoam/foundation/<version>/` |
| Domain-specific capability | `extensions/<domain>/` |
| Simulation pipeline script | `workflows/<domain>/` |
| Example/demo | `examples/` |

## CI Architecture Check

The `tools/audit/check_architecture.py` script enforces the dependency rules:

```bash
python tools/audit/check_architecture.py
```

It will fail on any import of forbidden modules from `core/` or `openfoam/backend/`.
