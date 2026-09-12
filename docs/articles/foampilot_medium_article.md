# From Dictionaries to Code: How foampilot Transforms CFD Engineering

*An in-depth analysis of an open-source Python platform that turns OpenFOAM from a manual craft into a modern, testable software workflow.*

---

## Introduction: The CFD Bottleneck

Computational Fluid Dynamics (CFD) is one of the most powerful tools in an engineer's arsenal. OpenFOAM, as the leading open-source CFD toolbox, offers unparalleled flexibility — but that flexibility comes at a steep cost. Every simulation requires painstaking manual editing of dozens of dictionary files (`controlDict`, `fvSchemes`, `fvSolution`, `0/U`, `0/p`, boundary fields, transport properties…). A single typo can crash an entire run, and reproducing a study means copying directories by hand and hoping nothing was missed.

This is where **foampilot** enters the scene. Born from the frustration of repetitive OpenFOAM workflows, foampilot is a Python platform that *fully orchestrates* OpenFOAM simulations — from geometry and meshing to solver execution, post-processing, and report generation. But beyond its utility, foampilot is a masterclass in software architecture: it demonstrates how intentional repetition, a universal interface, and property-driven configuration can transform a chaotic manual process into an elegant, predictable API.

In this article, we will dissect the foampilot codebase to reveal the design principles that make it a standout project in the CFD Python ecosystem.

---

## Part 1: Architecture & Orchestration — The Three Pillars

At its core, foampilot is built around three orchestrating modules: **Solver**, **Meshing**, and **Report**. Understanding how they interact reveals a clean, layered architecture inspired by modern software design.

### 1.1 The Solver: Property-Driven Configuration

The `Solver` class (`foampilot/solver/solver.py`) is the central brain of a simulation. Rather than asking the user to manually select solvers and edit configuration files, foampilot uses a **property-driven approach**. Setting `solver.compressible = True` automatically selects the correct solver module, updates the field manager, and prepares the boundary condition infrastructure.

```python
# foampilot — Solver configuration
solver = Solver(case_path="./my_case")
solver.transient = True
solver.turbulence_model = "kOmegaSST"
solver.compressible = True
solver.boundary.set_condition("inlet", "velocityInlet", velocity=(10, 0, 0))
solver.write_case()
solver.run_simulation(nb_proc=4)
```

Behind the scenes, every property setter triggers `_update_solver()`, which instantiates the appropriate `BaseSolver` subclass (e.g., `ChtSolver` for conjugate heat transfer) and rebuilds the boundary manager. The `BaseSolver` class (`foampilot/solver/base_solver.py`) contains all the common logic — `run_simulation()`, `run_parallel()`, `write_case()` — while specialized solvers like `ChtSolver` override only what is necessary for multi-region physics.

### 1.2 The Meshing Factory

The `Meshing` class (`foampilot/base/meshing.py`) acts as a **factory and orchestrator** for different meshing strategies. Whether you prefer the classic `blockMesh`, the flexibility of Gmsh, or the power of `snappyHexMesh`, the interface remains identical:

```python
meshing = Meshing(case_path="./my_case", mesher="gmsh")
meshing.add_file("surfaceFeatureExtractDict", {...})
meshing.write()
```

The factory pattern hides the complexity of initializing `BlockMesher`, `GmshMesher`, or `SnappyMesher`. Each mesher knows how to `write()` its own dictionary files and `run()` its own pipeline. The `Meshing` manager simply delegates.

### 1.3 The Report Generator

Post-processing is handled by the `CFDReportGenerator` (`foampilot/report/report_generator.py`), which aggregates statistics, figures, and tables into professional outputs. It supports multiple backends:

- **LaTeX / PDF** via `LatexDocument`
- **Typst** via `ScientificDocument` and `TypstRenderer`
- **Interactive HTML** with embedded Plotly visualizations

This means the same simulation data can produce a publication-ready PDF, a lightweight HTML dashboard, or a Typst document — all from a single, unified API.

---

## Part 2: Simplicity Through Abstraction

The most impressive aspect of foampilot is how it **masks the complexity** of OpenFOAM configuration files while preserving full control.

### 2.1 The OpenFOAMFile Base Class

Every OpenFOAM dictionary in foampilot inherits from or uses the `OpenFOAMFile` class (`foampilot/base/openFOAMFile.py`). This class understands OpenFOAM syntax: headers, nested blocks, lists, and even unit-aware physical quantities. It handles the formatting of Python types into OpenFOAM-compatible strings — booleans become `true`/`false`, floats get high-precision formatting, and `ValueWithUnit` objects automatically convert to the correct units (e.g., `Pa.s` for viscosity).

```python
# Under the hood — what foampilot writes for you
foam_file = OpenFOAMFile(
    object_name="controlDict",
    application="simpleFoam",
    startFrom="startTime",
    startTime=0,
    endTime=1000,
    deltaT=1,
    writeControl="timeStep",
    writeInterval=100,
    purgeWrite=0,
    writeFormat="ascii",
    writePrecision=6,
    writeCompression="off",
    timeFormat="general",
    graphFormat="raw",
    runTimeModifiable="true"
)
foam_file.write_file(case_path / "system" / "controlDict")
```

Compare this to manually editing `controlDict` — dozens of lines of keyword-value pairs, semicolons, and braces that must be perfectly formatted. foampilot generates it from a clean Python dictionary.

### 2.2 Dynamic Field Generation

The `CaseFieldsManager` (`foampilot/base/cases_variables.py`) is a brilliant piece of logic. It inspects the physical configuration of the simulation and automatically determines which initial field files are needed. Enable compressibility? Add `p` instead of `p_rgh`. Enable energy? Add `T`. Enable turbulence? Add `k`, `epsilon`, `omega`, or `nut` depending on the model.

```python
# One line replaces dozens of manual file edits
solver = Solver(case_path="./case")
solver.compressible = True
solver.energy_activated = True
solver.turbulence_model = "kOmegaSST"
solver.setup_case()  # Creates all required 0/ files automatically
```

For conjugate heat transfer (CHT), the `ChtSolver` extends this further. Each `FluidRegion` and `SolidRegion` gets its own field set — fluids receive velocity, pressure, and turbulence fields, while solids receive only temperature. The `_generate_region_fields()` method handles this combinatorial logic without any user intervention.

### 2.3 Boundary Conditions Made Declarative

OpenFOAM boundary conditions are notoriously verbose. A single patch might require entries in `U`, `p`, `k`, `epsilon`, `nut`, `T`, and more. foampilot's `Boundary` class (`foampilot/boundaries/boundaries_dict.py`) replaces all of this with a single, declarative call:

```python
solver.boundary.apply_condition_with_wildcard("inlet", "velocityInlet", velocity=(10, 0, 0))
solver.boundary.apply_condition_with_wildcard("outlet", "pressureOutlet")
solver.boundary.write_boundary_conditions()
```

The `apply_condition_with_wildcard` method uses regex patterns to match boundary names, applies the correct condition type across all required fields, and resolves default parameters automatically. What once required editing 6+ files is now a 2-line Python script.

---

## Part 3: The Repetition Principle — A Positive Design Pattern

Here lies the most insightful architectural choice in foampilot: **intentional, pervasive repetition**. In modern software engineering, we are often taught to eliminate duplication at all costs. But foampilot embraces a different philosophy: *strategic repetition reduces cognitive load*.

### 3.1 The Universal `write()` Method

Every major object in foampilot implements a `write()` method with the same signature:

| Object | What `write()` does |
|--------|---------------------|
| `OpenFOAMFile` | Writes a single dictionary file |
| `SystemDirectory` | Writes `controlDict`, `fvSchemes`, `fvSolution` |
| `ConstantDirectory` | Writes `transportProperties`, `turbulenceProperties`, etc. |
| `Meshing` | Writes `blockMeshDict`, `snappyHexMeshDict`, plus system files |
| `BlockMesher` | Writes `blockMeshDict` |
| `GmshMesher` | Exports mesh to OpenFOAM |
| `SnappyMesher` | Writes `snappyHexMeshDict` |
| `ChtSolver` | Writes all region-specific files |

This uniformity means you never have to ask "how do I write this object?" The answer is always the same: `obj.write()`. You can iterate over a collection of objects and write them all:

```python
for component in [solver.system, solver.constant, meshing]:
    component.write()
```

This is the **Command Pattern** in its simplest form — every object encapsulates the action of persisting itself to disk. The repetition of the `write()` name across the entire codebase is not a code smell; it is a deliberate design decision that makes the API predictable and self-documenting.

### 3.2 The Constant Signature of `apply_condition_with_wildcard`

Another example of intentional repetition is the `Boundary` class's method signature:

```python
def apply_condition_with_wildcard(self, pattern: str, condition_type: str, **kwargs):
```

Whether you are applying a `velocityInlet`, a `pressureOutlet`, a `wall`, or a custom CSV-driven condition, the entry point is always the same: a pattern, a type, and keyword arguments. This consistency means engineers can learn one method and apply it everywhere.

### 3.3 Why Repetition Here Is a Virtue

In foampilot, repetition serves three purposes:

1. **Discoverability**: When every object has `write()`, you never have to hunt through documentation to find the right method name.
2. **Composability**: Uniform interfaces allow generic algorithms. You can write a `write_all()` function that works on any combination of objects because they all share the same protocol.
3. **Reduced Cognitive Load**: Engineers working with OpenFOAM already carry a heavy mental burden. A consistent API eliminates one more variable to track.

This is the opposite of the "abstraction at all costs" anti-pattern. foampilot repeats `write()` not because it couldn't abstract it further, but because *predictability is more valuable than brevity*.

---

## Part 4: From Manual Craft to Modern Software

The ultimate achievement of foampilot is transforming CFD from a **manual, error-prone craft** into a **reproducible, testable, and version-controlled software workflow**.

### 4.1 Reproducibility by Design

A foampilot workflow is a Python script. It can be committed to Git, reviewed in pull requests, and executed on any machine with the same dependencies. The case is *generated* — never manually maintained. This means:

- **No more "it works on my machine"** — the exact same script generates the exact same case.
- **Parametric studies are trivial** — loop over parameters in Python instead of manually editing files.
- **Continuous Integration is possible** — a CFD simulation can be part of an automated test suite.

### 4.2 Testability

Because foampilot generates cases programmatically, you can write unit tests that verify the generated OpenFOAM files:

```python
def test_controlDict_generation():
    solver = Solver(case_path="./test_case")
    solver.transient = True
    solver.write_case()
    content = (Path("./test_case/system/controlDict")).read_text()
    assert "application" in content
    assert "deltaT" in content
```

This level of testing is impossible when files are edited by hand.

### 4.3 The Gmsh Direct Export

One of the most technically impressive features is the `DirectOpenFOAMExporter` (`foampilot/mesh/direct_openfoam_exporter.py`). This module bypasses the traditional `gmshToFoam` utility entirely, writing `constant/polyMesh` (points, faces, owner, neighbour, boundary, cellZones) directly from the Gmsh Python API. It handles:

- Tetrahedron and hexahedron cell types
- Automatic face orientation using geometric centroid-normal checks
- Internal face sorting by (owner, neighbour) with cyclic rotation for OpenFOAM's upper-triangular ordering
- Unused point compaction per region
- Multi-region CHT meshes

This eliminates an external dependency, reduces conversion errors, and gives the user full programmatic control over the mesh export pipeline.

---

## Conclusion: Engineering Elegance

foampilot is more than a wrapper around OpenFOAM utilities. It is a carefully designed Python platform that respects both the power of CFD and the principles of good software engineering.

Its architecture — with `Solver`, `Meshing`, and `Report` as orchestrating modules — provides a clean separation of concerns. Its use of `OpenFOAMFile` as a universal abstraction makes configuration declarative and type-safe. And its embrace of **intentional repetition** — the universal `write()`, the consistent `apply_condition_with_wildcard` signature — proves that sometimes the best way to reduce complexity is to make the API *predictable* rather than *minimal*.

For engineers and researchers tired of manually editing dictionaries, foampilot offers a path forward. It transforms CFD from a craft mastered through years of painful repetition into a modern, scriptable, and maintainable engineering discipline. In doing so, it sets a new standard for what an open-source CFD platform can be.

---

*foampilot is released under the MIT License. Full documentation: https://stevendaix.github.io/foampilot/*
