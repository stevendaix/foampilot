# [OpenFOAM 13] I built a Python wrapper for OpenFOAM that automates case generation, meshing, and post-processing

**Flair: Showcase (r/Python)**

---

## What My Project Does

**foampilot** is a Python library that wraps OpenFOAM utilities and case files in a Pythonic API. Instead of manually creating dozens of dictionary files (`controlDict`, `fvSchemes`, `fvSolution`, `0/U`, `0/p`, etc.), you configure your entire simulation using Python properties and object-oriented patterns:

```python
from foampilot import Solver

solver = Solver(case_path="./my_case")
solver.transient = True
solver.turbulence_model = "kOmegaSST"
solver.compressible = True
solver.with_gravity = True
solver.boundary.set_condition("inlet", "velocityInlet", velocity=(10, 0, 0))
solver.write_case()
solver.run_simulation(nb_proc=4)
```

It also integrates Gmsh for geometry and meshing (including a direct Gmsh → polyMesh exporter that bypasses `gmshToFoam`), and generates automated reports (LaTeX, Typst, HTML with Plotly) from simulation results.

The project handles:
- Automatic solver selection based on physics flags (compressible, transient, VOF, turbulence model)
- Dynamic field generation (`0/p`, `0/U`, `0/T`, turbulence fields) based on the active physics
- Three meshing backends: `blockMesh`, `gmsh`, `snappyHexMesh` via a unified factory interface
- Direct Gmsh → OpenFOAM polyMesh export (tetrahedra, hexahedra, face orientation, internal face sorting)
- Multi-region CHT cases (`chtMultiRegionFoam`)
- Automated PDF/HTML report generation

---

## Target Audience

This is a **personal / research / open-source project** aimed at:

- CFD engineers and researchers who use OpenFOAM regularly and are tired of manual dictionary editing
- Python developers who want to script OpenFOAM workflows without shell scripts or copy-pasted case directories
- Engineering teams that need reproducible, testable CFD setups (unit-testable case generation is a first-class feature)
- Anyone doing conjugate heat transfer (CHT) with Gmsh who wants to skip the `gmshToFoam` conversion step

It is **not** a replacement for OpenFOAM itself — foampilot generates valid OpenFOAM cases and runs them through the standard solvers. Think of it as a case generator and orchestrator, not a new CFD solver.

---

## Comparison

| Feature | Manual OpenFOAM | foampilot |
|---------|----------------|-----------|
| Case setup | Edit 20+ files by hand | Python properties, ~10 lines |
| Reproducibility | Copy-paste directories, easy to miss files | Git-tracked Python scripts |
| Testing | Impossible | Unit tests for every generated file |
| Meshing | `blockMesh` / Gmsh / snappyHexMesh separately | Unified `Meshing` factory |
| Gmsh → OpenFOAM | `gmshToFoam` (external tool, conversion errors) | Direct `polyMesh` export, no intermediate step |
| Post-processing | manual `postProcess` + ParaView | Automated PDF/HTML reports |
| CHT support | Manual region management | `export_multi_region()` per region |

I deliberately repeated patterns like `write()` across classes instead of over-abstracting. The result is an API that is learnable in ~10 minutes and predictable: every major object has the same interface for persistence.

---

## The Architecture (Full Technical Breakdown)

foampilot is built around three main modules: **Solver**, **Meshing**, and **Report**.

### The Solver: Property-Driven Configuration

The `Solver` class uses Python properties to drive OpenFOAM configuration automatically. Every time you set a property (`solver.transient = True`), `_update_solver()` selects the correct OpenFOAM solver and rebuilds the dependency chain: field manager, boundary conditions, and physical properties.

### The Meshing: Factory for Meshing Strategies

The `Meshing` class unifies access to three backends (`blockMesh`, `gmsh`, `snappyHexMesh`) using the Strategy pattern. Each mesher knows how to write its own files and run its own pipeline.

### The Report: Automated Documentation

The `report` module generates professional documents from simulation results. Three backends: LaTeX/PDF, Typst, and interactive HTML with Plotly.

### Direct Gmsh → polyMesh Export

The `DirectOpenFOAMExporter` writes `points`, `faces`, `owner`, `neighbour`, `boundary`, and `cellZones` directly from the Gmsh Python API. It handles tetrahedra and hexahedra, face orientation via geometric centroid-normal checks, internal face sorting by `(owner, neighbour)` with cyclic rotation for OpenFOAM's upper-triangular ordering, unused point compaction per region, and multi-region meshes for CHT.

### OpenFOAMFile: The Python → OpenFOAM Translator

Every dictionary goes through `OpenFOAMFile`, which handles `FoamFile` headers, nested blocks, dimensions, unit conversion, and syntax formatting (`True` → `true`, floats → 15 significant digits).

---

## Key Design Decision: Intentional Repetition

The most counter-intuitive aspect of foampilot: I deliberately repeated the `write()` method across 8+ classes (`OpenFOAMFile`, `SystemDirectory`, `ConstantDirectory`, `Meshing`, `BlockMesher`, `GmshMesher`, `SnappyMesher`, `ChtSolver`). This is not a flaw — it is an interface contract (Command Pattern) that makes the API predictable and composable.

Same logic applies to `Boundary.apply_condition_with_wildcard()`: one signature for every boundary condition type.

---

## Project Communication & Adoption

Beyond the code:
- **README**: Written as a promise, not a manual. Features with action verbs, "What foampilot is not" section to avoid misunderstandings.
- **Multilingual READMEs**: English, French, Chinese — OpenFOAM is used worldwide, and the English barrier is real.
- **Examples directory**: Complete, runnable cases (electronics CHT, heated channel, aeroacoustics) as "living sales arguments."
- **MkDocs documentation**: Navigable by both new users and contributors.
- **Tests**: Unit tests in `test/` with documented commands — proof of rigor in the scientific world.

---

## Resources

- **GitHub**: https://github.com/stevendaix/foampilot
- **Documentation**: https://stevendaix.github.io/foampilot/
- **License**: MIT

---

*I'm the author of foampilot. I built this because I spent years editing OpenFOAM dictionaries by hand and wanted Python to handle the boilerplate. Happy to answer questions about the architecture, the direct exporter, or the design choices here.*
