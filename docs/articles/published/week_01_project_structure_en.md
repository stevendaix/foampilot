# I Built a Python Wrapper for OpenFOAM: Architecture, Design, and Lessons from an Open-Source Project

*A first-person account spanning software architecture, fluid mechanics, and scientific project communication.*

---

## Introduction: The Engineer Who Was Tired of Dictionaries

I am a CFD engineer. If you have ever used OpenFOAM, you know exactly what I mean: that strange mix of raw power and cognitive friction. You set up your case, everything is ready, and then… you have to edit `controlDict`. Then `fvSchemes`. Then `fvSolution`. Then every field in `0/`. And if you forget a semicolon, the run crashes.

After years of copying directories and editing text files by hand, I eventually asked myself: *what if Python could do this work for me?*

That frustration is what gave birth to **foampilot**. But beyond the code itself, this project taught me two essential things:

1. **Scientific tool design is about reducing cognitive load, not minimizing code.**
2. **An open-source project lives not only by its publication — it lives by its tool.**

In this article, I take you behind the scenes of foampilot: how I designed the architecture, why I made counter-intuitive design choices, and how I structured the project so that it would be adopted — not just used.

---

## Part 1: The Problem — Why OpenFOAM Deserves a Wrapper

OpenFOAM is a remarkable toolbox. But its dictionary-based configuration model, while powerful, creates three chronic problems:

### 1.1 Tedious repetition

Every simulation requires creating **dozens of files**:
- `system/controlDict`, `fvSchemes`, `fvSolution`, `decomposeParDict`
- `constant/transportProperties`, `turbulenceProperties`, `physicalProperties`
- `0/U`, `0/p`, `0/k`, `0/epsilon`, `0/nut`, `0/T`, etc.

Each file follows strict syntax: blocks, semicolons, quotes, `FoamFile` headers. A single typo is enough to break `blockMesh` or `foamRun`.

### 1.2 Poor reproducibility

In a manual workflow, reproducing a study means copying an entire directory, hoping no hidden file was missed, then manually editing the changed parameters. The result: two "identical" simulations can produce different results because one file was modified by hand between them.

### 1.3 No tests

When cases are built manually, you cannot write a unit test that verifies the consistency of the generated files. The smallest regression goes unnoticed until execution time.

**The solution?** Turn OpenFOAM into a **case generator**, not a file editor. That is exactly what foampilot does.

---

## Part 2: Architecture — How foampilot Orchestrates OpenFOAM

foampilot is built around three main modules: **Solver**, **Meshing**, and **Report**. Each plays an orchestrating role, and their interaction forms a clean, extensible layered architecture.

### 2.1 The Solver: property-driven configuration

The heart of foampilot is the `Solver` class (`foampilot/solver/solver.py`). Instead of asking users to select a solver and edit files, it uses **Python properties** to drive configuration automatically.

Here is a concrete excerpt:

```python
# foampilot/solver/solver.py
class Solver:
    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)
        self._solver: Optional[BaseSolver] = None
        self._compressible = False
        self._with_gravity = False
        self._is_vof = False
        self._is_solid = False
        self._transient = False
        self._turbulence_model = "kEpsilon"
        self._update_solver()

    @property
    def compressible(self) -> bool:
        return self._compressible

    @compressible.setter
    def compressible(self, value: bool):
        self._compressible = value
        self._update_solver()
```

Every time you modify a property (`solver.compressible = True`, `solver.transient = True`, etc.), `_update_solver()` is called. It automatically selects the correct OpenFOAM solver and rebuilds the entire dependency chain: field manager, boundary conditions, and physical properties.

**The effect for users?** A declarative configuration that looks like standard Python:

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

Everything is generated automatically. The `0/`, `system/`, and `constant/` directories are created without a single dictionary being edited by hand.

### 2.2 Meshing: a factory for meshing strategies

The `Meshing` class (`foampilot/base/meshing.py`) is a **factory** that unifies access to three supported meshing backends: `blockMesh`, `gmsh`, and `snappyHexMesh`.

```python
from foampilot.base import Meshing

meshing = Meshing(case_path="./my_case", mesher="gmsh")
meshing.add_file("surfaceFeatureExtractDict", {...})
meshing.write()
```

Behind this uniform interface, each mesher knows how to write its own files and run its own pipeline. The factory simply chooses the appropriate implementation — the **Strategy pattern** applied to CFD.

### 2.3 Report: from post-processing to automated PDF

The `report` module (`foampilot/report/report_generator.py`) completes the chain by generating professional documents from simulation results. It supports three backends:

- **LaTeX/PDF** via `LatexDocument`
- **Typst** via `ScientificDocument`
- **Interactive HTML** with Plotly

Same data flow, three output formats. The user chooses the format without changing any code.

---

## Part 3: Simplicity Through Abstraction — How to Hide Complexity Without Losing It

The greatest challenge of an OpenFOAM wrapper is to **generate valid files** without preventing users from accessing OpenFOAM's raw power when they need it.

### 3.1 OpenFOAMFile: the Python → OpenFOAM translator

Every OpenFOAM dictionary in foampilot goes through the `OpenFOAMFile` class (`foampilot/base/openFOAMFile.py`). It knows OpenFOAM syntax: `FoamFile` headers, nested blocks, lists, dimensions, and even unit conversion.

Here is what it does for you:

```python
# foampilot/base/openFOAMFile.py
class OpenFOAMFile:
    DEFAULT_UNITS = {
        "nu": "m^2/s", "mu": "Pa.s", "rho": "kg/m^3",
        "k": "m^2/s^2", "epsilon": "m^2/s^3", "omega": "1/s",
        "U": "m/s", "p": "Pa", "T": "K",
    }

    def _format_value(self, key: str, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return format(value, ".15g")
        if ValueWithUnit and isinstance(value, ValueWithUnit):
            unit = self.DEFAULT_UNITS.get(key)
            val = value.get_in(unit) if unit else value.magnitude
            return f'{val:.15g}'
        return str(value)
```

`True` becomes `true`. Floats are formatted to 15 significant digits. A `ValueWithUnit(10, "m/s")` becomes `10` in the right context. Developers no longer need to worry about OpenFOAM syntax.

### 3.2 CaseFieldsManager: dynamic configuration logic

The `CaseFieldsManager` class (`foampilot/base/cases_variables.py`) determines which initial fields are needed based on the physics:

```python
# foampilot/base/cases_variables.py
def _generate_fields(self) -> None:
    self.fields.clear()
    pressure_name = "p_rgh" if self.with_gravity and not self.compressible else "p"
    self.fields[pressure_name] = {"value": ValueWithUnit(0, "Pa")}
    if not self.is_solid:
        self.fields["U"] = {"value": ValueWithUnit(0, "m/s")}
    if self.energy_activated or self.compressible:
        self.fields["T"] = {"value": ValueWithUnit(300, "K")}
    if self.turbulence_model:
        self._generate_turbulence_fields()
```

Five lines of logic replace the manual creation of 3 to 6 `0/` files with the correct names, dimensions, and default values. For CHT, `_generate_region_fields()` extends this logic per region: solids receive only `T`, while fluids receive `U`, `p`, `T`, and turbulence fields.

---

## Part 4: The Key Design Pattern — Intentional Repetition

This is the most counter-intuitive aspect of foampilot: **I deliberately repeated code patterns** where most architects would try to factorize.

### 4.1 The universal `write()`

In foampilot, almost every major object exposes a `write()` method with the same name:

| Class | What `write()` does |
|---|---|
| `OpenFOAMFile` | Writes a dictionary file |
| `SystemDirectory` | Writes `controlDict`, `fvSchemes`, `fvSolution` |
| `ConstantDirectory` | Writes `transportProperties`, `turbulenceProperties`, etc. |
| `Meshing` | Writes `blockMeshDict`/`snappyHexMeshDict` + system files |
| `BlockMesher` | Writes `blockMeshDict` |
| `GmshMesher` | Exports the mesh to OpenFOAM |
| `SnappyMesher` | Writes `snappyHexMeshDict` |
| `ChtSolver` | Writes all region-specific files |

This uniformity has a profound effect: **you never have to wonder "how do I write this object?"**. The answer is always the same.

```python
for component in [solver.system, solver.constant, meshing]:
    component.write()
```

In object-oriented programming, this is the **Command Pattern** in its purest form: each object encapsulates the action of persisting itself. The repetition of `write()` is not a flaw — it is an **interface contract** that makes the API predictable.

### 4.2 `apply_condition_with_wildcard`: a single signature

The `Boundary` class (`foampilot/boundaries/boundaries_dict.py`) exposes a single method for applying boundary conditions:

```python
def apply_condition_with_wildcard(self, pattern: str, condition_type: str, **kwargs):
```

Whether you apply a `velocityInlet`, `pressureOutlet`, or `wall` with friction, the entry point is always the same: a regex pattern, a type, and keyword arguments.

### 4.3 Why repetition is a virtue here

In most projects, code duplication is an anti-pattern. In foampilot, it serves three purposes:

1. **Discoverability**: When every object has `write()`, you do not need to consult documentation to find the right method name.
2. **Composability**: Uniform interfaces enable generic algorithms.
3. **Reduced cognitive load**: A CFD engineer already has much to remember. An API where everything works the same way eliminates one more mental variable.

I like to say that foampilot **repeats to simplify**. Factorization at all costs creates invisible abstractions; intentional repetition creates an interface **learnable in 10 minutes**.

---

## Part 5: The Technical Demo — Direct Gmsh → polyMesh Export

If I had to name one technical achievement of this project, it would be the **direct export from Gmsh to OpenFOAM's polyMesh format**, without going through `gmshToFoam`.

The `direct_openfoam_exporter.py` module directly writes `points`, `faces`, `owner`, `neighbour`, `boundary`, and `cellZones` by querying the Gmsh Python API. It handles:

- Tetrahedra (type 4) and hexahedra (type 5)
- Face orientation via geometric centroid-normal checks
- Internal face sorting by `(owner, neighbour)` with cyclic rotation for OpenFOAM's upper-triangular ordering
- Unused point compaction per region
- Multi-region meshes for CHT

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("case")
# ... geometry construction, meshing ...
exporter = DirectOpenFOAMExporter("/path/to/case")
exporter.export_single_region()  # or export_multi_region() for CHT
gmsh.finalize()
```

This feature is not just a convenience: it removes an external dependency, eliminates conversion errors, and gives full control over the meshing pipeline.

---

## Part 6: How to Present a Scientific Open-Source Project

Building the tool is only half the work. The other half is **making sure others use it**.

### 6.1 A README as a promise, not documentation

The foampilot README is not a user manual. It is a **promise**: a clear title, an elevator pitch, features written with action verbs, and a "What foampilot is not" section to avoid misunderstandings. The goal: within 30 seconds, a visitor knows whether the tool is meant for them.

### 6.2 Multilingual documentation as an adoption strategy

The README exists in three versions: English, French, and Chinese. This is not accidental. OpenFOAM is used worldwide, but many users still face the English technical-language barrier. A multilingual README sends a strong message: *"this project is made for you, whatever your language."*

### 6.3 Examples as sales arguments

The `examples/` directory contains complete cases: electronics CHT, heated channel, aeroacoustics, and more. Each example is a **living sales argument**. An engineer who wonders *"can foampilot handle my CHT case?"* can clone the repo and run `examples/cht/` in 5 minutes.

### 6.4 Technical documentation as project memory

The documentation site, built with MkDocs, serves two audiences: **new users** who want to get started, and **contributors** who want to understand the architecture. A good open-source project must be navigable by both.

### 6.5 Tests as proof of maturity

The `test/` directory and documented test commands send an important signal: *"this project is tested, so it is reliable."* In the scientific world, where reproducibility is king, tests are not a luxury — they are proof of rigor.

---

## Part 7: Conclusion — The Tool as a Communication Product

If I had to summarize what foampilot taught me, it would be this:

**A scientific open-source project is a product with two faces:**
- **The technical face**: code, algorithms, reference documentation.
- **The communication face**: the README, examples, user documentation, demos.

Most researchers excel at the first face. Few invest enough in the second.

Building foampilot forced me to think like an engineer *and* like a product manager. Every `write()` method is a design choice. Every example in `examples/` is an adoption argument. Every line in the README is a promise kept — or broken.

If you are a researcher or engineer hesitating to open-source your tool, know this: **your tool deserves to be used, but it must also be understandable**. Invest in presentation as much as code. Write examples. Document concepts. Make tutorials.

And if you have spent too much time editing OpenFOAM dictionaries by hand… maybe your next project should start with `pip install foampilot`.

---

**Resources:**
- GitHub repo: [https://github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation: [https://stevendaix.github.io/foampilot/](https://stevendaix.github.io/foampilot/)
- License: MIT

---

*Article ready for publication — version 1.0*
