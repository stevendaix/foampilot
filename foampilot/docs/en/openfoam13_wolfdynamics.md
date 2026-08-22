# OpenFOAM 13 integration: Wolf Dynamics and Figshare resources

**Status:** integration plan and porting criteria. This page does not claim that an older case runs on OpenFOAM 13 until it has been executed and checked in an OpenFOAM 13 environment.

## Scope

This page turns the Wolf Dynamics catalogue into a FoamPilot roadmap. It separates teaching material, validation cases and material explicitly published for OpenFOAM 13. External training files remain linked rather than copied: the Figshare record states that the training archives are protected and that unauthorised distribution or duplication is prohibited [1]. FoamPilot should therefore contain original adapters, metadata, case generators and tests, not redistribution of protected slides or archives.

OpenFOAM 13 replaces much of the historical application-solver workflow with `foamRun` and selectable solver modules [2]. Porting an older tutorial requires checking the solver, dictionaries, field names, physical models and utilities rather than merely renaming a command.

## Selection and compatibility matrix

**Exists** means that the module or utility is documented for OpenFOAM 13. **Port required** means that the Wolf Dynamics resource targets an older version or does not specify one; it is not an OpenFOAM 13 validation claim.

| Resource | Source version | OpenFOAM 13 target | FoamPilot value | Initial status |
|---|---:|---|---|---|
| Driven cavity revisited | OF4.1–5.x | `foamRun -solver incompressibleFluid` | Minimal generation, BC, convergence and post-processing case | Module exists; port required |
| Hagen–Poiseuille | OF4.1–5.x | `incompressibleFluid` | Analytical flow-rate and pressure-drop check | Module exists; analytical test needed |
| Dam break VOF | OF4.1–5.x | `incompressibleVoF` | Interface, MULES and time-step regression | Module exists; dictionaries need migration |
| Cylinder vortex shedding | OF4.1–5.x | `incompressibleFluid` | Transient flow and Strouhal-number check | Module exists; quantitative validation needed |
| 3D dam-break validation case | OF7 | `incompressibleVoF` | 3D multiphase regression | OF13 port required |
| High-speed NACA 0012 | OF7 | `shockFluid` or suitable compressible module | Compressible flow and external geometry | Confirm module from case |
| High-lift MDA 30P-30N | OF7 | `incompressibleFluid` or `fluid` | Aerodynamics and integrated forces | High porting cost |
| Wigley Hull free surface | OF7 | `incompressibleVoF` | Free-surface VOF and hull forces | OF13 port required |
| Two Ahmed bodies | OF7 | `incompressibleFluid` | External turbulent aerodynamics and drag | OF13 port required |
| Turbulence training | Not stated | `incompressibleFluid`; `fluid` if compressible | RANS/LES guidance and quality controls | Synthesise support, select cases |
| Multiphase training | Figshare DOI | `incompressibleVoF`, `incompressibleMultiphaseVoF`, `compressibleVoF`, `multiphaseEuler` | High-priority modern multiphase catalogue | Modules documented; inventory cases |
| Dynamic-mesh training | Figshare | `movingMesh` and modules with mesh motion/topology | Motion, AMR, NCC and redistribution | Check each case |
| FVM/discretisation training | Not stated | `fvSchemes`, `fvSolution`, `foamRun` | Numerical-method and practice guide | Guide material, not a case |
| Chemical processes and combustion | **Explicitly OF13** | `XiFluid`, `multicomponentFluid`, `fluid`, `multiphaseEuler` as applicable | Highest priority: directly targets OF13 | Inventory and execute examples |

The official OpenFOAM 13 module guide lists, among others, `incompressibleFluid`, `incompressibleVoF`, `incompressibleMultiphaseVoF`, `compressibleVoF`, `multiphaseEuler`, `fluid`, `XiFluid`, `solid` and `movingMesh` [2]. Module availability alone does not prove that an older tutorial dictionary is compatible.

## Integration strategy: the FoamPilot API

The integration of Wolf Dynamics tutorials into FoamPilot does not use any specific adapter class or external wrapper. Each tutorial is a simple Python script (`run_tutorial.py`) that uses FoamPilot's public `BaseSolver` API. This approach guarantees that:

1. **No external scripts are executed**: The `Allrun` scripts provided in the archives are never called. Execution passes exclusively through the `BaseSolver.run_simulation()` API.
2. **The case is copied and isolated**: The source archive remains untouched. The case is copied into a disposable `.runs/` directory.
3. **Test parameters are controlled**: The script injects reduced `endTime` and `writeInterval` values to numerically validate the startup (smoke test) without consuming the resources of a full simulation.
4. **The completeness contract is respected**: The `validate_generated_case` method checks for the presence of fundamental dictionaries and the declaration of kinematic viscosity (`nu`) for incompressible cases before launching.
5. **Final case data is written by FoamPilot**: Initial and boundary fields (`0/`), physical and chemistry properties (`constant/`, excluding mesh), and numerical dictionaries (`system/`) are read from the source then rewritten through `OpenFOAMDictAddFile`. Each execution creates `foampilot-input-manifest.json`, recording generated input paths, roles and SHA-256 hashes.
6. **Mesh handling is explicit**: When a tutorial supplies `constant/polyMesh`, that topology asset is imported as reference geometry and declared in the manifest. When it requires generation, FoamPilot launches `blockMesh` via the `BaseSolver.run_command()` API; the DamBreak VOF tutorial follows this second path.

## Integration roadmap

**Phase A — catalogue and provenance.** Create a FoamPilot manifest for each selected case: source URL, author, announced version, physical domain, historical solver, OF13 target module, external geometry, licence, run command and verification status. Figshare archives should be treated as sources to consult; only necessary metadata, links and original adaptations should be versioned.

**Phase B — minimal teaching cases.** Port the cavity, Hagen–Poiseuille, cylinder and dam-break cases first. **Note:** The original archives for these beginner tutorials (OF4.1–5.x) are no longer hosted on the Wolf Dynamics website (404 error on public links). FoamPilot integration therefore focuses on rebuilding VOF (DamBreak) cases from existing `incompressibleVoF` models.

**Phase C — validation and advanced models.** Add one representative 3D VOF case, turbulent external-aerodynamics case, compressible case and dynamic-mesh case. Use **validation** only when a reference quantity and reproducible procedure exist; otherwise label the item as a **teaching example** or **structural regression test**.

**Phase D — OF13 chemistry and combustion.** The October 2025 Figshare edition explicitly announces OpenFOAM 13 and provides examples and material on chemical processes, compressible flow, FVM and turbulence [1].

**Cases integrated and validated via the FoamPilot API:**
- **CounterFlow Flame (LTS)**: Counter-flow flame using the `multicomponentFluid` module. The case validates thermodynamic coupling and chemistry. The script was used to run the mesh check (`checkMesh`) and the first 20 iterations of species (O2, H2O, CH4, CO2) and enthalpy resolution.
- **SandiaD Flame (EDC)**: Turbulent modelling of the Sandia D flame with the EDC (Eddy Dissipation Concept) model and `multicomponentFluid`. The case validates the integration of complex chemical properties (reduced GRI30 mechanism) and the resolution of turbulent kinetic energy ($k$) and its dissipation rate ($\omega$).

## OpenFOAM 13 practices required in FoamPilot

> A successful tutorial run is not automatically a physical validation. Wolf Dynamics warns that its beginner tutorials are didactic and should not be used as standards, benchmarks or validation cases [3].

Every generated case must be checked before launch. The minimum contract includes `system/controlDict`, `system/fvSchemes`, `system/fvSolution`, `constant`, `0` and the expected boundary conditions. For incompressible cases, `constant/transportProperties` must explicitly declare `nu`; missing `nu` must produce a clear validation error rather than a partially written case.

Historical calls such as `simpleFoam`, `pimpleFoam`, `interFoam` and `XiFoam` must be documented as legacy references. The OF13 case must use `foamRun` with an appropriate module and a consistent `controlDict`. The official installation guide demonstrates `foamRun` with `incompressibleFluid` for `pitzDailySteady` [4].

For VOF cases, check the MULES controls in `fvSolution` and control the time step using the Courant number. OpenFOAM 13 improves MULES boundedness and provides more structured controls for phase fractions [5]. These changes do not remove the need to check alpha bounds, mass conservation and time-step independence.

For dynamic meshes, use `dynamicMeshDict` to separate motion (`mover`), topology change (`topoChanger`) and redistribution (`distributor`) when required. This reflects the modern OpenFOAM model and replaces assumptions based on a single `dynamicFvMesh` class [6]. Adaptive and parallel cases must check mesh quality after changes and sub-domain balance.

Reports should record the exact `foamVersion`, environment path, FoamPilot commit, executed command, processor count, tolerances and input files. Generated output is not evidence of validity without comparison to an analytical, experimental or published reference.

## PR acceptance criteria

| Criterion | Expected check |
|---|---|
| Version compatibility | Source and OF13 target are stated; older versions are not silently presented as OF13 validated |
| Case completeness | Dictionaries, fields, properties and geometry are present or explicitly externalised |
| FoamPilot generation | Case is generated without manual dictionary edits and is inspectable before launch |
| `nu` contract | Every incompressible configuration writes and checks `constant/transportProperties: nu` |
| Execution | Short case runs with the documented OF13 module, or is marked unexecuted with a reason |
| Numerical quality | `checkMesh`, residuals, Courant, phase bounds and balances are recorded as appropriate |
| Provenance | Links and citations are included; protected training material is not redistributed |
| Reproducibility | Deterministic command and test reproduce the checks |

## References

[1]: https://figshare.com/articles/presentation/Overview_of_Chemical_Processes_with_OpenFOAM_Theory_and_applications/27640866 "Wolf Dynamics/Tonkomo — Overview of Chemical Processes with OpenFOAM: Theory and applications"
[2]: https://doc.cfd.direct/openfoam/user-guide-v13/solvers-modules "OpenFOAM v13 User Guide — Solver modules"
[3]: https://www.wolfdynamics.com/tutorials.html?id=126 "Wolf Dynamics — Getting started with OpenFOAM: Beginner tutorials"
[4]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM 13 — Download for Ubuntu"
[5]: https://openfoam.org/release/13/ "OpenFOAM 13 Released"
[6]: https://cfd.direct/openfoam/free-software/dynamic-meshes/ "CFD Direct — Dynamic meshes in OpenFOAM"
