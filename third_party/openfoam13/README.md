# OpenFOAM 13 external physics catalog

FoamPilot does not copy third-party C++ sources into the core package. The five requested repositories are tracked as optional integrations through `foampilot.openfoam13.PhysicsConfig`; the generated manifest records their provenance and the selected OpenFOAM vendor.

The current Foundation 13-safe baseline is dictionary generation and static validation. Compiling a vendor library remains opt-in because several upstream projects target OpenFOAM ESI releases or older Foundation APIs. This prevents a successful Python installation from silently advertising an ABI-incompatible turbulence or boundary-condition library.

| Integration | Role in FoamPilot | OpenFOAM 13 policy |
|---|---|---|
| ZhangYanTJU/boundaryConditions | Runtime boundary-condition catalogue | Optional external library; user supplies a validated build |
| mthsmcd/MachineLearningTurbulenceModels | ML/RST turbulence models | Optional ESI-oriented library; no automatic Foundation build |
| OpenFOAM-BuildingPhysics/urbanMicroclimateFoam-tutorials | Urban microclimate case conventions | Portable properties and case metadata |
| airshaper/adaptive-mesh-refinement | Field-driven refinement workflow | Portable `dynamicMeshDict` generator |
| argonne-lcf/PythonFOAM | Python/C++ coupling patterns | Optional coupling metadata; no implicit Python ABI link |

Sources: [boundaryConditions](https://github.com/ZhangYanTJU/boundaryConditions), [MachineLearningTurbulenceModels](https://github.com/mthsmcd/MachineLearningTurbulenceModels), [urbanMicroclimateFoam-tutorials](https://github.com/OpenFOAM-BuildingPhysics/urbanMicroclimateFoam-tutorials), [adaptive-mesh-refinement](https://github.com/airshaper/adaptive-mesh-refinement), and [PythonFOAM](https://github.com/argonne-lcf/PythonFOAM).
