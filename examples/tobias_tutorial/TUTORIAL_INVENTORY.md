# Tobias Holzmann tutorial inventory

This inventory tracks every case directory currently present in `shor-ty/OpenFOAMTutorials`. A case is **validated** only when its dedicated FoamPilot `run.py` completes the intended workflow with OpenFOAM 13 and a detailed `report.md` is present.

| Source family | Source case | FoamPilot directory | Status |
| --- | --- | --- | --- |
| openfoam.com | 2dArbitraryMeshInterface | — | Pending |
| openfoam.com | 45DegreePipeWithLayers | `meshing_pipe_45deg` | Validated equivalent |
| openfoam.com | 90DegreePipeWithLayers | `meshing_pipe_90deg` | Validated equivalent |
| openfoam.com | ACMIHeatTransfer | — | Pending |
| openfoam.com | arbitraryRotatingInletACMI | — | Pending |
| openfoam.com | arbitraryWaterPump | — | Pending |
| openfoam.com | cellZoneGeneration | `cell_zone_generation` | Validated equivalent |
| openfoam.com | combustionChamber | — | Pending |
| openfoam.com | complexMeshMotion | — | Pending |
| openfoam.com | complexMeshMotionACMI | — | Pending |
| openfoam.com | dakotaGeometricVariation | — | Pending; external Dakota dependency |
| openfoam.com | dakotaTeslaOneWayValve2D | — | Pending; external Dakota dependency |
| openfoam.com | fallingDroplets | `falling_droplets` | Validated equivalent |
| openfoam.com | fanRotationAndAMI | — | Pending |
| openfoam.com | fluentMeshForCHTSolver | — | Pending |
| openfoam.com | ginTonicCHT | — | Pending |
| openfoam.com | magnusEffect | `magnus_effect` | Validated equivalent |
| openfoam.com | snappySphereAndLayer | — | Pending |
| openfoam.com | solarChimney | — | Pending |
| openfoam.com | suzannesHead | — | Pending |
| openfoam.com | tankWithSavetyValve | — | Pending |
| openfoam.com | thinGapMeshing | — | Pending |
| openfoam.org | 2dArbitraryMeshInterfaceNCC | `2d_ami_ncc` | Validated |
| openfoam.org | 2dAxisSymmetricMeshing | `2d_rotational_axis_symmetric` | Validated equivalent |
| openfoam.org | 45DegreePipeWithLayers | `meshing_pipe_45deg` | Validated |
| openfoam.org | 90DegreePipeWithLayers | `meshing_pipe_90deg` | Validated |
| openfoam.org | NCCHeatTransfer | `NCCHeatTransfer` | Validated short dynamic mesh calculation |
| openfoam.org | adaptiveMeshRefinement | `adaptive_mesh_refinement` | Validated |
| openfoam.org | arbitraryRotatingInletNCC | — | Pending |
| openfoam.org | arbitraryWaterPump | — | Pending |
| openfoam.org | batteryCooling | `battery_cooling` | Validated short thermo-fluid calculation |
| openfoam.org | catalystHeatUp | `catalystHeatUp` | Validated short multi-region CHT calculation |
| openfoam.org | cellZoneGeneration | `cell_zone_generation` | Validated equivalent |
| openfoam.org | combustionChamber | `combustion_chamber` | Validated short calculation |
| openfoam.org | dakotaGeometricVariation | — | Pending; external Dakota dependency |
| openfoam.org | dakotaTeslaOneWayValve2D | `dakotaTeslaOneWayValve2D` | Validated optimization workflow |
| openfoam.org | fallingDroplets | `falling_droplets` | Validated equivalent |
| openfoam.org | fanRotationAndNCC | — | Pending |
| openfoam.org | fluentMeshForCHTSolver | — | Pending |
| openfoam.org | fluidicOscillator | `fluidic_oscillator` | Validated |
| openfoam.org | ginTonicCHT | — | Pending |
| openfoam.org | kaplanTurbineNCC | — | Pending |
| openfoam.org | magnusEffect | `magnus_effect` | Validated equivalent |
| openfoam.org | meshingAHelix | — | Pending |
| openfoam.org | pitotTube variants | `pitot_tube` | Validated short calculation representative |
| openfoam.org | rotatingRotorNCC | `rotatingRotorNCC` | Validated short dynamic mesh calculation |
| openfoam.org | snappyFeatureEdgeRefinement | `snappy_feature_edge_refinement` | Validated meshing workflow |
| openfoam.org | snappyHexMeshCellZoneMeshing | — | Pending |
| openfoam.org | snappySphereAndLayer | `snappy_sphere_and_layer` | Validated meshing workflow |
| openfoam.org | sneezingSimulation | — | Pending |
| openfoam.org | solarChimney | — | Pending |
| openfoam.org | suzannesHead | — | Pending |
| openfoam.org | thinGapMeshing | `thin_gap_meshing` | Validated short calculation |
| openfoam.org | verticalAxialWindTurbineNCC | — | Pending |

The inventory contains **53 source case directories**. Several entries are corresponding `.org` and `.com` versions of the same tutorial; they remain listed separately because their dictionaries, solvers, boundary conditions or utilities may differ and must be checked before consolidating a port.

The next step is an API audit: every required operation will be matched against existing FoamPilot methods before any shared code is changed. Case-local OpenFOAM syntax adaptations do not automatically justify an API extension.
