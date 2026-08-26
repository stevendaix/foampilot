# Matrice d’intégration des tutoriels OpenFOAM 13 dans Foampilot

> Règle de traitement : les cas sont traités strictement un par un. Un cas n’est marqué « validé » qu’après reproduction de sa mise en données avec des appels Foampilot uniquement, exécution avec OpenFOAM 13 et vérification des fichiers générés et des résultats.

| Ordre | Famille | Tutoriel OpenFOAM 13 | Chemin source | Équivalent Foampilot | Statut | Fonctions Foampilot ajoutées | Preuves / remarques |
|---:|---|---|---|---|---|---|---|
| 1 | `incompressibleFluid` | `cavity` | `/opt/openfoam13/tutorials/incompressibleFluid/cavity` | `01_cavity_laminar` | Validé | `Meshing`, `BlockMesher.write`, `BlockMesher.run`, `Solver.setup_case`, `Boundary.set_raw_condition`, `Solver.run_simulation` | Maillage 20x20x1, cavité 0,1 m x 0,1 m x 0,01 m, movingWall à 1 m/s, calcul jusqu’à t=1 s sans erreur fatale. |
| 2 | `incompressibleFluid` | `drivaerFastback` | `/opt/openfoam13/tutorials/incompressibleFluid/drivaerFastback` | `02_simpleCar_turbulent` | Validé | `Meshing`, `BlockMesher.write`, `SnappyMesher.write_surface_features_dict`, `SnappyMesher.add_feature`, `SnappyMesher.run_surface_features`, `SnappyMesher.run`, `Solver.setup_case`, `Boundary.set_raw_condition`, `Solver.run_simulation` | Maillage blockMesh + snappyHexMesh, extraction de features v13, kOmegaSST, calcul jusqu’à 300 itérations et résidus exportés sans erreur fatale. |
| 3 | `fluid` | `cavity` | `/opt/openfoam13/tutorials/fluid/cavity` | `14_fluid_cavity` | Validé OF13 — `End=1 s` | `SystemDirectory.import_reference_file`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Mise en données thermo-compressible reproduite avec FoamPilot : `hePsiThermo`, `kOmegaSST`, `thermophysicalTransport`, blockMesh et `foamRun -solver fluid` jusqu’à `End=1 s` sans `FOAM FATAL`. |
| 4 | `fluid` | `pitzDaily` | `/opt/openfoam13/tutorials/fluid/pitzDaily` | `03_pitzDaily_step` | Validation longue partielle | `Solver.solver_name`, `CaseFieldsManager.register_field`, `CaseFieldsManager.custom_initial_values`, `Boundary.set_patch_type`, `Boundary.write_boundary_conditions` avec overrides, `PhysicalPropertiesFile.configure_reference`, support LES `kEqn`, `BlockMesher` sans `defaultPatch` vide, `FvSolutionFile` module `fluid` | Mise en données source reproduite ; `foamRun -solver fluid` atteint t=0,22589 s sans erreur fatale (≈639 s). Le délai d’exécution de l’environnement a interrompu avant t=0,3 s ; aucune divergence observée. |
| 5 | `XiFluid` | `engine2Valve2D` | `/opt/openfoam13/tutorials/XiFluid/engine2Valve2D` | `11_XiFluid_engine2Valve2D` | Validé OF13 — `End=3600 CAD` | `SystemDirectory.run_utility`; `update_dictionary_entries`; `rename_dictionary_entries`; `remove_dictionary_entries`; `BlockMesher.copy_mesh`; `BlockMesher.write_mesh_times`; `CaseFieldsManager.import_reference_field` | Pipeline `Allmesh` reproduit avec FoamPilot : 24 meshes temporels, renommages de patches, baffles, transformations, création des couples non conformes et calcul `foamRun -solver XiFluid` jusqu’à `End` sans `FOAM FATAL` ni `SIGFPE`. Référence et validation exclusivement OpenFOAM 13.`
| 6 | `XiFluid` | `moriyoshiHomogeneous` | `/opt/openfoam13/tutorials/XiFluid/moriyoshiHomogeneous/moriyoshiHomogeneous` | `12_XiFluid_moriyoshiHomogeneous` | Validé OF13 — propane `End=0.015 s`, hydrogène `End=0.005 s` | `SystemDirectory.update_dictionary_entries`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Deux variantes reproduites uniquement par FoamPilot : cas propane et variante hydrogène avec remplacement des propriétés thermophysiques/combustion, maillage blockMesh et exécution `foamRun -solver XiFluid` sans `FOAM FATAL` jusqu’aux temps de fin de la référence.`
| 7 | `XiFluid` | `stratified` | `/opt/openfoam13/tutorials/XiFluid/stratified` | `13_XiFluid_stratified` | Validé OF13 — `End=0.04 s` | `SystemDirectory.import_reference_file`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Mise en données OF13 reproduite avec FoamPilot : blockMesh, setFields et `foamRun -solver XiFluid` jusqu’à `End=0.04 s` sans `FOAM FATAL`. La référence utilise l’ignition `constantbXiIgnition` et les champs stratifiés `ft/fu/egr`.
| 8 | `compressibleMultiphaseVoF` | `damBreak4phaseLaminar` | `/opt/openfoam13/tutorials/compressibleMultiphaseVoF/damBreak4phaseLaminar` | `15_compressibleMultiphaseVoF_damBreak4phaseLaminar` | Validé OF13 — `End=10 s` | `SystemDirectory.import_reference_file`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Quatre phases `water/oil/mercury/air` reproduites avec FoamPilot : blockMesh, setFields et `foamRun -solver compressibleMultiphaseVoF` jusqu’à `End=10 s` sans `FOAM FATAL`. |
| 9 | `compressibleVoF` | `angledDuct` | `/opt/openfoam13/tutorials/compressibleVoF/angledDuct` | `16_compressibleVoF_angledDuct` | Validé OF13 — `End=10 s` | `SystemDirectory.import_reference_file`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Mise en données OF13 reproduite avec la ressource officielle `resources/blockMesh/angledDuct`, propriétés `air/water`, champs `alpha.water`, `T.air/T.water`, `p_rgh`, puis `foamRun -solver compressibleVoF` jusqu’à `End=10 s` sans `FOAM FATAL`.
| 10 | `compressibleVoF` | `ballValve` | `/opt/openfoam13/tutorials/compressibleVoF/ballValve` | `10_compressibleVoF_ballValve` | Validé — `End=0.1 s` | `CaseFieldsManager.set_vof_primary_phase`; `PhasePhysicalPropertiesFile` accepte `thermo_type`/`mixture`; `PhasePropertiesFile` accepte un `sigma` dictionnaire; `ConstantDirectory.configure_vof` conserve `pRef`; `ConstantDirectory.import_reference_file`; `SystemDirectory.import_reference_file`; `BlockMesher.import_reference_dict`; `BlockMesher.import_reference_asset`; `BlockMesher.create_non_conformal_couples` | Mise en données OF13 reproduite : phases `vapour/water`, `physicalProperties` thermodynamiques, RAS `realizableKE`, `pRef`, fraction `alpha.vapour`, asset torique officiel, `potentialFoam`, `compressibleVoF`; validation complète sans `FOAM FATAL` jusqu’à `End=0.1 s`. |
| 11 | `compressibleVoF` | `climbingRod` | `/opt/openfoam13/tutorials/compressibleVoF/climbingRod` | `17_compressibleVoF_climbingRod` | Validé OF13 — `End=25 s` | `SystemDirectory.import_reference_file`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Chaîne OF13 reproduite avec FoamPilot : `blockMesh`, `extrudeMesh`, `setFields`, puis `foamRun -solver compressibleVoF`; champ `alpha.liquid` importé depuis `alpha.liquid.orig`, `sigma.liquid` et transports `air/liquid` conservés. Validation jusqu’à `End=25 s` sans `FOAM FATAL`.
| 12 | `compressibleVoF` | `damBreak` | `/opt/openfoam13/tutorials/compressibleVoF/damBreak` | `18_compressibleVoF_damBreak` | Validé OF13 — `End=1 s` | `ConstantDirectory.remove_files`; `SystemDirectory.import_reference_file`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Mise en données OF13 reproduite avec FoamPilot : `blockMesh`, `setFields`, puis `foamRun -solver compressibleVoF`. Le nettoyage générique des fichiers par défaut (`transportProperties`, `turbulenceProperties`, `physicalProperties`, `pRef`) est nécessaire après import pour éviter la divergence thermique; validation finale jusqu’à `End=1 s` sans `FOAM FATAL`.
| 13 | `compressibleVoF` | `depthCharge2D` | `/opt/openfoam13/tutorials/compressibleVoF/depthCharge2D` | `19_compressibleVoF_depthCharge2D` | Validé OF13 — `End=0.5 s` | `ConstantDirectory.remove_files`; `SystemDirectory.import_reference_file`; `CaseFieldsManager.import_reference_field`; `BlockMesher.import_reference_dict` | Champs `T`, `p`, `p_rgh` et `alpha.water` importés depuis les `.orig`, puis `blockMesh`, `setFields` et `foamRun -solver compressibleVoF`; validation jusqu’à `End=0.5 s` sans `FOAM FATAL`.
| 14 | `compressibleVoF` | `depthCharge3D` | `/opt/openfoam13/tutorials/compressibleVoF/depthCharge3D` | — | À traiter | — | — |
| 15 | `compressibleVoF` | `sloshingTank2D` | `/opt/openfoam13/tutorials/compressibleVoF/sloshingTank2D` | — | À traiter | — | — |
| 16 | `compressibleVoF` | `throttle` | `/opt/openfoam13/tutorials/compressibleVoF/throttle` | — | À traiter | — | — |
| 17 | `fluid` | `BernardCells` | `/opt/openfoam13/tutorials/fluid/BernardCells` | — | À traiter | — | — |
| 18 | `fluid` | `aerofoilNACA0012` | `/opt/openfoam13/tutorials/fluid/aerofoilNACA0012` | — | À traiter | — | — |
| 19 | `fluid` | `aerofoilNACA0012Steady` | `/opt/openfoam13/tutorials/fluid/aerofoilNACA0012Steady` | — | À traiter | — | — |
| 20 | `fluid` | `angledDuct` | `/opt/openfoam13/tutorials/fluid/angledDuct` | — | À traiter | — | — |
| 21 | `fluid` | `angledDuctExplicitFixedCoeff` | `/opt/openfoam13/tutorials/fluid/angledDuctExplicitFixedCoeff` | — | À traiter | — | — |
| 22 | `fluid` | `angledDuctLTS` | `/opt/openfoam13/tutorials/fluid/angledDuctLTS` | — | À traiter | — | — |
| 23 | `fluid` | `annularThermalMixer` | `/opt/openfoam13/tutorials/fluid/annularThermalMixer` | — | À traiter | — | — |
| 24 | `fluid` | `blockedChannel` | `/opt/openfoam13/tutorials/fluid/blockedChannel` | — | À traiter | — | — |
| 25 | `fluid` | `buoyantCavity` | `/opt/openfoam13/tutorials/fluid/buoyantCavity` | `08_thermalBuoyancy` | Validé | `CaseFieldsManager.register_field` pour valeurs initiales de référence | Géométrie source `convertToMeters 0,001`, maillage `35×150×15`, patches `topAndBottom/frontAndBack/hot/cold`, Boussinesq `fluid`, valeurs `k=3,75e-4`, `omega=0,12`, calcul jusqu’à t=1000 sans erreur OpenFOAM. |
| 26 | `fluid` | `decompressionTank` | `/opt/openfoam13/tutorials/fluid/decompressionTank/decompressionTank` | — | À traiter | — | — |
| 27 | `fluid` | `externalCoupledCavity` | `/opt/openfoam13/tutorials/fluid/externalCoupledCavity` | — | À traiter | — | — |
| 28 | `fluid` | `forwardStep` | `/opt/openfoam13/tutorials/fluid/forwardStep` | — | À traiter | — | — |
| 29 | `fluid` | `helmholtzResonance` | `/opt/openfoam13/tutorials/fluid/helmholtzResonance` | — | À traiter | — | — |
| 30 | `fluid` | `hotRadiationRoom` | `/opt/openfoam13/tutorials/fluid/hotRadiationRoom` | — | À traiter | — | — |
| 31 | `fluid` | `hotRadiationRoomFvDOM` | `/opt/openfoam13/tutorials/fluid/hotRadiationRoomFvDOM` | — | À traiter | — | — |
| 32 | `fluid` | `hotRoom` | `/opt/openfoam13/tutorials/fluid/hotRoom` | — | À traiter | — | — |
| 33 | `fluid` | `hotRoomBoussinesq` | `/opt/openfoam13/tutorials/fluid/hotRoomBoussinesq` | — | À traiter | — | — |
| 34 | `fluid` | `hotRoomBoussinesqSteady` | `/opt/openfoam13/tutorials/fluid/hotRoomBoussinesqSteady` | — | À traiter | — | — |
| 35 | `fluid` | `hotRoomComfort` | `/opt/openfoam13/tutorials/fluid/hotRoomComfort` | — | À traiter | — | — |
| 36 | `fluid` | `iglooWithFridges` | `/opt/openfoam13/tutorials/fluid/iglooWithFridges` | — | À traiter | — | — |
| 37 | `fluid` | `mixerVessel2DMRF` | `/opt/openfoam13/tutorials/fluid/mixerVessel2DMRF` | — | À traiter | — | — |
| 38 | `fluid` | `nacaAirfoil` | `/opt/openfoam13/tutorials/fluid/nacaAirfoil` | — | À traiter | — | — |

| 39 | `fluid` | `prism` | `/opt/openfoam13/tutorials/fluid/prism` | — | À traiter | — | — |
| 40 | `fluid` | `roomHeating` | `/opt/openfoam13/tutorials/fluid/roomHeating` | — | À traiter | — | — |
| 41 | `fluid` | `shockTube` | `/opt/openfoam13/tutorials/fluid/shockTube` | — | À traiter | — | — |
| 42 | `fluid` | `squareBend` | `/opt/openfoam13/tutorials/fluid/squareBend` | — | À traiter | — | — |
| 43 | `fluid` | `squareBendLiq` | `/opt/openfoam13/tutorials/fluid/squareBendLiq` | — | À traiter | — | — |
| 44 | `fluid` | `squareBendLiqSteady` | `/opt/openfoam13/tutorials/fluid/squareBendLiqSteady` | — | À traiter | — | — |
| 45 | `fluid` | `stackPlume` | `/opt/openfoam13/tutorials/fluid/stackPlume` | — | À traiter | — | — |
| 46 | `incompressibleDenseParticleFluid` | `Goldschmidt` | `/opt/openfoam13/tutorials/incompressibleDenseParticleFluid/Goldschmidt` | — | À traiter | — | — |
| 47 | `incompressibleDenseParticleFluid` | `GoldschmidtMPPIC` | `/opt/openfoam13/tutorials/incompressibleDenseParticleFluid/GoldschmidtMPPIC` | — | À traiter | — | — |
| 48 | `incompressibleDenseParticleFluid` | `column` | `/opt/openfoam13/tutorials/incompressibleDenseParticleFluid/column` | — | À traiter | — | — |
| 49 | `incompressibleDenseParticleFluid` | `cyclone` | `/opt/openfoam13/tutorials/incompressibleDenseParticleFluid/cyclone` | — | À traiter | — | — |
| 50 | `incompressibleDenseParticleFluid` | `injectionChannel` | `/opt/openfoam13/tutorials/incompressibleDenseParticleFluid/injectionChannel` | — | À traiter | — | — |
| 51 | `incompressibleDriftFlux` | `dahl` | `/opt/openfoam13/tutorials/incompressibleDriftFlux/dahl` | — | À traiter | — | — |
| 52 | `incompressibleDriftFlux` | `mixerVessel2DMRF` | `/opt/openfoam13/tutorials/incompressibleDriftFlux/mixerVessel2DMRF` | — | À traiter | — | — |
| 53 | `incompressibleDriftFlux` | `tank3D` | `/opt/openfoam13/tutorials/incompressibleDriftFlux/tank3D` | — | À traiter | — | — |
| 54 | `incompressibleFluid` | `T3A` | `/opt/openfoam13/tutorials/incompressibleFluid/T3A` | — | À traiter | — | — |
| 55 | `incompressibleFluid` | `TJunction` | `/opt/openfoam13/tutorials/incompressibleFluid/TJunction` | — | À traiter | — | — |
| 56 | `incompressibleFluid` | `TJunctionFan` | `/opt/openfoam13/tutorials/incompressibleFluid/TJunctionFan` | — | À traiter | — | — |
| 57 | `incompressibleFluid` | `airFoil2D` | `/opt/openfoam13/tutorials/incompressibleFluid/airFoil2D` | — | À traiter | — | — |
| 58 | `incompressibleFluid` | `ballValve` | `/opt/openfoam13/tutorials/incompressibleFluid/ballValve` | — | À traiter | — | — |
| 59 | `incompressibleFluid` | `blockedChannel` | `/opt/openfoam13/tutorials/incompressibleFluid/blockedChannel` | — | À traiter | — | — |
| 60 | `incompressibleFluid` | `boxTurb16` | `/opt/openfoam13/tutorials/incompressibleFluid/boxTurb16` | — | À traiter | — | — |
| 61 | `incompressibleFluid` | `cavity` | `/opt/openfoam13/tutorials/incompressibleFluid/cavity` | — | À traiter | — | — |
| 62 | `incompressibleFluid` | `cavityCoupledU` | `/opt/openfoam13/tutorials/incompressibleFluid/cavityCoupledU` | — | À traiter | — | — |
| 63 | `incompressibleFluid` | `channel395` | `/opt/openfoam13/tutorials/incompressibleFluid/channel395` | — | À traiter | — | — |
| 64 | `incompressibleFluid` | `cylinder` | `/opt/openfoam13/tutorials/incompressibleFluid/cylinder` | — | À traiter | — | — |
| 65 | `incompressibleFluid` | `ductSecondaryFlow` | `/opt/openfoam13/tutorials/incompressibleFluid/ductSecondaryFlow` | — | À traiter | — | — |
| 66 | `incompressibleFluid` | `elipsekkLOmega` | `/opt/openfoam13/tutorials/incompressibleFluid/elipsekkLOmega` | — | À traiter | — | — |
| 67 | `incompressibleFluid` | `flowWithOpenBoundary` | `/opt/openfoam13/tutorials/incompressibleFluid/flowWithOpenBoundary` | — | À traiter | — | — |
| 68 | `incompressibleFluid` | `hopperEmptying` | `/opt/openfoam13/tutorials/incompressibleFluid/hopperParticles/hopperEmptying` | — | À traiter | — | — |
| 69 | `incompressibleFluid` | `hopperInitialState` | `/opt/openfoam13/tutorials/incompressibleFluid/hopperParticles/hopperInitialState` | — | À traiter | — | — |
| 70 | `incompressibleFluid` | `impeller` | `/opt/openfoam13/tutorials/incompressibleFluid/impeller` | — | À traiter | — | — |
| 71 | `incompressibleFluid` | `mixerSRF` | `/opt/openfoam13/tutorials/incompressibleFluid/mixerSRF` | — | À traiter | — | — |
| 72 | `incompressibleFluid` | `mixerVessel2D` | `/opt/openfoam13/tutorials/incompressibleFluid/mixerVessel2D` | — | À traiter | — | — |
| 73 | `incompressibleFluid` | `mixerVessel2DMRF` | `/opt/openfoam13/tutorials/incompressibleFluid/mixerVessel2DMRF` | — | À traiter | — | — |
| 74 | `incompressibleFluid` | `mixerVesselHorizontal2DParticles` | `/opt/openfoam13/tutorials/incompressibleFluid/mixerVesselHorizontal2DParticles` | — | À traiter | — | — |
| 75 | `incompressibleFluid` | `moodyChart` | `/opt/openfoam13/tutorials/incompressibleFluid/moodyChart` | — | À traiter | — | — |
| 76 | `incompressibleFluid` | `motorBike` | `/opt/openfoam13/tutorials/incompressibleFluid/motorBike/motorBike` | `07_motorBike` | Validé CFD — rapport PDF à corriger | `SnappyMesher.import_reference_surface`, `SnappyMesher.run_surface_features` | Asset officiel `resources/geometry/motorBike.obj.gz` importé via Foampilot, maillage snappy et solveur Spalart–Allmaras exécutés jusqu’à End sans erreur OpenFOAM; visualisations et statistiques produites. Génération PDF échouée uniquement faute de `pdflatex`. |
| 77 | `incompressibleFluid` | `motorBikeSteady` | `/opt/openfoam13/tutorials/incompressibleFluid/motorBikeSteady` | — | À traiter | — | — |
| 78 | `incompressibleFluid` | `movingCone` | `/opt/openfoam13/tutorials/incompressibleFluid/movingCone` | — | À traiter | — | — |
| 79 | `incompressibleFluid` | `offsetCylinder` | `/opt/openfoam13/tutorials/incompressibleFluid/offsetCylinder` | — | À traiter | — | — |
| 80 | `incompressibleFluid` | `oscillatingInlet` | `/opt/openfoam13/tutorials/incompressibleFluid/oscillatingInlet` | — | À traiter | — | — |
| 81 | `incompressibleFluid` | `pipeCyclic` | `/opt/openfoam13/tutorials/incompressibleFluid/pipeCyclic` | — | À traiter | — | — |
| 82 | `incompressibleFluid` | `pitzDaily` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDaily` | — | À traiter | — | — |
| 83 | `incompressibleFluid` | `pitzDailyLES` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailyLES` | — | À traiter | — | — |
| 84 | `incompressibleFluid` | `pitzDailyLESDevelopedInlet` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailyLESDevelopedInlet` | — | À traiter | — | — |
| 85 | `incompressibleFluid` | `pitzDailyLTS` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailyLTS` | — | À traiter | — | — |
| 86 | `incompressibleFluid` | `pitzDailyPulse` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailyPulse` | — | À traiter | — | — |
| 87 | `incompressibleFluid` | `pitzDailyScalarTransport` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailyScalarTransport` | `05_scalarTransport` | Validé | `Functions.copy_reference_fields`, `Functions.scalar_transport`, `Functions.coded_function_object`, `Functions.write_function_object`, `ControlDictFile.sub_solver`, `CaseFieldsManager.register_field` avec valeurs `uniform`/`nonuniform` | Maillage officiel `resources/blockMesh/pitzDaily`, champs initiaux source `0/U,p,T,k,epsilon,nut,phi`, `solver functions` avec `subSolver incompressibleFluid`, function objects scalarTransport et mixingQualityCheck générés par Foampilot, calcul jusqu’à t=0,2 s sans erreur fatale. |
| 88 | `incompressibleFluid` | `pitzDailySteady` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailySteady` | — | À traiter | — | — |
| 89 | `incompressibleFluid` | `pitzDailySteadyExperimentalInlet` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailySteadyExperimentalInlet` | — | À traiter | — | — |
| 90 | `incompressibleFluid` | `pitzDailySteadyMappedToPart` | `/opt/openfoam13/tutorials/incompressibleFluid/pitzDailySteadyMappedToPart` | — | À traiter | — | — |
| 91 | `incompressibleFluid` | `planarContraction` | `/opt/openfoam13/tutorials/incompressibleFluid/planarContraction` | — | À traiter | — | — |
| 92 | `incompressibleFluid` | `planarCouette` | `/opt/openfoam13/tutorials/incompressibleFluid/planarCouette` | — | À traiter | — | — |
| 93 | `incompressibleFluid` | `planarPoiseuille` | `/opt/openfoam13/tutorials/incompressibleFluid/planarPoiseuille` | — | À traiter | — | — |
| 94 | `incompressibleFluid` | `porousBlockage` | `/opt/openfoam13/tutorials/incompressibleFluid/porousBlockage` | — | À traiter | — | — |
| 95 | `incompressibleFluid` | `propeller` | `/opt/openfoam13/tutorials/incompressibleFluid/propeller` | — | À traiter | — | — |
| 96 | `incompressibleFluid` | `roomResidenceTime` | `/opt/openfoam13/tutorials/incompressibleFluid/roomResidenceTime` | — | À traiter | — | — |
| 97 | `incompressibleFluid` | `rotor2D` | `/opt/openfoam13/tutorials/incompressibleFluid/rotor2D` | — | À traiter | — | — |
| 98 | `incompressibleFluid` | `rotor2DSRF` | `/opt/openfoam13/tutorials/incompressibleFluid/rotor2DSRF` | — | À traiter | — | — |
| 99 | `incompressibleFluid` | `rotorDisk` | `/opt/openfoam13/tutorials/incompressibleFluid/rotorDisk` | — | À traiter | — | — |
| 100 | `incompressibleFluid` | `simpleRushtonMRF` | `/opt/openfoam13/tutorials/incompressibleFluid/simpleRushtonMRF` | — | À traiter | — | — |
| 101 | `incompressibleFluid` | `simpleRushtonNCC` | `/opt/openfoam13/tutorials/incompressibleFluid/simpleRushtonNCC` | — | À traiter | — | — |
| 102 | `incompressibleFluid` | `turbineSiting` | `/opt/openfoam13/tutorials/incompressibleFluid/turbineSiting` | — | À traiter | — | — |
| 103 | `incompressibleFluid` | `venturiTube` | `/opt/openfoam13/tutorials/incompressibleFluid/venturiTube` | — | À traiter | — | — |
| 104 | `incompressibleFluid` | `waveSubSurface` | `/opt/openfoam13/tutorials/incompressibleFluid/waveSubSurface` | — | À traiter | — | — |
| 105 | `incompressibleFluid` | `windAroundBuildings` | `/opt/openfoam13/tutorials/incompressibleFluid/windAroundBuildings` | `06_buildingAero` | Validé CFD — post-traitement à corriger | `SnappyMesher.import_reference_surface`, `SnappyMesher.run_surface_features` | Asset officiel `buildings.obj.gz`, maillage et solveur validés jusqu’à t=400 sans erreur OpenFOAM. Les exports/statistiques Foampilot sont produits, mais VTK signale des Jacobiennes non inversibles sur certaines cellules du maillage snappy; ce point reste documenté pour le post-traitement. |
| 106 | `incompressibleFluid` | `wingMotion2D_steady` | `/opt/openfoam13/tutorials/incompressibleFluid/wingMotion/wingMotion2D_steady` | — | À traiter | — | — |
| 107 | `incompressibleFluid` | `wingMotion2D_transient` | `/opt/openfoam13/tutorials/incompressibleFluid/wingMotion/wingMotion2D_transient` | — | À traiter | — | — |
| 108 | `incompressibleMultiphaseVoF` | `damBreak4phase` | `/opt/openfoam13/tutorials/incompressibleMultiphaseVoF/damBreak4phase` | — | À traiter | — | — |
| 109 | `incompressibleMultiphaseVoF` | `damBreak4phaseFineLaminar` | `/opt/openfoam13/tutorials/incompressibleMultiphaseVoF/damBreak4phaseFineLaminar` | — | À traiter | — | — |
| 110 | `incompressibleMultiphaseVoF` | `damBreak4phaseLaminar` | `/opt/openfoam13/tutorials/incompressibleMultiphaseVoF/damBreak4phaseLaminar` | — | À traiter | — | — |
| 111 | `incompressibleMultiphaseVoF` | `mixerVessel2DMRF` | `/opt/openfoam13/tutorials/incompressibleMultiphaseVoF/mixerVessel2DMRF` | — | À traiter | — | — |
| 112 | `incompressibleVoF` | `DTCHull` | `/opt/openfoam13/tutorials/incompressibleVoF/DTCHull` | — | À traiter | — | — |
| 113 | `incompressibleVoF` | `DTCHullMoving` | `/opt/openfoam13/tutorials/incompressibleVoF/DTCHullMoving` | — | À traiter | — | — |
| 114 | `incompressibleVoF` | `DTCHullWave` | `/opt/openfoam13/tutorials/incompressibleVoF/DTCHullWave` | — | À traiter | — | — |
| 115 | `incompressibleVoF` | `angledDuct` | `/opt/openfoam13/tutorials/incompressibleVoF/angledDuct` | — | À traiter | — | — |
| 116 | `incompressibleVoF` | `capillaryRise` | `/opt/openfoam13/tutorials/incompressibleVoF/capillaryRise` | — | À traiter | — | — |
| 117 | `incompressibleVoF` | `cavitatingBullet` | `/opt/openfoam13/tutorials/incompressibleVoF/cavitatingBullet` | — | À traiter | — | — |
| 118 | `incompressibleVoF` | `climbingRod` | `/opt/openfoam13/tutorials/incompressibleVoF/climbingRod` | — | À traiter | — | — |
| 119 | `incompressibleVoF` | `containerDischarge2D` | `/opt/openfoam13/tutorials/incompressibleVoF/containerDischarge2D` | — | À traiter | — | — |
| 120 | `incompressibleVoF` | `damBreak` | `/opt/openfoam13/tutorials/incompressibleVoF/damBreak` | — | À traiter | — | — |
| 121 | `incompressibleVoF` | `damBreak3D` | `/opt/openfoam13/tutorials/incompressibleVoF/damBreak3D` | — | À traiter | — | — |
| 122 | `incompressibleVoF` | `damBreakLaminar` | `/opt/openfoam13/tutorials/incompressibleVoF/damBreakLaminar` | `04_damBreak_multiphase` | Validé | `Solver.configure_vof`, `Functions.write_set_fields_dict`, `Solver.run_command`, `Boundary.set_raw_condition`, `Boundary.write_boundary_conditions`, `Solver.run_simulation` | Maillage blockMesh source reproduit, initialisation de la colonne d’eau par `setFields`, calcul interFoam jusqu’à t=1 s, alpha.water non uniforme et aucune erreur fatale. |
| 123 | `incompressibleVoF` | `damBreakTracer` | `/opt/openfoam13/tutorials/incompressibleVoF/damBreakTracer` | — | À traiter | — | — |
| 124 | `incompressibleVoF` | `floatingObject` | `/opt/openfoam13/tutorials/incompressibleVoF/floatingObject` | — | À traiter | — | — |
| 125 | `incompressibleVoF` | `floatingObjectWaves` | `/opt/openfoam13/tutorials/incompressibleVoF/floatingObjectWaves` | — | À traiter | — | — |
| 126 | `incompressibleVoF` | `forcedUpstreamWave` | `/opt/openfoam13/tutorials/incompressibleVoF/forcedUpstreamWave` | — | À traiter | — | — |
| 127 | `incompressibleVoF` | `mixerVessel` | `/opt/openfoam13/tutorials/incompressibleVoF/mixerVessel` | — | À traiter | — | — |
| 128 | `incompressibleVoF` | `mixerVessel2DMRF` | `/opt/openfoam13/tutorials/incompressibleVoF/mixerVessel2DMRF` | — | À traiter | — | — |
| 129 | `incompressibleVoF` | `mixerVesselHorizontal2D` | `/opt/openfoam13/tutorials/incompressibleVoF/mixerVesselHorizontal2D` | — | À traiter | — | — |
| 130 | `incompressibleVoF` | `nozzleFlow2D` | `/opt/openfoam13/tutorials/incompressibleVoF/nozzleFlow2D` | — | À traiter | — | — |
| 131 | `incompressibleVoF` | `parshallFlume` | `/opt/openfoam13/tutorials/incompressibleVoF/parshallFlume` | — | À traiter | — | — |
| 132 | `incompressibleVoF` | `planingHullW3` | `/opt/openfoam13/tutorials/incompressibleVoF/planingHullW3` | — | À traiter | — | — |
| 133 | `incompressibleVoF` | `propeller` | `/opt/openfoam13/tutorials/incompressibleVoF/propeller` | — | À traiter | — | — |
| 134 | `incompressibleVoF` | `rotatingCube` | `/opt/openfoam13/tutorials/incompressibleVoF/rotatingCube` | — | À traiter | — | — |
| 135 | `incompressibleVoF` | `sloshingCylinder` | `/opt/openfoam13/tutorials/incompressibleVoF/sloshingCylinder` | — | À traiter | — | — |
| 136 | `incompressibleVoF` | `sloshingTank2D` | `/opt/openfoam13/tutorials/incompressibleVoF/sloshingTank2D` | — | À traiter | — | — |
| 137 | `incompressibleVoF` | `sloshingTank2D3DoF` | `/opt/openfoam13/tutorials/incompressibleVoF/sloshingTank2D3DoF` | — | À traiter | — | — |
| 138 | `incompressibleVoF` | `sloshingTank3D` | `/opt/openfoam13/tutorials/incompressibleVoF/sloshingTank3D` | — | À traiter | — | — |
| 139 | `incompressibleVoF` | `sloshingTank3D3DoF` | `/opt/openfoam13/tutorials/incompressibleVoF/sloshingTank3D3DoF` | — | À traiter | — | — |
| 140 | `incompressibleVoF` | `sloshingTank3D6DoF` | `/opt/openfoam13/tutorials/incompressibleVoF/sloshingTank3D6DoF` | — | À traiter | — | — |
| 141 | `incompressibleVoF` | `testTubeMixer` | `/opt/openfoam13/tutorials/incompressibleVoF/testTubeMixer` | — | À traiter | — | — |
| 142 | `incompressibleVoF` | `trayedPipe` | `/opt/openfoam13/tutorials/incompressibleVoF/trayedPipe` | — | À traiter | — | — |
| 143 | `incompressibleVoF` | `waterChannel` | `/opt/openfoam13/tutorials/incompressibleVoF/waterChannel` | — | À traiter | — | — |
| 144 | `incompressibleVoF` | `wave` | `/opt/openfoam13/tutorials/incompressibleVoF/wave` | — | À traiter | — | — |
| 145 | `incompressibleVoF` | `wave3D` | `/opt/openfoam13/tutorials/incompressibleVoF/wave3D` | — | À traiter | — | — |
| 146 | `incompressibleVoF` | `weirOverflow` | `/opt/openfoam13/tutorials/incompressibleVoF/weirOverflow` | — | À traiter | — | — |
| 147 | `isothermalFilm` | `rivuletPanel` | `/opt/openfoam13/tutorials/isothermalFilm/rivuletPanel` | — | À traiter | — | — |
| 148 | `isothermalFluid` | `potentialFreeSurfaceOscillatingBox` | `/opt/openfoam13/tutorials/isothermalFluid/potentialFreeSurfaceOscillatingBox` | — | À traiter | — | — |
| 149 | `legacy` | `europeanCall` | `/opt/openfoam13/tutorials/legacy/basic/financialFoam/europeanCall` | — | À traiter | — | — |
| 150 | `legacy` | `flange` | `/opt/openfoam13/tutorials/legacy/basic/laplacianFoam/flange` | — | À traiter | — | — |
| 151 | `legacy` | `angledDuctExplicit` | `/opt/openfoam13/tutorials/legacy/compressible/rhoPorousSimpleFoam/angledDuctExplicit` | — | À traiter | — | — |
| 152 | `legacy` | `angledDuctImplicit` | `/opt/openfoam13/tutorials/legacy/compressible/rhoPorousSimpleFoam/angledDuctImplicit` | — | À traiter | — | — |
| 153 | `legacy` | `chargedWire` | `/opt/openfoam13/tutorials/legacy/electromagnetics/electrostaticFoam/chargedWire` | — | À traiter | — | — |
| 154 | `legacy` | `hartmann` | `/opt/openfoam13/tutorials/legacy/electromagnetics/mhdFoam/hartmann` | — | À traiter | — | — |
| 155 | `legacy` | `pitzDaily` | `/opt/openfoam13/tutorials/legacy/incompressible/adjointShapeOptimisationFoam/pitzDaily` | — | À traiter | — | — |
| 156 | `legacy` | `elbow` | `/opt/openfoam13/tutorials/legacy/incompressible/icoFoam/elbow` | — | À traiter | — | — |
| 157 | `legacy` | `angledDuctExplicit` | `/opt/openfoam13/tutorials/legacy/incompressible/porousSimpleFoam/angledDuctExplicit` | — | À traiter | — | — |
| 158 | `legacy` | `angledDuctImplicit` | `/opt/openfoam13/tutorials/legacy/incompressible/porousSimpleFoam/angledDuctImplicit` | — | À traiter | — | — |
| 159 | `legacy` | `squareBump` | `/opt/openfoam13/tutorials/legacy/incompressible/shallowWaterFoam/squareBump` | — | À traiter | — | — |
| 160 | `legacy` | `freeSpacePeriodic` | `/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/freeSpacePeriodic` | — | À traiter | — | — |
| 161 | `legacy` | `freeSpaceStream` | `/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/freeSpaceStream` | — | À traiter | — | — |
| 162 | `legacy` | `supersonicCorner` | `/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/supersonicCorner` | — | À traiter | — | — |
| 163 | `legacy` | `wedge15Ma5` | `/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/wedge15Ma5` | — | À traiter | — | — |
| 164 | `legacy` | `periodicCubeArgon` | `/opt/openfoam13/tutorials/legacy/lagrangian/mdEquilibrationFoam/periodicCubeArgon` | — | À traiter | — | — |
| 165 | `legacy` | `periodicCubeWater` | `/opt/openfoam13/tutorials/legacy/lagrangian/mdEquilibrationFoam/periodicCubeWater` | — | À traiter | — | — |
| 166 | `legacy` | `nanoNozzle` | `/opt/openfoam13/tutorials/legacy/lagrangian/mdFoam/nanoNozzle` | — | À traiter | — | — |
| 167 | `movingMesh` | `SnakeRiverCanyon` | `/opt/openfoam13/tutorials/movingMesh/SnakeRiverCanyon` | — | À traiter | — | — |
| 168 | `multiRegion` | `VoFcoolingCylinder2D` | `/opt/openfoam13/tutorials/multiRegion/CHT/VoFcoolingCylinder2D` | — | À traiter | — | — |
| 169 | `multiRegion` | `circuitBoardCooling` | `/opt/openfoam13/tutorials/multiRegion/CHT/circuitBoardCooling` | — | À traiter | — | — |
| 170 | `multiRegion` | `coolingCylinder2D` | `/opt/openfoam13/tutorials/multiRegion/CHT/coolingCylinder2D` | — | À traiter | — | — |
| 171 | `multiRegion` | `templates` | `/opt/openfoam13/tutorials/multiRegion/CHT/coolingSphere/templates` | — | À traiter | — | — |
| 172 | `multiRegion` | `engine2Valve2D` | `/opt/openfoam13/tutorials/multiRegion/CHT/engine2Valve2D` | — | À traiter | — | — |
| 173 | `multiRegion` | `heatExchanger` | `/opt/openfoam13/tutorials/multiRegion/CHT/heatExchanger` | — | À traiter | — | — |
| 174 | `multiRegion` | `heatedDuct` | `/opt/openfoam13/tutorials/multiRegion/CHT/heatedDuct` | `09_CHT_heatedDuct` | Validé — `foamMultiRun` jusqu’à `End=20 s`, VTK et post-traitement sans erreur | `ChtSolver.set_region_boundary_conditions`, `set_region_internal_field`, `set_region_gravity`, `set_region_momentum_transport`, `write_region_system_files`; support OF13 multi-région dans `SolidRegion` | Référence OF13 alignée sur trois régions `fluid/heater/metal`; génération régionale de `g`, `momentumTransport`, `physicalProperties`, `fvSchemes`, `fvSolution`, conditions limites et solveurs compressibles/solides. Le post-traitement charge désormais les VTK multi-régions. |
| 175 | `multiRegion` | `misalignedDuct` | `/opt/openfoam13/tutorials/multiRegion/CHT/misalignedDuct` | — | À traiter | — | — |
| 176 | `multiRegion` | `multiphaseCoolingCylinder2D` | `/opt/openfoam13/tutorials/multiRegion/CHT/multiphaseCoolingCylinder2D` | — | À traiter | — | — |
| 177 | `multiRegion` | `notchedRoller` | `/opt/openfoam13/tutorials/multiRegion/CHT/notchedRoller` | — | À traiter | — | — |
| 178 | `multiRegion` | `reverseBurner` | `/opt/openfoam13/tutorials/multiRegion/CHT/reverseBurner` | — | À traiter | — | — |
| 179 | `multiRegion` | `shellAndTubeHeatExchanger` | `/opt/openfoam13/tutorials/multiRegion/CHT/shellAndTubeHeatExchanger` | — | À traiter | — | — |
| 180 | `multiRegion` | `wallBoiling` | `/opt/openfoam13/tutorials/multiRegion/CHT/wallBoiling` | — | À traiter | — | — |
| 181 | `multiRegion` | `VoFToFilm` | `/opt/openfoam13/tutorials/multiRegion/film/VoFToFilm` | — | À traiter | — | — |
| 182 | `multiRegion` | `cylinder` | `/opt/openfoam13/tutorials/multiRegion/film/cylinder` | — | À traiter | — | — |
| 183 | `multiRegion` | `cylinderDripping` | `/opt/openfoam13/tutorials/multiRegion/film/cylinderDripping` | — | À traiter | — | — |
| 184 | `multiRegion` | `cylinderVoF` | `/opt/openfoam13/tutorials/multiRegion/film/cylinderVoF` | — | À traiter | — | — |
| 185 | `multiRegion` | `hotBoxes` | `/opt/openfoam13/tutorials/multiRegion/film/hotBoxes` | — | À traiter | — | — |
| 186 | `multiRegion` | `rivuletBox` | `/opt/openfoam13/tutorials/multiRegion/film/rivuletBox` | — | À traiter | — | — |
| 187 | `multiRegion` | `rivuletPanel` | `/opt/openfoam13/tutorials/multiRegion/film/rivuletPanel` | — | À traiter | — | — |
| 188 | `multiRegion` | `splashPanel` | `/opt/openfoam13/tutorials/multiRegion/film/splashPanel` | — | À traiter | — | — |
| 189 | `multicomponentFluid` | `DLR_A_LTS` | `/opt/openfoam13/tutorials/multicomponentFluid/DLR_A_LTS` | — | À traiter | — | — |
| 190 | `multicomponentFluid` | `SandiaD_LTS` | `/opt/openfoam13/tutorials/multicomponentFluid/SandiaD_LTS` | — | À traiter | — | — |
| 191 | `multicomponentFluid` | `aachenBomb` | `/opt/openfoam13/tutorials/multicomponentFluid/aachenBomb` | — | À traiter | — | — |
| 192 | `multicomponentFluid` | `counterFlowFlame2D` | `/opt/openfoam13/tutorials/multicomponentFluid/counterFlowFlame2D` | — | À traiter | — | — |
| 193 | `multicomponentFluid` | `counterFlowFlame2DLTS` | `/opt/openfoam13/tutorials/multicomponentFluid/counterFlowFlame2DLTS` | — | À traiter | — | — |
| 194 | `multicomponentFluid` | `counterFlowFlame2DLTS_GRI_TDAC` | `/opt/openfoam13/tutorials/multicomponentFluid/counterFlowFlame2DLTS_GRI_TDAC` | — | À traiter | — | — |
| 195 | `multicomponentFluid` | `counterFlowFlame2D_GRI` | `/opt/openfoam13/tutorials/multicomponentFluid/counterFlowFlame2D_GRI` | — | À traiter | — | — |
| 196 | `multicomponentFluid` | `counterFlowFlame2D_GRI_TDAC` | `/opt/openfoam13/tutorials/multicomponentFluid/counterFlowFlame2D_GRI_TDAC` | — | À traiter | — | — |
| 197 | `multicomponentFluid` | `filter` | `/opt/openfoam13/tutorials/multicomponentFluid/filter` | — | À traiter | — | — |
| 198 | `multicomponentFluid` | `lockExchange` | `/opt/openfoam13/tutorials/multicomponentFluid/lockExchange` | — | À traiter | — | — |
| 199 | `multicomponentFluid` | `membrane` | `/opt/openfoam13/tutorials/multicomponentFluid/membrane` | — | À traiter | — | — |
| 200 | `multicomponentFluid` | `nc7h16` | `/opt/openfoam13/tutorials/multicomponentFluid/nc7h16` | — | À traiter | — | — |
| 201 | `multicomponentFluid` | `parcelInBox` | `/opt/openfoam13/tutorials/multicomponentFluid/parcelInBox` | — | À traiter | — | — |
| 202 | `multicomponentFluid` | `simplifiedSiwek` | `/opt/openfoam13/tutorials/multicomponentFluid/simplifiedSiwek` | — | À traiter | — | — |
| 203 | `multicomponentFluid` | `smallPoolFire2D` | `/opt/openfoam13/tutorials/multicomponentFluid/smallPoolFire2D` | — | À traiter | — | — |
| 204 | `multicomponentFluid` | `smallPoolFire3D` | `/opt/openfoam13/tutorials/multicomponentFluid/smallPoolFire3D` | — | À traiter | — | — |
| 205 | `multicomponentFluid` | `verticalChannel` | `/opt/openfoam13/tutorials/multicomponentFluid/verticalChannel` | — | À traiter | — | — |
| 206 | `multicomponentFluid` | `verticalChannelLTS` | `/opt/openfoam13/tutorials/multicomponentFluid/verticalChannelLTS` | — | À traiter | — | — |
| 207 | `multicomponentFluid` | `verticalChannelSteady` | `/opt/openfoam13/tutorials/multicomponentFluid/verticalChannelSteady` | — | À traiter | — | — |
| 208 | `multiphaseEuler` | `Grossetete` | `/opt/openfoam13/tutorials/multiphaseEuler/Grossetete` | — | À traiter | — | — |
| 209 | `multiphaseEuler` | `LBend` | `/opt/openfoam13/tutorials/multiphaseEuler/LBend` | — | À traiter | — | — |
| 210 | `multiphaseEuler` | `aeratedStirredTankMRF` | `/opt/openfoam13/tutorials/multiphaseEuler/aeratedStirredTankMRF` | — | À traiter | — | — |
| 211 | `multiphaseEuler` | `bed` | `/opt/openfoam13/tutorials/multiphaseEuler/bed` | — | À traiter | — | — |
| 212 | `multiphaseEuler` | `boilingBed` | `/opt/openfoam13/tutorials/multiphaseEuler/boilingBed` | — | À traiter | — | — |
| 213 | `multiphaseEuler` | `bubbleColumn` | `/opt/openfoam13/tutorials/multiphaseEuler/bubbleColumn` | — | À traiter | — | — |
| 214 | `multiphaseEuler` | `bubbleColumnEvaporating` | `/opt/openfoam13/tutorials/multiphaseEuler/bubbleColumnEvaporating` | — | À traiter | — | — |
| 215 | `multiphaseEuler` | `bubbleColumnEvaporatingDissolving` | `/opt/openfoam13/tutorials/multiphaseEuler/bubbleColumnEvaporatingDissolving` | — | À traiter | — | — |
| 216 | `multiphaseEuler` | `bubbleColumnEvaporatingReacting` | `/opt/openfoam13/tutorials/multiphaseEuler/bubbleColumnEvaporatingReacting` | — | À traiter | — | — |
| 217 | `multiphaseEuler` | `bubbleColumnIATE` | `/opt/openfoam13/tutorials/multiphaseEuler/bubbleColumnIATE` | — | À traiter | — | — |
| 218 | `multiphaseEuler` | `bubbleColumnLES` | `/opt/openfoam13/tutorials/multiphaseEuler/bubbleColumnLES` | — | À traiter | — | — |
| 219 | `multiphaseEuler` | `bubbleColumnLaminar` | `/opt/openfoam13/tutorials/multiphaseEuler/bubbleColumnLaminar` | — | À traiter | — | — |
| 220 | `multiphaseEuler` | `bubblePipe` | `/opt/openfoam13/tutorials/multiphaseEuler/bubblePipe` | — | À traiter | — | — |
| 221 | `multiphaseEuler` | `damBreak4phase` | `/opt/openfoam13/tutorials/multiphaseEuler/damBreak4phase` | — | À traiter | — | — |
| 222 | `multiphaseEuler` | `fluidisedBed` | `/opt/openfoam13/tutorials/multiphaseEuler/fluidisedBed` | — | À traiter | — | — |
| 223 | `multiphaseEuler` | `fluidisedBedLaminar` | `/opt/openfoam13/tutorials/multiphaseEuler/fluidisedBedLaminar` | — | À traiter | — | — |
| 224 | `multiphaseEuler` | `hydrofoil` | `/opt/openfoam13/tutorials/multiphaseEuler/hydrofoil` | — | À traiter | — | — |
| 225 | `multiphaseEuler` | `injection` | `/opt/openfoam13/tutorials/multiphaseEuler/injection` | — | À traiter | — | — |
| 226 | `multiphaseEuler` | `mixerVessel2D` | `/opt/openfoam13/tutorials/multiphaseEuler/mixerVessel2D` | — | À traiter | — | — |
| 227 | `multiphaseEuler` | `mixerVessel2DMRF` | `/opt/openfoam13/tutorials/multiphaseEuler/mixerVessel2DMRF` | — | À traiter | — | — |
| 228 | `multiphaseEuler` | `pipeBend` | `/opt/openfoam13/tutorials/multiphaseEuler/pipeBend` | — | À traiter | — | — |
| 229 | `multiphaseEuler` | `steamInjection` | `/opt/openfoam13/tutorials/multiphaseEuler/steamInjection` | — | À traiter | — | — |
| 230 | `multiphaseEuler` | `titaniaSynthesis` | `/opt/openfoam13/tutorials/multiphaseEuler/titaniaSynthesis` | — | À traiter | — | — |
| 231 | `multiphaseEuler` | `titaniaSynthesisSurface` | `/opt/openfoam13/tutorials/multiphaseEuler/titaniaSynthesisSurface` | — | À traiter | — | — |
| 232 | `multiphaseEuler` | `wallBoilingIATE` | `/opt/openfoam13/tutorials/multiphaseEuler/wallBoilingIATE` | — | À traiter | — | — |
| 233 | `multiphaseEuler` | `wallBoilingPolydisperse` | `/opt/openfoam13/tutorials/multiphaseEuler/wallBoilingPolydisperse` | — | À traiter | — | — |
| 234 | `multiphaseEuler` | `wallBoilingPolydisperseTwoGroups` | `/opt/openfoam13/tutorials/multiphaseEuler/wallBoilingPolydisperseTwoGroups` | — | À traiter | — | — |
| 235 | `potentialFoam` | `cylinder` | `/opt/openfoam13/tutorials/potentialFoam/cylinder` | — | À traiter | — | — |
| 236 | `potentialFoam` | `pitzDaily` | `/opt/openfoam13/tutorials/potentialFoam/pitzDaily` | — | À traiter | — | — |
| 237 | `shockFluid` | `LadenburgJet60psi` | `/opt/openfoam13/tutorials/shockFluid/LadenburgJet60psi` | — | À traiter | — | — |
| 238 | `shockFluid` | `biconic25-55Run35` | `/opt/openfoam13/tutorials/shockFluid/biconic25-55Run35` | — | À traiter | — | — |
| 239 | `shockFluid` | `diffuserIntake` | `/opt/openfoam13/tutorials/shockFluid/diffuserIntake` | — | À traiter | — | — |
| 240 | `shockFluid` | `forwardStep` | `/opt/openfoam13/tutorials/shockFluid/forwardStep` | — | À traiter | — | — |
| 241 | `shockFluid` | `movingCone` | `/opt/openfoam13/tutorials/shockFluid/movingCone` | — | À traiter | — | — |
| 242 | `shockFluid` | `obliqueShock` | `/opt/openfoam13/tutorials/shockFluid/obliqueShock` | — | À traiter | — | — |
| 243 | `shockFluid` | `shockTube` | `/opt/openfoam13/tutorials/shockFluid/shockTube` | — | À traiter | — | — |
| 244 | `shockFluid` | `wedge15Ma5` | `/opt/openfoam13/tutorials/shockFluid/wedge15Ma5` | — | À traiter | — | — |
| 245 | `solidDisplacement` | `beamEndLoad` | `/opt/openfoam13/tutorials/solidDisplacement/beamEndLoad` | — | À traiter | — | — |
| 246 | `solidDisplacement` | `plateHole` | `/opt/openfoam13/tutorials/solidDisplacement/plateHole` | — | À traiter | — | — |
