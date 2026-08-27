# Contrôle de présence des runners FoamPilot — OpenFOAM 13

## Méthode

Le contrôle compare les équivalents FoamPilot déclarés dans [`openfoam13_foampilot_integration.md`](openfoam13_foampilot_integration.md) avec le chemin attendu `foampilot/tutorials/<équivalent>/run.py`. Un tutoriel est considéré comme présent uniquement lorsque son dossier et son fichier `run.py` existent effectivement dans le dépôt.

La vérification couvre désormais les équivalents déclarés dans la matrice ainsi que les runners récemment ajoutés jusqu’à `incompressibleVoF/propeller`.

## Résultats

| Équivalent FoamPilot | Dossier | `run.py` | Présence | Statut matrice |
|---|---|---|---|---|
| `01_cavity_laminar` | Présent | Présent | Conforme | Validé |
| `02_simpleCar_turbulent` | Présent | Présent | Conforme | Validé |
| `03_pitzDaily_step` | Présent | Présent | Conforme | Validation longue partielle |
| `04_damBreak_multiphase` | Présent | Présent | Conforme | Validé |
| `05_scalarTransport` | Présent | Présent | Conforme | Validé OF13 — `pitzDailyScalarTransport`, `End=0.2 s`, `subSolver`, `scalarTransport` et `mixingQualityCheck` reproduits |
| `06_buildingAero` | Présent | Présent | Conforme OF13 (`windAroundBuildings`) | Validé — `End=400 s`, 185 237 cellules, chaîne officielle reproduite |
| `07_motorBike` | Présent | Présent | Conforme | Validé |
| `08_thermalBuoyancy` | Présent | Présent | Conforme | Validé |
| `09_CHT_heatedDuct` | Présent | Présent | Conforme | Validé OF13 |
| `10_compressibleVoF_ballValve` | Présent | Présent | Conforme | Validé OF13 |
| `11_XiFluid_engine2Valve2D` | Présent | Présent | Conforme | Validé OF13 |
| `12_XiFluid_moriyoshiHomogeneous` | Présent | Présent | Conforme | Validé OF13 |
| `13_XiFluid_stratified` | Présent | Présent | Conforme | Validé OF13 |
| `14_fluid_cavity` | Présent | Présent | Conforme | Validé OF13 |
| `15_compressibleMultiphaseVoF_damBreak4phaseLaminar` | Présent | Présent | Conforme | Validé OF13 |
| `16_compressibleVoF_angledDuct` | Présent | Présent | Conforme | Validé OF13 |
| `17_compressibleVoF_climbingRod` | Présent | Présent | Conforme | Validé OF13 |
| `18_compressibleVoF_damBreak` | Présent | Présent | Conforme | Validé OF13 |
| `19_compressibleVoF_depthCharge2D` | Présent | Présent | Conforme | Validé OF13 |
| `20_compressibleVoF_depthCharge3D` | Présent | Présent | Conforme | Validé OF13 |
| `21_compressibleVoF_sloshingTank2D` | Présent | Présent | Conforme | Validé OF13 |
| `22_compressibleVoF_throttle` | Présent | Présent | Conforme | Validé OF13 après calcul parallèle à 4 processus |
| `23_fluid_BernardCells` | Présent | Présent | Conforme | Validé OF13 — `End=1000` |
| `24_fluid_aerofoilNACA0012` | Présent | Présent | Conforme | Validé OF13 — `End=0.15 s` |
| `25_fluid_aerofoilNACA0012Steady` | Présent | Présent | Conforme | Validé OF13 — convergence à `1575` itérations |
| `26_fluid_angledDuct` | Présent | Présent | Conforme | Validé OF13 — `End=10 s` |
| `27_fluid_angledDuctExplicitFixedCoeff` | Présent | Présent | Conforme | Validé OF13 — convergence à `516` itérations |
| `28_fluid_angledDuctLTS` | Présent | Présent | Conforme | Validé OF13 — `End=500` |
| `29_fluid_annularThermalMixer` | Présent | Présent | Conforme | Validé OF13 — `End=2 s` |
| `30_fluid_blockedChannel` | Présent | Présent | Conforme | Validé OF13 — `End=0.03 s` |
| `31_fluid_decompressionTank` | Présent | Présent | Conforme | Validé OF13 — `End=0.0001 s` |
| `32_fluid_externalCoupledCavity` | Présent | Présent | Conforme | Validé OF13 — `End=100 s` |
| `33_fluid_forwardStep` | Présent | Présent | Conforme | Validé OF13 — `End=10 s` |
| `34_fluid_helmholtzResonance` | Présent | Présent | Conforme | Validé OF13 — variantes `resolved/modelled`, `End=0.05 s` |
| `35_fluid_hotRadiationRoom` | Présent | Présent | Conforme | Validé OF13 — convergence à `900` itérations |
| `36_fluid_hotRadiationRoomFvDOM` | Présent | Présent | Conforme | Validé OF13 — convergence à `914` itérations |
| `37_fluid_hotRoom` | Présent | Présent | Conforme | Validé OF13 — `End=2000` |
| `38_fluid_hotRoomBoussinesq` | Présent | Présent | Conforme | Validé OF13 — `End=2000` |
| `39_fluid_hotRoomBoussinesqSteady` | Présent | Présent | Conforme | Validé OF13 — `End=2000` |
| `40_fluid_hotRoomComfort` | Présent | Présent | Conforme | Validé OF13 — convergence à `2356` itérations |
| `41_fluid_iglooWithFridges` | Présent | Présent | Conforme | Validé OF13 — `End=4000` |
| `42_fluid_mixerVessel2DMRF` | Présent | Présent | Conforme | Validé OF13 — `End=0.1 s` |
| `43_fluid_nacaAirfoil` | Présent | Présent | Conforme | Non validé — délai atteint à `t≈0.000556 s`, reprise nécessaire |
| `44_fluid_prism` | Présent | Présent | Conforme | Validé OF13 — `End=0.0004 s` |
| `45_fluid_roomHeating` | Présent | Présent | Conforme | Partiel — steady `End=2000 s`; transitoire interrompu vers `t≈3476/6000 s` |
| `46_fluid_shockTube` | Présent | Présent | Conforme | Validé OF13 — `End=0.007 s` |
| `47_fluid_squareBend` | Présent | Présent | Conforme | Validé OF13 — `End=500 s` |
| `48_fluid_squareBendLiq` | Présent | Présent | Conforme | Validé OF13 — `End=0.5 s` |
| `49_fluid_squareBendLiqSteady` | Présent | Présent | Conforme | Validé OF13 — `End=500 s` |
| `50_fluid_stackPlume` | Présent | Présent | Conforme | Validé OF13 — `End=250 s` |
| `51_incompressibleDenseParticleFluid_Goldschmidt` | Présent | Présent | Conforme | Accepté avec réserve — `t≈0.04004/5 s`, sans erreur fatale visible |
| `52_incompressibleDenseParticleFluid_GoldschmidtMPPIC` | Présent | Présent | Conforme | Accepté avec réserve — `t≈0.5744/5 s`, sans erreur fatale visible |
| `53_incompressibleDenseParticleFluid_column` | Présent | Présent | Conforme | Validé OF13 — `End=1 s` |
| `54_incompressibleDenseParticleFluid_cyclone` | Présent | Présent | Conforme | Accepté avec réserve — `t≈0.9694/7 s`, sans erreur fatale visible |
| `55_incompressibleDenseParticleFluid_injectionChannel` | Présent | Présent | Conforme | Accepté avec réserve — `t≈0.0528/0.1 s`, `createZones` validé |
| `56_incompressibleDriftFlux_dahl` | Présent | Présent | Conforme | Accepté avec réserve — `t≈445.44/6400 s`, sans erreur fatale visible |
| `57_incompressibleDriftFlux_mixerVessel2DMRF` | Présent | Présent | Conforme | Accepté avec réserve — `t≈9.67/10 s`, sans erreur fatale visible |
| `58_incompressibleDriftFlux_tank3D` | Présent | Présent | Conforme | En cours — runner créé et calcul lancé |
| `59_incompressibleFluid_T3A` | Présent | Présent | Conforme | Accepté avec réserve — convergence à `Time=268` |
| `60_incompressibleFluid_TJunction` | Présent | Présent | Conforme | Validé OF13 — `End=1.5 s` |
| `61_incompressibleFluid_TJunctionFan` | Présent | Présent | Conforme | Validé OF13 — `End=1.5 s` |
| `62_incompressibleFluid_airFoil2D` | Présent | Présent | Conforme | Accepté avec réserve — convergence SIMPLE à `Time=313` avant `End=500` |
| `63_incompressibleFluid_ballValve` | Présent | Présent | Conforme | Validé OF13 — `End=1 s`, 8 processus et reconstruction réussis |
| `64_incompressibleFluid_blockedChannel` | Présent | Présent | Conforme | Validé OF13 — `End=0.03 s` |
| `65_incompressibleFluid_boxTurb16` | Présent | Présent | Conforme | Validé OF13 — `End=10 s` |
| `66_incompressibleFluid_cavity` | Présent | Présent | Conforme | Validé OF13 — `End=10 s` |
| `67_incompressibleFluid_cavityCoupledU` | Présent | Présent | Conforme | Validé OF13 — `End=10 s` |
| `68_incompressibleFluid_channel395` | Présent | Présent | Conforme | En cours — calcul parallèle à 4 processus jusqu’à `End=1000` |
| `69_incompressibleFluid_cylinder` | Présent | Présent | Conforme | Validé OF13 — `End=5000 s` |
| `70_incompressibleFluid_ductSecondaryFlow` | Présent | Présent | Conforme | Accepté avec réserve — convergence à `Time=5207` avant `End=20000` |
| `71_incompressibleFluid_elipsekkLOmega` | Présent | Présent | Conforme | En cours — calcul long jusqu’à `End=1 s` |
| `72_incompressibleFluid_flowWithOpenBoundary` | Présent | Présent | Conforme | Validé OF13 — `End=100 s` |
| `73_incompressibleFluid_hopperParticles_hopperEmptying` | Présent | Présent | Conforme | Validé OF13 — `End=5 s` |
| `74_incompressibleFluid_hopperParticles_hopperInitialState` | Présent | Présent | Conforme | Accepté avec réserve — interrompu avant `End=0.25 s` |
| `75_incompressibleFluid_impeller` | Présent | Présent | Conforme | Accepté avec réserve — progression sans erreur visible vers `End=5 s` |
| `76_incompressibleFluid_mixerSRF` | Présent | Présent | Conforme | Préparé; validation à reprendre |
| `77_incompressibleFluid_mixerVessel2D` | Présent | Présent | Conforme | En cours — calcul jusqu’à `End=5 s` |
| `78_incompressibleFluid_mixerVessel2DMRF` | Présent | Présent | Conforme | En cours — calcul jusqu’à `End=500 s` |
| `79_incompressibleFluid_mixerVesselHorizontal2DParticles` | Présent | Présent | Conforme | En cours — calcul particulaire OF13 jusqu’à `End=0.25 s` |
| `80_incompressibleFluid_moodyChart` | Présent | Présent | Conforme | Validé OF13 — `End=2 s`, frictionFactor et Uprofile générés |
| `81_incompressibleFluid_motorBikeSteady` | Présent | Présent | Conforme | Validé OF13 — `End=500 s`, snappyHexMesh MPI et reconstruction réussis |
| `82_incompressibleFluid_movingCone` | Présent | Présent | Conforme | Validé OF13 — `End=0.0099 s`, maillages mobiles et cutPlane générés |
| `83_incompressibleFluid_offsetCylinder` | Présent | Présent | Conforme | Validé OF13 — `End=2 s`, CrossPowerLaw reproduit |
| `84_incompressibleFluid_oscillatingInlet` | Présent | Présent | Conforme | Validé OF13 — `End=5 s`, mouvement oscillant et patchFlowRate reproduits |
| `85_incompressibleFluid_pipeCyclic` | Présent | Présent | Conforme | Accepté avec réserve — convergence à `Time=251 s` avant `End=1000 s` |
| `86_incompressibleFluid_pitzDaily` | Présent | Présent | Conforme | Validé OF13 — `End=0.3 s`, maillage partagé, RAS `kEpsilon` et `patchAverage` reproduits |
| `87_incompressibleFluid_pitzDailyLES` | Présent | Présent | Conforme | Validé OF13 — `End=0.1 s`, LES `dynamicKEqn`, scalaire `s` et fonctions de post-traitement reproduits |
| `88_incompressibleFluid_pitzDailyLESDevelopedInlet` | Présent | Présent | Conforme | Validé OF13 — `End=0.1 s`, maillage `mappedInternal` et champs `mappedInternalValue` reproduits |
| `89_incompressibleFluid_pitzDailyLTS` | Présent | Présent | Conforme | Validé OF13 — `End=1000 s`, schéma `localEuler` et paramètres LTS reproduits |
| `90_incompressibleFluid_pitzDailyPulse` | Présent | Présent | Conforme | Validé OF13 — `End=1 s`, inlet `uniformFixedValue coded` pulsé et `patchAverage` reproduits |
| `91_incompressibleFluid_pitzDailySteady` | Présent | Présent | Conforme | Accepté avec réserve — convergence SIMPLE à 285 itérations avant `End=2000`, streamlines et `kEpsilon:G` générés |
| `92_incompressibleFluid_pitzDailySteadyExperimentalInlet` | Présent | Présent | Conforme | Accepté avec réserve — convergence SIMPLE à 786 itérations avant `End=1000`, données `boundaryData` et streamlines reproduites |
| `93_incompressibleFluid_pitzDailySteadyMappedToPart` | Présent | Présent | Conforme | Validé OF13 — source `Time=292`, cible `Time=18`, `mapFieldsPar` et reconstruction `-withZero` réussis |
| `94_incompressibleFluid_planarContraction` | Présent | Présent | Conforme | Validé OF13 — `End=0.25 s`, champ `sigma` et sorties `graphCell` `lineA`, `lineB`, `lineC` générés |
| `95_incompressibleFluid_planarCouette` | Présent | Présent | Conforme | Validé OF13 — `End=25 s`, modèle Maxwell, champ `sigma` et patches cycliques reproduits |
| `96_incompressibleFluid_planarPoiseuille` | Présent | Présent | Conforme | Validé OF13 — `End=25 s`, modèle Maxwell, champ `sigma`, `residuals`, `graphCell` et `probes` générés |
| `97_incompressibleFluid_porousBlockage` | Présent | Présent | Conforme | Validé OF13 — `End=100 s`, zone `porousBlockage`, force `DarcyForchheimer` et `createZones` reproduits |
| `98_incompressibleFluid_propeller` | Présent | Présent | Conforme | Accepté avec réserve — maillage OF13 complet, calcul parallèle stable à `Time=0.01294 s` avant limite de 900 s, forces générées |
| `99_incompressibleFluid_roomResidenceTime` | Présent | Présent | Conforme | Validé OF13 — convergence SIMPLE à 774 itérations, champ `age`, sondes `probes1`/`probes2` et `inletFlowRate=-0.1008` générés |
| `100_incompressibleFluid_rotor2D` | Présent | Présent | Conforme | Validé OF13 — maillage `#codeStream` de 2 880 cellules, `solidBody`/`rotatingMotion`, `End=2 s` atteint |
| `101_incompressibleFluid_rotor2DSRF` | Présent | Présent | Conforme | Validé OF13 — maillage partagé, SRF `omega=60 [rpm]`, conditions MRF et `End=2 s` atteints |
| `102_incompressibleFluid_rotorDisk` | Présent | Présent | Conforme | Accepté avec réserve — zone `rotatingZone`, modèle rotorDisk à `1000 rpm`, convergence SIMPLE à `Time=103 s` avant `End=1000` |
| `103_incompressibleFluid_simpleRushtonMRF` | Présent | Présent | Conforme | Validé OF13 — double `mirrorMesh`, zone MRF, baffles `stirrer`/`baffles`, `End=4000 s`, sorties de puissance générées |
| `104_incompressibleFluid_simpleRushtonNCC` | Présent | Présent | Conforme | Validé OF13 — couples non conformes, `splitBaffles`, rotation `5 rpm`, `End=100 s`, erreurs de flux nulles et sorties de puissance générées |
| `105_incompressibleFluid_turbineSiting` | Présent | Présent | Conforme | Validé OF13 — terrain STL, chaîne parallèle à 4 processus, maillage de 120 246 cellules, convergence SIMPLE à 164 itérations et reconstruction réussie |
| `106_incompressibleFluid_venturiTube` | Présent | Présent | Conforme | Accepté avec réserve — convergence PIMPLE à 380 itérations, sondes et profils `graphA`–`graphF` générés avant `End=1000` |
| `107_incompressibleFluid_waveSubSurface` | Présent | Présent | Conforme | Accepté avec réserve — `setWaves` Stokes5, calcul MPI stable à `Time=96.89 s` avant `End=100 s`, reconstruction non atteinte sous la limite de temps |
| `108_incompressibleFluid_wingMotion2D_steady` | Présent | Présent | Conforme OF13 | Validé — `End=3000 s`, maillage extrudé et patch `wing` créés, aucun `FOAM FATAL` |
| `109_incompressibleFluid_wingMotion2D_transient` | Présent | Présent | Conforme OF13 | Accepté avec réserve — sixDoF MPI stable jusqu’à `t≈0,722 s`, limite d’exécution atteinte avant reconstruction |
| `110_incompressibleMultiphaseVoF_damBreak4phase` | Présent | Présent | Conforme OF13 | Validé — quatre phases initialisées par `setFields`, `End=6 s`, aucun `FOAM FATAL` |
| `111_incompressibleMultiphaseVoF_damBreak4phaseFineLaminar` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage fin et quatre phases stables jusqu’à `t≈0,560 s`, limite atteinte avant `End=6 s` |
| `112_incompressibleMultiphaseVoF_damBreak4phaseLaminar` | Présent | Présent | Conforme OF13 | Validé — modèle laminaire, gravité et quatre phases, `End=6 s`, aucun `FOAM FATAL` |
| `113_incompressibleMultiphaseVoF_mixerVessel2DMRF` | Présent | Présent | Conforme OF13 | Validé — zone `rotor`, MRF `60 rpm`, quatre phases, `End=4 s`, aucun `FOAM FATAL` |
| `114_incompressibleVoF_DTCHull` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage 851 477 cellules, calcul MPI stable jusqu’à `Time≈340 s`, limite atteinte avant `End=4000 s` |
| `115_incompressibleVoF_DTCHullMoving` | Présent | Présent | Conforme OF13 | Accepté avec réserve — mouvement rigide Newmark confirmé jusqu’à `Time≈0,885 s`, fraction d’eau stable, limite avant `End=50 s` et Courant maximal ≈4,93 |
| `116_incompressibleVoF_DTCHullWave` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage 960 381 cellules, `setWaves` Stokes2 et mouvement MPI démarrés sans `FOAM FATAL`, limite avant `End=20 s` |
| `117_incompressibleVoF_angledDuct` | Présent | Présent | Conforme OF13 | Validé — VoF water/air, résidus alpha faibles, `End=10 s`, aucun `FOAM FATAL` |
| `118_incompressibleVoF_capillaryRise` | Présent | Présent | Conforme OF13 | Accepté avec réserve — montée capillaire stable jusqu’à `Time≈0,42635 s`, Courant max ≈0,20, limite avant `End=0,5 s` |
| `119_incompressibleVoF_cavitatingBullet` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage snappy 372 938 cellules, cavitation Schnerr–Sauer démarrée, résidus alpha faibles, limite avant `End=0,05 s` |
| `120_incompressibleVoF_climbingRod` | Présent | Présent | Conforme OF13 | Validé — maillage axisymétrique extrudé, stabilisation de phase, `End=25 s`, aucun `FOAM FATAL` |
| `121_incompressibleVoF_containerDischarge2D` | Présent | Présent | Conforme OF13 | Validé — vidange gravitaire liquid/gas, fraction liquide décroissante, `End=1,5 s`, aucun `FOAM FATAL` |
| `122_incompressibleVoF_damBreak` | Présent | Présent | Conforme OF13 | Validé — fusion laminaire/RAS générique, champs turbulents, `kEpsilon`, `End=1 s`, aucun `FOAM FATAL` |
| `123_incompressibleVoF_damBreak3D` | Présent | Présent | Conforme OF13 | Accepté avec réserve — `subsetMesh -noFields`, raffinement dynamique stable, progression à `Time≈0,369745 s`, aucun `FOAM FATAL` |
| `124_incompressibleVoF_damBreakTracer` | Présent | Présent | Conforme OF13 | Validé — deux champs `tracer.*`, deux `phaseScalarTransport`, `End=1 s`, aucun `FOAM FATAL` |
| `125_incompressibleVoF_floatingObject` | Présent | Présent | Conforme OF13 | Validé — mouvement rigide Newmark `Py/Ry`, raffinement dynamique, forces écrites, `End=6 s`, aucun `FOAM FATAL` |
| `126_incompressibleVoF_floatingObjectWaves` | Présent | Présent | Conforme OF13 | Accepté avec réserve — vagues Stokes5, mouvement Newmark et raffinement actifs jusqu’à `Time≈0,871186 s`, aucun `FOAM FATAL` |
| `127_incompressibleVoF_forcedUpstreamWave` | Présent | Présent | Conforme OF13 | Validé — vague Airy, six domaines MPI, `End=200 s`, `reconstructPar` réussi, aucun `FOAM FATAL` |
| `128_incompressibleVoF_mixerVessel` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage NCC de 1 006 267 cellules, 8 domaines et solveur MRF démarrés sans `FOAM FATAL`; limite avant premier temps écrit |
| `129_incompressibleVoF_mixerVessel2DMRF` | Présent | Présent | Conforme OF13 | Validé — zone `rotor`, MRF `60 rpm`, `End=4 s`, alpha bornée, aucun `FOAM FATAL` |
| `130_incompressibleVoF_mixerVesselHorizontal2D` | Présent | Présent | Conforme OF13 | Validé — zones `rotor/stator` à `±60 rpm`, NCC flux nul, `End=2 s`, aucun `FOAM FATAL` |
| `131_incompressibleVoF_nozzleFlow2D` | Présent | Présent | Conforme OF13 | Accepté avec réserve — `blockMesh`, `refineMesh` et VoF démarrés; progression à `Time≈3,19643e-05 s`, alpha bornée, aucun `FOAM FATAL` |
| `132_incompressibleVoF_parshallFlume` | Présent | Présent | Conforme OF13 | Validé — débit `1,0`, quatre domaines MPI, `End=250 s`, reconstruction réussie, aucun `FOAM FATAL` |
| `133_incompressibleVoF_planingHullW3` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage `.1` de 1 106 939 cellules, 16 domaines et Newmark démarrés sans `FOAM FATAL`; arrêt préventif pour pression mémoire avant premier temps écrit |
| `134_incompressibleVoF_propeller` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage, baffles et 45 816 couplages NCC terminés; VoF stable jusqu’à `Time≈0,00117 s`, aucun `FOAM FATAL`; endTime très coûteux |
| `135_incompressibleVoF_rotatingCube` | Présent | Présent | Conforme OF13 | Validé — NCC, rotation `-60 rpm`, raffinement dynamique, `End=2 s` et reconstruction `-cellProc` réussis, aucun `FOAM FATAL` |
| `136_incompressibleVoF_sloshingCylinder` | Présent | Présent | Conforme OF13 | Validé — maillage snappy de 33 568 cellules, multiMotion oscillant/rotatif, `End=0,5 s`, alpha bornée et aucun `FOAM FATAL` |
| `137_incompressibleVoF_sloshingTank2D` | Présent | Présent | Conforme OF13 | Validé — ressource blockMesh officielle, 1 360 cellules, mouvement SDA, `End=40 s`, alpha bornée et aucun `FOAM FATAL` |
| `138_incompressibleVoF_sloshingTank2D3DoF` | Présent | Présent | Conforme OF13 | Validé — ressource blockMesh officielle, SDA trois degrés de liberté, `End=40 s`, alpha bornée et aucun `FOAM FATAL` |
| `139_incompressibleVoF_sloshingTank3D` | Présent | Présent | Conforme OF13 | Accepté avec réserve — maillage 3D de 25 840 cellules, SDA stable jusqu’à `Time≈33,74/40 s`, alpha bornée, aucun `FOAM FATAL`; arrêt pour coût temps/disque |
| `140_incompressibleVoF_sloshingTank3D3DoF` | Présent | Présent | Conforme OF13 | Validé — maillage 3D adaptatif, SDA trois degrés de liberté, `End=10 s`, alpha bornée et aucun `FOAM FATAL` |
| `141_incompressibleVoF_sloshingTank3D6DoF` | Présent | Présent | Conforme OF13 | Accepté avec réserve — sixDoFMotion avec `6DoF.dat` officiel, maillage adaptatif stable jusqu’à `Time≈37,47/40 s`, alpha bornée et aucun `FOAM FATAL`; arrêt préventif pour coût temps/disque |
| `142_incompressibleVoF_testTubeMixer` | Présent | Présent | Conforme OF13 | Validé — maillage de 1 250 cellules, multiMotion à 60 rpm et 40 rad/s, `End=1 s`, alpha bornée et aucun `FOAM FATAL` |
| `143_incompressibleVoF_trayedPipe` | Présent | Présent | Conforme OF13 | Validé — maillage de 1 250 cellules, zone `wall`, 150 faces de baffles, `End=2 s`, alpha bornée et aucun `FOAM FATAL` |
| `144_incompressibleVoF_waterChannel` | Présent | Présent | Conforme OF13 | Validé — maillage de 8 000 cellules, kOmegaSST, flux entrant `-50`, `End=200 s`, fonctions `surfaceFieldValue` et aucun `FOAM FATAL` |
| `145_incompressibleVoF_wave` | Présent | Présent | Conforme OF13 | Validé — extrusion/raffinement, vague Airy, 6 domaines MPI, `End=200 s`, reconstruction réussie et aucun `FOAM FATAL` |
| `146_incompressibleVoF_wave3D` | Présent | Présent | Conforme OF13 | Accepté avec réserve — `blockMesh` 353 440 cellules et `refineMesh` jusqu’à 1 024 024 cellules; SIGTERM du sandbox avant `setWaves`, calcul et reconstruction |
| `147_incompressibleVoF_weirOverflow` | Présent | Présent | Conforme OF13 | Validé — débit d’entrée 75, modèle kEpsilon, débordement VoF stable, `End=60 s`, alpha bornée et aucun `FOAM FATAL` |
| `148_isothermalFilm_rivuletPanel` | Présent | Présent | Conforme OF13 | Accepté avec réserve — `filmWall`/`filmContactAngle`, Courant maximal ≈0,198, stable jusqu’à `Time≈2,508/5 s`, arrêt préventif pour coût temps |
| `149_isothermalFluid_potentialFreeSurfaceOscillatingBox` | Présent | Présent | Conforme OF13 | Validé — `subsetMesh -noFields`, surface libre oscillante à 1 Hz, fonction `poolHeight`, `End=20 s` et aucun `FOAM FATAL` |
| `150_legacy_financialFoam_europeanCall` | Présent | Présent | Conforme OF13 | Validé — champ financier `V`, paramètres strike/rate/volatilité importés, `End=0,5 s`, résidus de l’ordre de `10^-18` et aucun `FOAM FATAL` |
| `151_legacy_laplacianFoam_flange` | Présent | Présent | Conforme OF13 | Validé — conversion `flange.ans` à l’échelle 0,001, laplacien thermique jusqu’à `End=3 s`, exports Ensight/VTK et aucun `FOAM FATAL` |
| `152_legacy_rhoPorousSimpleFoam_angledDuctExplicit` | Présent | Présent | Conforme OF13 | Validé — ressource `blockMesh/angledDuct`, 22 000 cellules, zone Darcy–Forchheimer explicite, `End=1000 s` et aucun `FOAM FATAL` |
| `153_legacy_rhoPorousSimpleFoam_angledDuctImplicit` | Présent | Présent | Conforme OF13 | Validé — même maillage et zone Darcy–Forchheimer, formulation implicite conservée, `End=100 s` et aucun `FOAM FATAL` |
| `154_legacy_electrostaticFoam_chargedWire` | Présent | Présent | Conforme OF13 | Validé — champs `phi`/`rho`, `epsilon0` officiel, résidus électriques inférieurs à `10^-9`, `End=0,02 s` et aucun `FOAM FATAL` |
| `155_legacy_mhdFoam_hartmann` | Présent | Présent | Conforme OF13 | Validé — maillage MHD de 4 000 cellules, champ `B` imposé, erreur de divergence magnétique ≈`10^-9`, profil `Ux`, `End=2 s` et aucun `FOAM FATAL` |
| `156_legacy_adjointShapeOptimisationFoam_pitzDaily` | Présent | Présent | Conforme OF13 | Validé — champs primaux/adjoints, maillage pitzDaily, `adjointShapeOptimisationFoam`, `End=1000 s` et aucun `FOAM FATAL` |
| `157_legacy_icoFoam_elbow` | Présent | Présent | Conforme OF13 | Validé — conversion `elbow.msh` Fluent, `icoFoam` jusqu’à `End=10 s`, exports Fluent et aucun `FOAM FATAL` |
| `158_legacy_porousSimpleFoam_angledDuctExplicit` | Présent | Présent | Conforme OF13 | Validé — ressource `blockMesh/angledDuct`, 22 000 cellules, zone Darcy–Forchheimer, `porousSimpleFoam`, `End=200 s` et aucun `FOAM FATAL` |
| `159_legacy_porousSimpleFoam_angledDuctImplicit` | Présent | Présent | Conforme OF13 | Validé — même maillage et zone Darcy–Forchheimer, formulation implicite conservée, `porousSimpleFoam`, `End=100 s` et aucun `FOAM FATAL` |
| `160_legacy_shallowWaterFoam_squareBump` | Présent | Présent | Conforme OF13 | Validé — maillage 2D de 400 cellules, topographie `bump` initialisée par `setFields`, `shallowWaterFoam`, `End=100 s` et aucun `FOAM FATAL` |
| `161_legacy_dsmcFoam_freeSpacePeriodic` | Présent | Présent | Conforme OF13 | Validé — maillage à périodicité X/Y/Z, `dsmcInitialise`, environ 64 009 particules DSMC, `dsmcFoam`, `End=1e-3 s` et aucun `FOAM FATAL` |
| `162_legacy_dsmcFoam_freeSpaceStream` | Présent | Présent | Conforme OF13 | Accepté avec réserve — modèle `FreeStream` N2/O2, injection et collisions stables jusqu’à `Time≈0,005494/0,02 s`, avertissements non fatals et aucun `FOAM FATAL` |
| `163_legacy_dsmcFoam_supersonicCorner` | Présent | Présent | Conforme OF13 | Accepté avec réserve — décomposition DSMC à 4 domaines, argon FreeStream supersonique et collisions stables jusqu’à `Time≈0,000132/0,01 s`, avertissements non fatals et aucun `FOAM FATAL` |
| `164_legacy_dsmcFoam_wedge15Ma5` | Présent | Présent | Conforme OF13 | Validé — wedge DSMC à 4 domaines, FreeStream N2/O2 à vitesse supersonique, `dsmcFoam` jusqu’à `End=0,02 s`, reconstruction et aucun `FOAM FATAL` |
| `165_legacy_mdEquilibrationFoam_periodicCubeArgon` | Présent | Présent | Conforme OF13 | Accepté avec réserve — cube périodique `12x12x12`, 2 197 molécules d’argon, `mdEquilibrationFoam` stable jusqu’à `Time=4,675e-11/5e-11 s`, arrêt au plafond de 300 s et aucun `FOAM FATAL` |
| `166_legacy_mdFoam_nanoNozzle` | Présent | Présent | Conforme OF13 | Accepté avec réserve — nano-nozzle de 27 136 cellules, décomposition corrigée à 4 domaines, 110 197 molécules initialisées, `mdFoam` stable jusqu’à `Time=7e-15/2e-13 s`, arrêt pour coût et aucun `FOAM FATAL` |
| `168_movingMesh_SnakeRiverCanyon` | Présent | Présent | Conforme OF13 | Validé — maillage movingMesh `20x60x60`, surface `AcrossRiver`, calcul parallèle à 2 domaines jusqu’à `Time=25 s`, reconstruction des temps 5 à 25 et aucun `FOAM FATAL` |
| `169_multiRegion_CHT_VoFcoolingCylinder2D` | Présent | Présent | Conforme OF13 | Validé — régions `fluid/solid`, champs régionaux correctement placés sous `0/fluid` et `0/solid`, `foamMultiRun` jusqu’à `Time=5 s`, interfaces couplées et aucun `FOAM FATAL` |
| `170_multiRegion_CHT_circuitBoardCooling` | Présent | Présent | Conforme OF13 | Validé — pipeline `blockMesh/createZones/extrudeToRegionMesh/createBaffles` avec `wallPatchFields`, régions `fluid/baffle3D`, `foamMultiRun` jusqu’à `Time=5000 s` et aucun `FOAM FATAL` |
| `171_multiRegion_CHT_coolingCylinder2D` | Présent | Présent | Conforme OF13 | Validé — régions `fluid/solid`, séparation `splitMeshRegions -cellZones`, `foamMultiRun` jusqu’à `Time=20 s` et aucun `FOAM FATAL` |
| `172_multiRegion_CHT_coolingSphere` | Présent | Présent | Conforme OF13 | Validé — le chemin #171 `coolingSphere/templates` est traité via son parent exécutable, avec `foamSetupCHT`, quatre domaines, calcul parallèle jusqu’à `Time=1 s`, reconstruction multi-régions et aucun `FOAM FATAL` |
| `173_multiRegion_CHT_engine2Valve2D` | Présent | Présent | Conforme OF13 | Accepté avec réserve — cinq régions solides, 24 maillages fluides temporels, couples non conformes et `foamMultiRun -parallel`; progression stable jusqu’à `CAD≈279,9` au plafond de 300 s, aucun `FOAM FATAL` observé |
| `174_multiRegion_CHT_heatExchanger` | Présent | Présent | Conforme OF13 | Validé — régions `air/porous`, baffles et zones rotor MRF, `foamMultiRun -parallel` à 4 domaines jusqu’à `Time=2000 s`, reconstruction des deux régions et aucun `FOAM FATAL` |
| `176_multiRegion_CHT_misalignedDuct` | Présent | Présent | Conforme OF13 | Validé — séparation `fluid/solid`, nettoyage FoamPilot des `cellToRegion`, quatre couples non conformes, `foamMultiRun` jusqu’à `Time=20 s` et aucun `FOAM FATAL` |
| `177_multiRegion_CHT_multiphaseCoolingCylinder2D` | Présent | Présent | Conforme OF13 | Validé — cas sériel exact avec séparation `fluid/solid`, MULES eau/huile, somme des fractions égale à 1, couplage thermique et `foamMultiRun` jusqu’à `Time=5 s` sans `FOAM FATAL` |
| `178_multiRegion_CHT_notchedRoller` | Présent | Présent | Conforme OF13 | Validé — baffles, régions `fluid/solid/roller`, zone tournante `rotating`, couples non conformes, calcul parallèle à 4 domaines jusqu’à `Time=20 s`, reconstruction et aucun `FOAM FATAL` |
| `179_multiRegion_CHT_reverseBurner` | Présent | Présent | Conforme OF13 | Accepté avec réserve — régions `gas/solid`, initialisation parallèle N2/O2/CH4, chimie multicomposant et produits H2O/CO2 résolus jusqu’à `Time≈0,73 s` sur `6 s` au plafond de 300 s, aucun `FOAM FATAL` observé |
| `180_multiRegion_CHT_shellAndTubeHeatExchanger` | Présent | Présent | Conforme OF13 | Accepté avec réserve — import des cinq STL, maillage initial snappy sur 8 domaines et lecture des régions prévues; le premier `snappyHexMesh` est interrompu par SIGKILL au plafond de ressources avant les couches et le solveur, aucun `FOAM FATAL` observé |
| `181_multiRegion_CHT_wallBoiling` | Présent | Présent | Conforme OF13 | Accepté avec réserve — extrusion et séparation `fluid/solid`, phases `gas/liquid`, modèles `heatTransferLimitedPhaseChange` et `wallBoiling` actifs jusqu’à `Time≈1,93 s` sur `8 s` au plafond de 300 s, aucun `FOAM FATAL` observé |
| `182_multiRegion_film_VoFToFilm` | Présent | Présent | Conforme OF13 | Validé — maillage VoF, extrusion film de 1 mm, initialisation `alpha.liquid`, solveurs VoF/film couplés jusqu’à `Time=5 s` et aucun `FOAM FATAL` |
| `183_multiRegion_film_cylinder` | Présent | Présent | Conforme OF13 | Validé — Allrun parallèle, maillage fluid/film dans les quatre processeurs, particules absorbées par le film, espèces `N2/O2/H2O`, `foamMultiRun` jusqu’à `Time=20 s`, reconstruction des deux régions et aucun `FOAM FATAL` |
| `184_multiRegion_film_cylinderDripping` | Présent | Présent | Conforme OF13 | Validé — extrusion film de 1 mm, couplage multicomposant fluid/film, `New film detached parcels=1105`, calcul jusqu’à `Time=1 s` et aucun `FOAM FATAL` |
| `185_multiRegion_film_cylinderVoF` | Présent | Présent | Conforme OF13 | Validé — Allrun parallèle, maillage VoF/film dans quatre processeurs, phase `alpha.liquid`, injection/absorption de parcelles, calcul jusqu’à `Time=20 s`, reconstruction des régions et aucun `FOAM FATAL` |
| `186_multiRegion_film_hotBoxes` | Présent | Présent | Conforme OF13 | Accepté avec réserve — quatre boîtes sélectionnées par `subsetMesh`, film créé dans douze processeurs, échauffement jusqu’à environ 338 K et calcul stable vers `Time≈0,358 s` sur `2 s` au plafond de 300 s, aucun `FOAM FATAL` |
| `187_multiRegion_film_rivuletBox` | Présent | Présent | Conforme OF13 | Accepté avec réserve — régions `box/panel/film`, extrusions `0,002/0,01`, frontières mappées créées par `foamDictionary`, calcul parallèle stable jusqu’à `Time≈0,665 s` sur `5 s` au plafond de 300 s, aucun `FOAM FATAL` |
| `188_multiRegion_film_rivuletPanel` | Présent | Présent | Conforme OF13 | Validé — panel de 43 200 cellules, décomposition Scotch sur 4 domaines, extrusion film de `0,01`, calcul `foamMultiRun` jusqu’à `Time=5 s`, temps reconstruits `0,1` à `5`, reconstruction finale film/panel et aucun `FOAM FATAL` |
| `189_multiRegion_film_splashPanel` | Présent | Présent | Conforme OF13 | Validé — maillage fluid de 4 000 cellules, extrusion film intrudée de `0,002`, couplages `mappedExtrudedWall`/`filmWall`/`mappedFilmSurface`, calcul sériel jusqu’à `Time=1 s`, 1 000 splash parcels, 1 819 absorptions et aucun `FOAM FATAL` |
| `190_multicomponentFluid_DLR_A_LTS` | Présent | Présent | Conforme OF13 | Validé — conversion Chemkin GRI30, maillage et `setFields`, décomposition forcée à 6 domaines, calcul `foamRun` LTS jusqu’à `Time=10000 s`, reconstruction des temps 1000–10000 et aucun `FOAM FATAL` |
| `191_multicomponentFluid_SandiaD_LTS` | Présent | Présent | Conforme OF13 | Accepté avec réserve — préparation et phase sans chimie jusqu’à `Time=1500 s` validées, phase chimique stable jusqu’à `Time≈2869 s` sur `5000 s` au plafond de 300 s, aucun `FOAM FATAL`; reconstruction finale à poursuivre |
| `192_multicomponentFluid_aachenBomb` | Présent | Présent | Conforme OF13 | Accepté avec réserve — Chemkin, maillage et décomposition Zoltan `2×2×3` validés, calcul parallèle stable jusqu’à `Time≈3,84e-4 s` sur `0,01 s` au plafond de 300 s, aucun `FOAM FATAL`; reconstruction finale à poursuivre |
| `193_multicomponentFluid_counterFlowFlame2D` | Présent | Présent | Conforme OF13 | Validé — maillage 2D `100×40×1`, réaction méthane simplifiée, calcul jusqu’à `Time=0,5 s`, Courant maximal ≈0,398 et aucun `FOAM FATAL` |
| `194_multicomponentFluid_counterFlowFlame2DLTS` | Présent | Présent | Conforme OF13 | Validé — maillage counter-flow 2D, schéma `localEuler`, calcul LTS jusqu’à `Time=1000 s`, erreurs de continuité de l’ordre de `10^-8` à `10^-10` et aucun `FOAM FATAL` |
| `195_multicomponentFluid_counterFlowFlame2DLTS_GRI_TDAC` | Présent | Présent | Conforme OF13 | Validé — décomposition hiérarchique `2×2×1`, chimie GRI sous TDAC, calcul LTS parallèle jusqu’à `Time=1500 s`, reconstruction des temps 20–1500 et aucun `FOAM FATAL` |
| `196_multicomponentFluid_counterFlowFlame2D_GRI` | Présent | Présent | Conforme OF13 | Accepté avec réserve — décomposition hiérarchique/Zoltan `6×2×1`, calcul GRI stable jusqu’à `Time≈0,151 s` sur `0,5 s` au plafond de 300 s, Courant maximal sous `0,4` et aucun `FOAM FATAL`; reconstruction finale à poursuivre |
| `197_multicomponentFluid_counterFlowFlame2D_GRI_TDAC` | Présent | Présent | Conforme OF13 | Validé — décomposition hiérarchique `2×2×1`, chimie GRI sous TDAC, calcul parallèle jusqu’à `Time=0,5 s`, reconstruction des temps 0,05–0,5 et aucun `FOAM FATAL` |
| `198_multicomponentFluid_filter` | Présent | Présent | Conforme OF13 | Validé — `blockMesh`, zone `filter`, `createBaffles`, calcul jusqu’à `Time=5 s`, Courant maximal ≈0,995 et aucun `FOAM FATAL` |
| `199_multicomponentFluid_lockExchange` | Présent | Présent | Conforme OF13 | Validé — espèces `water/sludge`, zone sludge initialisée par `setFields`, calcul jusqu’à `Time=100 s`, Courant maximal inférieur à 0,47 et aucun `FOAM FATAL` |
| `200_multicomponentFluid_membrane` | Présent | Présent | Conforme OF13 | Validé — STL `membrane.stl`, `snappyHexMesh` à 18 632 cellules, 960 baffles, patches mappés `membranePipe/membraneSleeve`, calcul jusqu’à `Time=10 s` et aucun `FOAM FATAL` |
| `201_multicomponentFluid_nc7h16` | Présent | Présent | Conforme OF13 | Validé — `zeroDimensionalMesh`, conversion Chemkin NC7H16, `massFractions`, calcul jusqu’à `Time=0,001 s`, erreurs de continuité de l’ordre de `10^-15` et aucun `FOAM FATAL` |
| `202_multicomponentFluid_parcelInBox` | Présent | Présent | Conforme OF13 | Validé — nuage 3D `cloud` avec un parcel, transport de `H2O`, calcul jusqu’à `Time=0,5 s`, erreurs de continuité faibles et aucun `FOAM FATAL` |
| `203_multicomponentFluid_simplifiedSiwek` | Présent | Présent | Conforme OF13 | Validé — clouds 2D `coalCloud`/`limestoneCloud`, radiation et réactions de surface, 27 et 18 parcels, calcul jusqu’à `Time=0,5 s`, Courant maximal <0,044 et aucun `FOAM FATAL` |
| `204_multicomponentFluid_smallPoolFire2D` | Présent | Présent | Conforme OF13 | Accepté avec réserve — `blockMesh`, `createPatch` validés, feu de nappe stable jusqu’à `Time≈2,684 s` sur `3 s` au plafond de 300 s, Courant maximal proche de `0,5` et aucun `FOAM FATAL` |
| `205_multicomponentFluid_smallPoolFire3D` | Présent | Présent | Conforme OF13 | Accepté avec réserve — préparation et décomposition hiérarchique `1×2×2` validées, calcul parallèle stable jusqu’à `Time≈2,726 s` sur `4 s` au plafond de 300 s, Courant maximal proche de `0,6` et aucun `FOAM FATAL`; reconstruction finale à poursuivre |
| `206_multicomponentFluid_verticalChannel` | Présent | Présent | Conforme OF13 | Accepté avec réserve — `potentialFoam`, suppression de `0/phi` et cloud injecté validés; calcul stable jusqu’à `Time≈0,246 s` sur `0,5 s` au plafond de 300 s, environ 10 150 parcels présents et aucun `FOAM FATAL`; `particleTracks` reste à exécuter |
| `207_multicomponentFluid_verticalChannelLTS` | Présent | Présent | Conforme OF13 | Validé — `localEuler`, `potentialFoam`, cloud `cloudTracks`, calcul jusqu’à `Time=300 s`, `steadyParticleTracks` et VTK de trajectoires écrits de 0 à 300 s, sans `FOAM FATAL` |
| `208_multicomponentFluid_verticalChannelSteady` | Présent | Présent | Conforme OF13 | Validé — `steadyState`, `potentialFoam`, cloud multiphasique, calcul jusqu’à `Time=500 s`, `steadyParticleTracks` et VTK de trajectoires jusqu’à 500 s, sans `FOAM FATAL` |
| `209_multiphaseEuler_Grossetete` | Présent | Présent | Conforme OF13 | Validé — phases gaz/liquide, extrusion wedge, MULES avec fractions bornées, calcul jusqu’à `Time=2 s`, Courant maximal proche de 0,25 et aucun `FOAM FATAL` |

## Conclusion

Aucun équivalent non vide déclaré dans la matrice ne possède de dossier ou de `run.py` manquant. Les runners suivis jusqu’à `legacy/lagrangian/dsmcFoam/wedge15Ma5` sont effectivement présents. Cette tranche ajoute les runners des ordres 46 à 55; `column` et `TJunction` sont validés jusqu’à leur `endTime`, plusieurs cas particulaires et drift-flux sont acceptés avec réserve après progression sans erreur fatale visible, et `tank3D` reste en cours de calcul.

Pour les prochains tutoriels, le contrôle doit être relancé après chaque création de runner et avant le marquage `Validé` dans la matrice. Toute nouvelle fonction ajoutée pour permettre un runner doit également être inscrite dans [`foampilot_api_evolution_openfoam13.md`](foampilot_api_evolution_openfoam13.md).
