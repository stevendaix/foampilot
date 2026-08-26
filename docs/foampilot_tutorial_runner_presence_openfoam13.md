# Contrôle de présence des runners FoamPilot — OpenFOAM 13

## Méthode

Le contrôle compare les équivalents FoamPilot déclarés dans [`openfoam13_foampilot_integration.md`](openfoam13_foampilot_integration.md) avec le chemin attendu `foampilot/tutorials/<équivalent>/run.py`. Un tutoriel est considéré comme présent uniquement lorsque son dossier et son fichier `run.py` existent effectivement dans le dépôt.

La vérification effectuée le 26 août 2026 confirme que les **61 runners actuellement présents** couvrent les équivalents déclarés dans la matrice ainsi que les runners récemment ajoutés jusqu’à `incompressibleFluid/wingMotion2D_steady`.

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

## Conclusion

Aucun équivalent non vide déclaré dans la matrice ne possède de dossier ou de `run.py` manquant. Les runners suivis jusqu’à `incompressibleMultiphaseVoF/damBreak4phaseLaminar` sont effectivement présents. Cette tranche ajoute les runners des ordres 46 à 55; `column` et `TJunction` sont validés jusqu’à leur `endTime`, plusieurs cas particulaires et drift-flux sont acceptés avec réserve après progression sans erreur fatale visible, et `tank3D` reste en cours de calcul.

Pour les prochains tutoriels, le contrôle doit être relancé après chaque création de runner et avant le marquage `Validé` dans la matrice. Toute nouvelle fonction ajoutée pour permettre un runner doit également être inscrite dans [`foampilot_api_evolution_openfoam13.md`](foampilot_api_evolution_openfoam13.md).
