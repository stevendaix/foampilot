# Contrôle de présence des runners FoamPilot — OpenFOAM 13

## Méthode

Le contrôle compare les équivalents FoamPilot déclarés dans [`openfoam13_foampilot_integration.md`](openfoam13_foampilot_integration.md) avec le chemin attendu `foampilot/tutorials/<équivalent>/run.py`. Un tutoriel est considéré comme présent uniquement lorsque son dossier et son fichier `run.py` existent effectivement dans le dépôt.

La vérification effectuée le 26 août 2026 confirme que les **48 runners actuellement présents** couvrent les équivalents déclarés dans la matrice ainsi que les runners récemment ajoutés jusqu’à `fluid/squareBendLiq`.

## Résultats

| Équivalent FoamPilot | Dossier | `run.py` | Présence | Statut matrice |
|---|---|---|---|---|
| `01_cavity_laminar` | Présent | Présent | Conforme | Validé |
| `02_simpleCar_turbulent` | Présent | Présent | Conforme | Validé |
| `03_pitzDaily_step` | Présent | Présent | Conforme | Validation longue partielle |
| `04_damBreak_multiphase` | Présent | Présent | Conforme | Validé |
| `05_scalarTransport` | Présent | Présent | Conforme | Validé |
| `06_buildingAero` | Présent | Présent | Conforme | Validé |
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

## Conclusion

Aucun équivalent non vide déclaré dans la matrice ne possède de dossier ou de `run.py` manquant. Les runners suivis jusqu’à `48_fluid_squareBendLiq` sont effectivement présents. `prism`, `shockTube`, `squareBend` et `squareBendLiq` sont validés sous OF13; `nacaAirfoil` est non validé après expiration de délai et `roomHeating` reste partiellement validé, avec steady terminé et transitoire interrompu avant `End=6000 s`.

Pour les prochains tutoriels, le contrôle doit être relancé après chaque création de runner et avant le marquage `Validé` dans la matrice. Toute nouvelle fonction ajoutée pour permettre un runner doit également être inscrite dans [`foampilot_api_evolution_openfoam13.md`](foampilot_api_evolution_openfoam13.md).
