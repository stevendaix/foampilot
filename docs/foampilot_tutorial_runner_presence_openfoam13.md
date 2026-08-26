# Contrôle de présence des runners FoamPilot — OpenFOAM 13

## Méthode

Le contrôle compare les équivalents FoamPilot déclarés dans [`openfoam13_foampilot_integration.md`](openfoam13_foampilot_integration.md) avec le chemin attendu `foampilot/tutorials/<équivalent>/run.py`. Un tutoriel est considéré comme présent uniquement lorsque son dossier et son fichier `run.py` existent effectivement dans le dépôt.

La vérification effectuée le 26 août 2026 confirme que les **32 runners actuellement présents** couvrent les équivalents déclarés dans la matrice ainsi que les runners récemment ajoutés jusqu’à `fluid/externalCoupledCavity`.

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

## Conclusion

Aucun équivalent non vide déclaré dans la matrice ne possède de dossier ou de `run.py` manquant. Les runners suivis jusqu’à `32_fluid_externalCoupledCavity` sont effectivement présents. Les cas ajoutés dans cette tranche sont validés sous OF13 aux temps de fin ou critères de convergence documentés dans la matrice.

Pour les prochains tutoriels, le contrôle doit être relancé après chaque création de runner et avant le marquage `Validé` dans la matrice. Toute nouvelle fonction ajoutée pour permettre un runner doit également être inscrite dans [`foampilot_api_evolution_openfoam13.md`](foampilot_api_evolution_openfoam13.md).
