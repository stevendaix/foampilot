# Validation exécutée des tutoriels Tobias — 25 août 2026

Cette validation a été exécutée dans le dépôt `stevendaix/foampilot` avec OpenFOAM Foundation 13 chargé via `FOAM_BASHRC` ou `WM_PROJECT_DIR`. Les runners Python ont été lancés avec `PYTHONPATH=foampilot/src`; pour le contrôle principal, le cas `2d_rotational_axis_symmetric` a également été lancé sans sourcer manuellement OpenFOAM dans le shell, à condition que l’environnement Foundation 13 soit fourni explicitement. Les commandes FoamPilot réutilisent alors cet environnement sans dépendre d’un chemin machine.

## Résultats observés

| Cas | Code retour | Résultat | Preuve observée |
| --- | ---: | --- | --- |
| `2d_rotational_axis_symmetric` | 0 | Validé | `blockMesh`, `surfaceFeatures`, `snappyHexMesh`, `extrudeMesh`, `createPatch`; `160590` cellules; `Finished meshing without any errors`; `End` |
| `2d_ami_ncc` | 0 | Validé | `ideasUnvToFoam`, `snappyHexMesh`, baffles, couples non conformes et `foamRun`; `140321` cellules; `End` |
| `catalystHeatUp` | 0 | Validé | `splitMeshRegions`, `createPatch` pour SCR1/SCR2 et `foamMultiRun`; `Finished meshing without any errors`; `End` |
| `pitot_tube` | 0 | Validé | maillage `206180` cellules; `changeDictionary`, `extrudeMesh` et `foamRun`; `Finished meshing without any errors`; `End` |
| `rotatingRotorNCC` | 0 | Validé | `snappyHexMesh`, baffles, `createNonConformalCouples` et `foamRun`; `265756` cellules; `End` |
| `snappy_feature_edge_refinement` | 0 | Validé | trois variantes de feature edges; variante représentative `85067` cellules; `Finished meshing without any errors`; `End` |
| `snappy_sphere_and_layer` | 0 | Validé | `48940` cellules avant couches, `53404` après couches; `Finished meshing without any errors`; `End` |
| `thin_gap_meshing` | 0 | Validé | transformations d’échelle, `snappyHexMesh`, retour d’échelle et `foamRun`; `Finished meshing without any errors`; `End` |

Les mêmes sorties de calcul et de maillage ont été contrôlées dans les répertoires `case` générés. Les fichiers structurants `system/controlDict`, `system/fvSchemes`, `system/fvSolution`, `constant` et `0` sont présents pour les cas réussis.

## Cas non validés dans cette passe

| Cas | Code retour | Diagnostic |
| --- | ---: | --- |
| `adaptive_mesh_refinement` | 1 | Le runner attend `cad/`, absent du clone courant. |
| `cell_zone_generation` | 1 | Le runner attend `cad/`, absent du clone courant. |
| `falling_droplets` | 1 | Le runner attend `cad/`, absent du clone courant. |
| `fluidic_oscillator` | 1 | Le runner attend `cad/`, absent du clone courant. |
| `magnus_effect` | 1 | Le runner attend `cad/`, absent du clone courant. |
| `meshing_pipe_45deg` | 1 | Le runner attend `cad/`, absent du clone courant. |
| `meshing_pipe_90deg` | 1 | Le runner attend `cad/`, absent du clone courant. |
| `dakotaTeslaOneWayValve2D` | 1 | Exécution bloquée par l’exécutable externe `dakota` absent. |
| `battery_cooling` | 124 | `snappyHexMesh` n’a pas terminé dans le délai de smoke test de 240 s. |
| `combustion_chamber` | 124 | `snappyHexMesh` n’a pas terminé dans le délai de smoke test de 240 s. |

Ces résultats ne transforment pas les cas bloqués en cas validés. Ils indiquent les prérequis restant à récupérer ou la nécessité d’un délai de calcul adapté.

## Nouveau portage exécuté

Le tutoriel source `fluentMeshForCHTSolver` a été porté dans `fluent_mesh_for_cht`. Le runner FoamPilot écrit les dictionnaires source avec `write_raw`, copie `cad/fluentMesh.cas`, puis exécute `fluentMeshToFoam -writeSets`, `topoSet -constant` et `splitMeshRegions -cellZonesOnly -overwrite` avec OpenFOAM 13. Le lancement s’est terminé avec le code retour 0. Les deux régions produites ont ensuite été contrôlées par `checkMesh` : la région `fluid` contient `128657` cellules et la région `solid` `73311` cellules ; les deux journaux terminent par `Mesh OK.` et `End`.

## Deuxième portage préparé

Le tutoriel source `fanRotationAndNCC` a été adapté dans `fan_rotation_ncc`. Le runner FoamPilot reproduit la séquence OpenFOAM 13 `ideasUnvToFoam`, `snappyHexMesh`, `createBaffles`, `splitBaffles`, `createNonConformalCouples`, `renumberMesh`, puis le calcul parallèle via `run_parallel(4)`. La génération des dictionnaires et des géométries BREP fonctionne. L’exécution s’arrête volontairement avec un diagnostic explicite, car `cad/backgroundMesh.unv` a été retiré du dépôt GitHub de Tobias ; le cas complet doit être téléchargé depuis Holzmann CFD avant de lancer le maillage.

## Troisième portage préparé

Le tutoriel source `arbitraryWaterPump` a été adapté dans `arbitrary_water_pump`. Le runner reproduit la préparation VOF avec `ideasUnvToFoam`, `snappyHexMesh`, `createBaffles`, `setFields` et le chemin parallèle FoamPilot. Les dictionnaires `alpha.water`, `phaseProperties`, `physicalProperties.*`, `momentumTransport` et `setFieldsDict` sont conservés. Comme pour le cas source, l’exécution attend `cad/backgroundMesh.unv`; cet asset est absent du dépôt GitHub Tobias et le runner arrête maintenant le cas avec un message explicite avant tout appel OpenFOAM.

## Quatrième portage préparé

Le tutoriel `dakotaGeometricVariation` a été adapté dans `dakota_geometric_variation`. Le maillage OpenFOAM 13 se génère maintenant avec `blockMesh`, `extrudeMesh`, `createPatch` et `renumberMesh`. Le portage a également corrigé une incompatibilité de dictionnaire : OpenFOAM 13 exige `nLayers` et `expansionRatio` dans `linearNormalCoeffs`, alors que le fichier source OpenFOAM 12 les plaçait au niveau racine. La phase d’optimisation reste bloquée après le maillage car l’exécutable `dakota` n’est pas installé dans l’environnement.

## Cinquième portage préparé

Le tutoriel `solarChimney` a été adapté dans `solar_chimney`. Le runner conserve les dictionnaires thermiques, de radiation et de convection naturelle, puis prévoit `ideasUnvToFoam`, `snappyHexMesh` et un lancement parallèle FoamPilot. La génération des fichiers fonctionne ; le lancement OpenFOAM 13 est bloqué avant le maillage par l’absence de `cad/backgroundMesh.unv` dans le dépôt GitHub source.

## Sixième portage préparé

Le tutoriel `TEGModule` a été adapté dans `teg_module` avec un runner partagé pour `testDevice` et `optimizedDevice`. Le module C++ `solverTEGModule` compile avec succès contre OpenFOAM 13 après ajout des quatre méthodes abstraites introduites par l’interface `solver` 13 : `momentumTransportPredictor`, `thermophysicalTransportPredictor`, `momentumTransportCorrector` et `thermophysicalTransportCorrector`. Les deux runners génèrent leurs dictionnaires et s’arrêtent avant le maillage faute de `cad/backgroundMesh.unv`; le solveur utilisateur `TEGFoam` devra aussi être exposé dans l’environnement avant le calcul.

## Septième portage préparé

Le tutoriel `meshingAHelix` a été adapté dans `meshing_a_helix`. Le runner conserve les deux passes de `snappyHexMesh` — une passe avec couches puis une passe sans couches —, les dictionnaires `meshQualityDict.layer` et `meshQualityDict.normal`, la reconstruction et le `checkMesh` final. Le lancement parallèle de `snappyHexMesh` est appelé explicitement via `mpirun`, car `FoamPilot.run_parallel` est destiné aux calculs `foamRun`. La génération s’arrête avant OpenFOAM faute de `cad/backgroundMesh.unv`, `layer_orig.stl` et `regionSTL_orig.stl` dans le clone Tobias.

## Huitième portage préparé

Le tutoriel `sneezingSimulation` a été adapté dans `sneezing_simulation`. Le runner conserve les champs `T`, `U`, `p`, les propriétés de nuage, les données CSV/ODS/PNG de distribution et la séquence `ideasUnvToFoam`, décomposition, `snappyHexMesh` parallèle, initialisation des champs, `renumberMesh` et `foamRun`. La génération du cas fonctionne ; l’exécution s’arrête avant le maillage faute de `cad/backgroundMesh.unv`, absent du dépôt Tobias.

## Neuvième portage préparé

Le tutoriel `arbitraryRotatingInletNCC` a été adapté dans `arbitrary_rotating_inlet_ncc`. Le runner reproduit la séquence OpenFOAM 13 `ideasUnvToFoam`, `snappyHexMesh`, `createPatch`, `createNonConformalCouples`, `topoSet`, `renumberMesh` et `foamRun`. La compilation Python et la génération des dictionnaires sont validées ; l’exécution s’arrête proprement avant le maillage car `cad/backgroundMesh.unv` n’est pas inclus dans le dépôt Tobias.

## Dixième portage préparé

Le tutoriel `ginTonicCHT` a été adapté dans `gin_tonic_cht`. Le runner conserve les trois régions `ginTonic`, `iceCube1` et `iceCube2`, les dictionnaires CHT par région, les changements de conditions aux limites et les propriétés thermiques fluides/solides. Le lancement multi-régions utilise explicitement `foamMultiRun` avec une décomposition par région, car `FoamPilot.run_parallel` est dédié à `foamRun`. La génération est validée ; l’exécution OpenFOAM 13 attend le `backgroundMesh.unv` absent du dépôt source.

## Onzième portage préparé

Le tutoriel `kaplanTurbineNCC` a été adapté dans `kaplan_turbine_ncc`. Le runner conserve les dictionnaires 6-DoF, les variantes de maillage, la séquence parallèle `snappyHexMesh`, reconstruction, `createNonConformalCouples`, renumérotation et `foamRun`. La génération et la compilation Python sont valides ; l’exécution OpenFOAM 13 reste conditionnée par `cad/backgroundMesh.unv`, absent du dépôt Tobias.

## Douzième portage préparé

Le tutoriel `suzannesHead` a été adapté dans `suzannes_head`. Le runner conserve le maillage parallèle `snappyHexMesh`, le `checkMesh` parallèle, la mise à l’échelle, la renumérotation et le calcul stationnaire `foamRun` sous OpenFOAM 13. La génération et la compilation Python sont valides ; l’exécution attend `cad/backgroundMesh.unv`, absent du dépôt Tobias.

## Treizième portage préparé

Le tutoriel `verticalAxialWindTurbineNCC` a été adapté dans `vertical_axial_wind_turbine_ncc`. Le runner conserve la génération des features, `snappyHexMesh`, `extrudeMesh`, le changement des conditions, la création des couples NCC et le calcul parallèle `foamRun` 6-DoF. La génération et la compilation Python sont valides ; l’exécution OpenFOAM 13 attend le `backgroundMesh.unv` absent du dépôt Tobias.

## Remplacement Python de Dakota

Le script `dakota_geometric_variation/run_python_optimization.py` remplace l’interface Dakota par une campagne Python reproductible. Par défaut, il reproduit les dix échantillons Latin Hypercube de `dakota.in`, avec la graine `124523`, les variables `angle1` et `angle2` dans `[0, 180]` degrés et `length` dans `[0.005, 0.03]` m. Chaque évaluation possède un répertoire isolé, génère les STL de baffles, modifie `blockMeshDict`, lance le workflow FoamPilot/OpenFOAM 13 et écrit les résultats dans `python_optimization.csv` et `python_optimization.json`.

Les deux réponses Dakota sont conservées : `objective_average = abs(320 - Taverage)` et `objective_distribution = abs(Tmax - Tmin)`. Le meilleur point est sélectionné selon l’objectif moyen puis la dispersion. Le script accepte `--samples`, `--seed`, `--target-temperature` et `--keep-runs`. Une campagne courte a été vérifiée ; elle produit un échec explicite avec les chemins manquants `baffle1_original.stl` et `baffle2_original.stl`, sans lancer de calcul incomplet.

## Corrections FoamPilot incluses

La classe `OpenFOAMDictAddFile` fournit désormais `write_raw`, qui conserve un header `FoamFile` existant, ajoute un header standard lorsqu’il manque, crée les dossiers parents et écrit sans déformer la syntaxe OpenFOAM originale. `BaseSolver.run_command`, ainsi que les chemins sériel et parallèle de `run_simulation`, chargent l’environnement OpenFOAM 13 avant l’exécution. Les imports CAD optionnels ne bloquent plus l’import de l’API OpenFOAM lorsque `jupyter_cadquery` n’est pas installé.

Les tests ciblés passent avec **6 tests réussis** : helpers OpenFOAM 13, writer brut, propagation de l’environnement, système et constantes.

## Références

[1]: https://openfoam.org/download/13-ubuntu/ — OpenFOAM Foundation, « Download v13 | Ubuntu ».

[2]: https://github.com/stevendaix/foampilot/tree/main/examples%2Ftobias_tutorial — Répertoire des tutoriels Tobias dans FoamPilot.

[3]: https://github.com/stevendaix/foampilot/tree/main/foampilot%2Fsrc%2Ffoampilot — Module source FoamPilot contrôlé.
