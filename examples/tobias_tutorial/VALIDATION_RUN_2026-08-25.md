# Validation exécutée des tutoriels Tobias — 25 août 2026

Cette validation a été exécutée dans le dépôt `stevendaix/foampilot` avec OpenFOAM Foundation 13 installé sous `/opt/openfoam13`. Les runners Python ont été lancés avec `PYTHONPATH=foampilot/src`; pour le contrôle principal, le cas `2d_rotational_axis_symmetric` a également été lancé sans sourcer manuellement OpenFOAM dans le shell. Le correctif de `BaseSolver` charge alors automatiquement `/opt/openfoam13/etc/bashrc` pour les commandes FoamPilot.

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

## Corrections FoamPilot incluses

La classe `OpenFOAMDictAddFile` fournit désormais `write_raw`, qui conserve un header `FoamFile` existant, ajoute un header standard lorsqu’il manque, crée les dossiers parents et écrit sans déformer la syntaxe OpenFOAM originale. `BaseSolver.run_command`, ainsi que les chemins sériel et parallèle de `run_simulation`, chargent l’environnement OpenFOAM 13 avant l’exécution. Les imports CAD optionnels ne bloquent plus l’import de l’API OpenFOAM lorsque `jupyter_cadquery` n’est pas installé.

Les tests ciblés passent avec **6 tests réussis** : helpers OpenFOAM 13, writer brut, propagation de l’environnement, système et constantes.

## Références

[1]: https://openfoam.org/download/13-ubuntu/ — OpenFOAM Foundation, « Download v13 | Ubuntu ».

[2]: https://github.com/stevendaix/foampilot/tree/main/examples%2Ftobias_tutorial — Répertoire des tutoriels Tobias dans FoamPilot.

[3]: https://github.com/stevendaix/foampilot/tree/main/foampilot%2Fsrc%2Ffoampilot — Module source FoamPilot contrôlé.
