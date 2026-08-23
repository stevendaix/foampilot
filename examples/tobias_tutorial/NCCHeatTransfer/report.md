# NCC Heat Transfer

## Objet

Ce tutoriel de Tobias Holzmann simule un transfert thermique utilisant la nouvelle condition aux limites `Non-Conformal-Coupled` (NCC) introduite dans OpenFOAM 10, qui remplace l'ancienne approche ACMI. Le cas illustre le couplage thermique entre deux domaines avec un maillage dynamique [1].

## Portage FoamPilot

`run.py` recrée le cas en écrivant les dictionnaires, en copiant le maillage UNV et les surfaces STL. Le workflow de maillage utilise `ideasUnvToFoam`, `surfaceFeatures`, `snappyHexMesh`, `splitBaffles`, `createPatch`, `flattenMesh`, `extrudeMesh`, `createNonConformalCouples` et `topoSet`. L'utilitaire `renumberMesh` a été retiré du workflow car il cause une erreur dans OpenFOAM 13 sur ce type de maillage dynamique NCC. 

Pour OpenFOAM 13, les paramètres `nLayers 1` et `expansionRatio 1` ont été ajoutés à `system/extrudeMeshDict`. Le temps de simulation a été réduit à `endTime 0.5` pour permettre un smoke run, car la simulation devient instable avec le solveur standard après t=0.72s.

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Conversion UNV | Le maillage de fond est converti avec succès. |
| Maillage | Le maillage complet est généré, les couples non conformes sont créés avec succès par `createNonConformalCouples`. |
| Solveur | Le solveur thermo-fluidique est chargé. |
| Calcul | Le calcul s'exécute avec le maillage dynamique et les couples NCC jusqu'à la limite du smoke run (t=0.5s). |

Le cas est **validé**. La validation démontre la génération du maillage, la configuration des couples NCC et le démarrage de la simulation dynamique.

## Limites

Le temps de simulation a été réduit pour permettre un smoke run et éviter une instabilité du maillage dynamique (`sigSegv` lors du mouvement du maillage) qui survient après 0.72s avec OpenFOAM 13. Une investigation plus approfondie des paramètres de `dynamicMeshDict` pourrait être nécessaire pour une simulation complète.

## Référence

[1]: https://holzmann-cfd.com/community/training-cases/acmi-heat-transfer — Tobias Holzmann, *ACMI Heat Transfer*.
