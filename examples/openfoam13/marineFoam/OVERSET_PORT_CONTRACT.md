# Contrat d’implémentation overset Foundation 13

Ce document fixe la frontière entre le prototype overset validé et le runtime nécessaire à une reproduction fidèle de `DTCMoving_Overset`.

## Compatibilité déclarée

Le driver `marineFoam` Foundation 13 supporte actuellement le mouvement rigide natif, la VoF `incompressibleVoF`, les zones MRF et les modèles `fvModels`. Il ne supporte pas encore l’overset matriciel.

Un cas ne doit être déclaré `foundation13_overset` que lorsque les composants suivants sont disponibles :

| Composant | Contrat minimal |
|---|---|
| Maillage | deux ensembles de cellules pouvant se déplacer sans raccordement conforme |
| Zone | champ `zoneID` consécutif, persistant et mis à jour après mouvement |
| Classification | statuts calculated, interpolated/acceptor et hole |
| Donneurs | au moins un stencil valide par accepteur, poids finis et somme égale à 1 |
| Patches | patch overset et conditions de champ dédiées |
| Matrices | insertion de l’interpolation et solveurs asymétriques |
| VoF | transport de `alpha` borné et conservation du volume de phase |
| Mouvement | déplacement du maillage mobile puis reconstruction de la connectivité |
| MPI | résultat cohérent en série et en parallèle |

## Correspondance OpenCFD à porter

La référence OpenCFD fournit `dynamicOversetFvMesh`, `oversetFvMeshBase`, `cellCellStencilObject`, `setCellMask`, `setInterpolatedCells`, `oversetAdjustPhi` et `oversetPatchPhiErr`. Ces symboles sont appelés par `overInterDyMFoam` et n’existent pas sous les mêmes noms dans Foundation 13.

Le portage Foundation 13 doit donc fournir des équivalents nouveaux, intégrés au framework `fvMesh`/`fvMeshMover`/`fvMeshTopoChanger`. Un simple `fvModel`, une source de quantité de mouvement ou un alias d’exécutable ne satisfait pas ce contrat.

## Gate de non-régression

Tant que tous les composants ne sont pas disponibles, le validateur doit rejeter les dictionnaires `dynamicOversetFvMesh`, `overInterDyMFoam` et les patches `overset` présentés comme Foundation 13 natifs. Les cas `DTCHullMoving` avec mover Foundation 13 restent valides, mais sont des cas morphing/mouvement de maillage et non des cas overset.

## Tests d’acceptation

Le portage sera accepté seulement après réussite d’un cas 2D analytique avec translation, interpolation exacte d’un champ constant, erreur contrôlée sur un champ linéaire, absence d’accepteur sans donneur, conservation de `alpha`, comparaison série/2-processeurs et exécution DTC réduite sans erreur de matrice.
