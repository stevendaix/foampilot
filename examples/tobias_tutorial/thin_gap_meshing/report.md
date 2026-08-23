# Thin Gap Meshing

## Objet

Ce tutoriel de Tobias Holzmann montre une stratégie de maillage de petits interstices avec `snappyHexMesh`. Une surface triangulée distincte est utilisée pour la région de raffinement du gap. Après la génération du maillage, un calcul stationnaire interne est réalisé avec `simpleFoam` dans le cas source [1].

## Portage FoamPilot

Le runner écrit les dictionnaires, les champs et les surfaces par FoamPilot, puis exécute toutes les opérations OpenFOAM via `Solver.run_command` : conversion du maillage de fond, mise à l’échelle du maillage et des STL, `snappyHexMesh`, retour à l’échelle originale et calcul avec `foamRun`/`incompressibleFluid`.

Le maillage et les surfaces STL ne figuraient pas dans le dépôt Git source ; ils ont été récupérés dans l’archive officielle OpenFOAM 12 du tutoriel. Le cas source utilise `simpleFoam`; le portage conserve le solveur incompressible déclaré dans `controlDict` et appelle l’interface générique `foamRun`, conformément au reste de la collection FoamPilot.

L’audit préalable de l’API n’a révélé aucune méthode manquante. Les transformations de maillage et de surface sont exécutées avec les utilitaires existants ; aucune extension FoamPilot n’a été ajoutée.

## Workflow exécuté

```text
ideasUnvToFoam cad/backgroundMesh.unv
transformPoints scale=(10000 10000 10000)
surfaceTransformPoints scale=(10000 10000 10000) regionSTL.orig.stl regionSTL.stl
surfaceTransformPoints scale=(10000 10000 10000) specialGapRefinement.orig.stl specialGapRefinement.stl
snappyHexMesh -overwrite
transformPoints scale=(0.0001 0.0001 0.0001)
foamRun
```

La durée source (`endTime 500`) est réduite localement à `0.001` pour obtenir un smoke run borné. Cette réduction valide la mise en données, le maillage, le chargement du solveur et l’intégration temporelle initiale ; elle ne prétend pas reproduire les dix minutes de calcul stationnaire de production mentionnées par Tobias.

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Conversion UNV | 9 212 points, 7 774 cellules et 2 730 faces de frontière lus ; fin normale. |
| Raffinement du gap | Les phases de raffinement de surface, shell et coarse-cell ont été exécutées. |
| Snapping | Maillage snappé de 437 872 cellules, 1 342 123 faces et 481 229 points. |
| Échelle finale | Le maillage a été reconverti à l’échelle originale après `snappyHexMesh`. |
| Calcul | `foamRun` a démarré et s’est terminé normalement avec `End`. |

Le cas est **validé** selon le protocole du projet : la mise en données est recréée par FoamPilot, le maillage de gap est produit et le calcul incompressible court se termine avec OpenFOAM 13.

## Limites

La validation ne remplace pas la campagne stationnaire longue du tutoriel. Une étude de convergence et une vérification détaillée de la résolution du gap restent nécessaires pour une utilisation de production.

## Référence

[1]: https://holzmann-cfd.de/community/training-cases/thin-gap-meshing — Tobias Holzmann, *Thin Gap Meshing*.
