# Tobias — Pseudo-2D Adaptive Mesh Refinement

## Objectif

Ce cas reproduit le tutoriel de raffinement adaptatif pseudo-2D de Tobias Holzmann. Le champ scalaire passif `S` sert de critère de raffinement. Le raffinement dynamique OpenFOAM étant tridimensionnel, la géométrie et les conditions aux limites sont construites comme un cas pseudo-2D.

## Implémentation FoamPilot

`run.py` génère le cas par FoamPilot, écrit les dictionnaires et champs à partir des templates, place le maillage UNV de fond et le STL du cylindre, puis exécute `ideasUnvToFoam`, `surfaceFeatures`, `snappyHexMesh` et `foamRun` via `Solver.run_command`. Le maillage et le STL proviennent de l’archive complète du tutoriel v12, le dépôt source ne contenant pas tous les actifs référencés.

L’audit de l’API n’a révélé aucune méthode FoamPilot manquante pour ce workflow. La seule correction apportée est locale au runner : réduire `endTime` à `0.001` pour obtenir une validation reproductible et bornée, tout en conservant le mécanisme de raffinement dynamique du cas.

## Workflow exécuté

```text
ideasUnvToFoam cad/backgroundMesh.unv
surfaceFeatures
snappyHexMesh -overwrite
foamRun
```

## Résultats de validation

| Vérification | Résultat |
| --- | --- |
| Conversion UNV | Terminée avec succès. |
| Extraction des caractéristiques | Terminée avec succès. |
| Maillage initial | Généré par `snappyHexMesh` avec contrôle final de qualité réussi. |
| Raffinement dynamique | La boucle AMR et le transport du scalaire `S` ont été chargés par `foamRun`. |
| Calcul OpenFOAM 13 | Le runner s’est terminé normalement et a affiché `Validated AMR smoke run`. |

Le cas est maintenant **validé** selon le protocole du projet : la mise en données est recréée par FoamPilot, les étapes de maillage s’exécutent avec OpenFOAM 13 et le solveur termine le smoke run configuré. Cette validation ne prétend pas remplacer une campagne de calcul longue ni une étude de convergence du raffinement adaptatif.

## Référence

[1]: https://holzmann-cfd.com/community/training-cases/adaptive-mesh-refinement — Tobias Holzmann, *Pseudo-2D Adaptive Mesh Refinement*.
