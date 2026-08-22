# Feature Edge Refinement

## Objet

Ce tutoriel de Tobias Holzmann étudie le raffinement explicite des arêtes utilisé par `snappyHexMesh`. Il compare trois configurations : le maillage d’arêtes standard, une version optimisée dans Blender, puis le maillage standard avec la syntaxe `levels ((distance level))`. L’objectif est de montrer que le résultat du raffinement dépend à la fois de la géométrie d’arêtes fournie et de la structure du dictionnaire de raffinement [1].

## Portage FoamPilot

Le dossier contient un `run.py`, des templates déclaratifs et les actifs `backgroundMesh.unv`, `cylinder.stl`, `featureEdgesStandard.obj` et `featureEdgesOptimized.obj`. Le runner écrit chaque cas par `OpenFOAMDictAddFile.write_raw`, convertit l’OBJ en `featureEdges.eMesh` avec `surfaceFeatureConvert`, convertit le maillage de fond avec `ideasUnvToFoam`, puis exécute `snappyHexMesh -overwrite`. Après chaque variante, le maillage final est copié dans un répertoire de résultat dédié afin de conserver les trois sorties comparables.

L’audit de FoamPilot n’a pas identifié de méthode manquante. Les utilitaires nécessaires sont lancés par l’API existante `Solver.run_command`; aucune extension du cœur FoamPilot n’a donc été ajoutée.

## Variantes exécutées

| Variante | Entrée d’arêtes | Adaptation |
| --- | --- | --- |
| `standard_edge_mesh` | `featureEdgesStandard.obj` | Dictionnaire source avec `level`. |
| `optimized_edge_mesh` | `featureEdgesOptimized.obj` | Même dictionnaire, géométrie d’arêtes optimisée. |
| `standard_edge_mesh_levels` | `featureEdgesStandard.obj` | Remplacement local de `level 2` par `levels ((0.01 2))`. |

Chaque variante a été reconstruite depuis un répertoire de cas propre. Cette précaution évite qu’un `polyMesh` ou un `featureEdges.eMesh` produit par une variante précédente ne contamine la suivante.

## Preuves OpenFOAM 13

Les trois séquences ont terminé normalement avec `End`. Pour la troisième variante, les journaux indiquent notamment la lecture de 169 781 points et 160 000 cellules de fond, la lecture de 19 200 faces de frontière, l’activation du raffinement explicite des features, 640 arêtes de feature détectées et un maillage final de 85 067 cellules, 272 135 faces et 103 401 points. Les journaux des deux autres variantes sont conservés sous `case/log.*` après exécution ; leurs maillages sont archivés dans `standard_edge_mesh/polyMesh`, `optimized_edge_mesh/polyMesh` et `standard_edge_mesh_levels/polyMesh`.

Le cas est **validé comme workflow de maillage** : FoamPilot recrée les fichiers, les trois conversions d’arêtes réussissent, les trois maillages `snappyHexMesh` sont écrits et OpenFOAM 13 termine chaque étape sans erreur. Il s’agit d’un tutoriel de maillage ; aucun calcul solver n’est requis par le workflow source.

## Limites

La validation confirme l’exécution et la production des maillages, mais ne constitue pas une comparaison visuelle automatisée des niveaux de raffinement. Une inspection dans ParaView peut être utilisée pour comparer les trois répertoires de sortie, sans être nécessaire au critère d’exécution du projet.

## Référence

[1]: https://holzmann-cfd.de/community/training-cases/feature-edge-refinement — Tobias Holzmann, *Feature Edge Refinement*.
