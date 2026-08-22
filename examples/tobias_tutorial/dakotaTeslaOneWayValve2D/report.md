# Dakota Tesla's One-Way-Valve

## Objet

Ce tutoriel de Tobias Holzmann montre le couplage de DAKOTA avec OpenFOAM pour optimiser une valve de Tesla 2D. DAKOTA modifie les paramètres de pression, regénère le maillage, exécute les calculs (dans les deux directions) et évalue la fonction objectif (le ratio des flux) [1].

## Portage FoamPilot

`run.py` écrit les dictionnaires, copie le maillage de fond UNV et les surfaces STL. Le script `dakota.sh` a été modifié pour que DAKOTA utilise un script Python (`solve.py`) qui appelle `foamRun` via `Solver.run_command` de FoamPilot. 

Pour OpenFOAM 13, les coefficients `nLayers 1` et `expansionRatio 1` ont été ajoutés à `system/extrudeMeshDict`. Le nombre d'itérations initiales de DAKOTA a été réduit, et la durée du calcul a été limitée à `endTime 5` pour permettre un smoke run.

L’audit de l’API n’a identifié aucune méthode FoamPilot manquante. 

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Conversion UNV | Le maillage de fond est converti avec succès. |
| Maillage | `snappyHexMesh` et `extrudeMesh` s'exécutent sans erreur. |
| Couplage DAKOTA | DAKOTA lance avec succès la boucle d'optimisation via le script `dakota.sh`. |
| Exécution FoamPilot | `dakota.sh` appelle `solve.py`, qui utilise `Solver.run_command` pour exécuter `foamRun`. |
| Boucle d'optimisation | Plusieurs itérations (calcul flux direct et inversé, calcul du ratio) ont été complétées avec succès. |

Le cas est **validé**. La validation démontre que le couplage DAKOTA fonctionne et utilise l'API FoamPilot pour exécuter les solveurs OpenFOAM 13.

## Limites

Le nombre d'échantillons et le temps de simulation ont été réduits. Il s'agit d'une démonstration du workflow d'optimisation et non d'une optimisation complète de la valve.

## Référence

[1]: https://holzmann-cfd.com/community/training-cases/teslas-one-way-valve — Tobias Holzmann, *Teslas One Way Valve*.
