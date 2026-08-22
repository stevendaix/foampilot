# Combustion Chamber — cold-flow case

## Objet

Ce tutoriel de Tobias Holzmann porte sur le maillage et l’analyse à froid d’une chambre de combustion complexe. Le cas source utilise une géométrie détaillée, des couches de paroi et un calcul instationnaire turbulent ; la campagne complète est annoncée comme très longue et vise notamment l’observation de structures tourbillonnaires [1]. Le portage présent valide la mise en données, le maillage et un smoke run court, sans prétendre reproduire la campagne vidéo longue.

## Portage FoamPilot

`run.py` écrit les dictionnaires et les champs via `OpenFOAMDictAddFile.write_raw`, copie le maillage UNV de fond, la surface de chambre et l’eMesh, puis exécute `ideasUnvToFoam`, `snappyHexMesh -overwrite` et `foamRun` par `Solver.run_command`. La durée source `endTime 20` est réduite localement à `0.001` pour rendre l’exécution bornée et reproductible.

L’audit de l’API FoamPilot n’a identifié aucune méthode manquante pour ce workflow. Les adaptations sont locales au cas et aucune nouvelle méthode partagée n’a été ajoutée.

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Conversion UNV | 190 281 points, 180 000 cellules et 20 200 faces de frontière lus avec succès. |
| Maillage snappy | Maillage final de 501 867 cellules, 1 534 364 faces et 537 706 points après ajout des couches. |
| Qualité | `snappyHexMesh` termine avec `Finished meshing without any errors`. |
| Solveur | Le solveur incompressible instationnaire est chargé et progresse jusqu’à la durée bornée configurée. |
| Fin du calcul | `foamRun` termine normalement avec `End`; le dernier temps journalisé est environ `0.00109981 s`. |

Le cas est **validé** selon le protocole du projet : FoamPilot recrée la mise en données, le maillage complexe est produit sans erreur et le calcul OpenFOAM 13 court atteint sa fin normale.

## Limites

Le calcul source complet est beaucoup plus long. La présente validation démontre l’exécution du pipeline, mais ne fournit ni étude de convergence, ni reproduction des résultats physiques de la campagne longue, ni validation expérimentale de la chambre.

## Référence

[1]: https://holzmann-cfd.de/community/training-cases/combustion-chamber — Tobias Holzmann, *Combustion Chamber*.
