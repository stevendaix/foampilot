# Battery Cooling — Tesla 4680 air heat exchanger

## Objet

Ce tutoriel de Tobias Holzmann simule le refroidissement par air d’un pack de cellules Tesla 4680. Les cellules dissipent une puissance thermique et sont refroidies par un écoulement traversant un échangeur. Le cas source est thermo-fluidique, utilise une géométrie détaillée et prévoit une exécution parallèle longue [1].

## Portage FoamPilot

`run.py` écrit les dictionnaires et les champs par `OpenFOAMDictAddFile.write_raw`, copie le maillage UNV et les STL officiels, puis exécute `ideasUnvToFoam`, `snappyHexMesh -overwrite` et `foamRun` via `Solver.run_command`. Le calcul source est parallélisé sur huit processus ; le portage utilise une exécution séquentielle afin de rester reproductible dans l’exemple FoamPilot.

Les niveaux de raffinement et le nombre de couches ont été réduits localement pour le smoke run : le niveau de la surface batterie passe de `(2 2)` à `(1 1)`, `nSurfaceLayers` de 2 à 1 et `nSolveIter` de 200 à 50. La géométrie, les conditions thermiques, les propriétés physiques et le mécanisme de couches sont conservés. `endTime` est réduit à `0.001`. Ces réductions sont documentées et ne constituent pas une prétention à reproduire la résolution de production du tutoriel.

L’audit de l’API n’a identifié aucune méthode FoamPilot manquante. Aucune extension du cœur n’a été ajoutée.

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Conversion UNV | 261 800 points, 248 193 cellules et 26 790 faces de frontière lus avec succès. |
| Maillage | 557 520 cellules après snapping ; 681 111 cellules après ajout des couches. |
| Qualité | `Finished meshing without any errors` apparaît dans le journal `snappyHexMesh`. |
| Solveur | Le solveur thermo-fluidique `fluid` est sélectionné et la contrainte `limitTemperature` est chargée. |
| Calcul | `foamRun` atteint la fin du smoke run et se termine avec `End`. |

Le cas est **validé comme smoke run thermo-fluidique**. La validation démontre que FoamPilot recrée le cas et que le maillage, les couches, les propriétés thermiques et le solveur OpenFOAM 13 s’exécutent correctement avec la résolution bornée documentée.

## Limites

La résolution est volontairement réduite par rapport au cas source et l’exécution n’est pas la campagne parallèle longue prévue pour produire les résultats finaux. Une étude de convergence spatiale, thermique et temporelle reste nécessaire avant toute interprétation de performance du pack batterie.

## Référence

[1]: https://holzmann-cfd.com/community/training-cases/battery-cooling — Tobias Holzmann, *Battery Cooling*.
