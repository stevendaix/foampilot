# Audit OF13 — multicomponentFluid/membrane

L’Allrun OpenFOAM 13 exécute `blockMesh`, `snappyHexMesh -overwrite`, `createBaffles`, `setFields`, puis `foamRun`. Le cas utilise la surface STL `constant/triSurface/membrane.stl`, une faceZone `membrane` dans `snappyHexMeshDict`, une cellule interne `pipe`, puis `createBafflesDict` transforme cette faceZone en deux patches mappés `membranePipe` et `membraneSleeve` avec `internalFacesOnly=true`. Le contrôle impose `endTime=10`, `deltaT=1e-3`, `writeInterval=1`, ajustement de pas et `maxCo=1`; la bibliothèque `libspecieTransfer.so` est conservée dans `controlDict`.

Le runner `200_multicomponentFluid_membrane/run.py` importe par FoamPilot les champs, constantes, dictionnaires système et géométrie STL de référence, puis reproduit exactement la préparation avec `blockMesh`, `snappyHexMesh -overwrite`, `createBaffles`, `setFields` et `foamRun` sous l’environnement OF13 explicite. Les conditions multi-espèces, les propriétés physiques et les fonctions de flux de membrane restent celles du tutoriel.

La validation est complète. `snappyHexMesh` termine sans erreur avec 18 632 cellules, 960 baffles reconnus et les patches de membrane créés. `createBaffles` et `setFields` terminent correctement. `foamRun` atteint `Time=10 s` et `End` en environ 14 secondes; le Courant maximal final est d’environ `0.966`, les erreurs de continuité restent faibles et aucun `FOAM FATAL`, problème de STL, baffle ou zone n’est observé.

Statut : **validé OF13 — `End=10 s`, membrane STL/baffles et calcul réussis**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037.
