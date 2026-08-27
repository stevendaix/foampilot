# Audit OF13 — multicomponentFluid/DLR_A_LTS

L’Allrun OpenFOAM 13 exécute `chemkinToFoam chemkin/grimech30.dat chemkin/thermo30.dat chemkin/transportProperties constant/reactionsGRI constant/thermo.compressibleGasGRI`, puis `blockMesh`, `setFields`, `decomposePar -force`, `runParallel foamRun` et `reconstructPar`. Le cas est mono-région `multicomponentFluid`, utilise les schémas LTS `localEuler`, les 36 espèces de la chimie GRI et une configuration `endTime=10000`, `deltaT=1`, `writeInterval=1000`, avec les limites LTS de `fvSolution` (`maxDeltaT=1e-4`, `maxCo=0.25`).

Le runner `190_multicomponentFluid_DLR_A_LTS/run.py` importe via FoamPilot les champs `0/`, les dictionnaires `system/`, les propriétés `constant/` et les fichiers Chemkin. Il reproduit ensuite l’intégralité de l’Allrun avec `BaseSolver.run_command`, y compris la conversion Chemkin et les phases de décomposition, calcul MPI à six processus et reconstruction. L’environnement OF13 est chargé explicitement dans les processus enfants pour sélectionner les bibliothèques OpenMPI et Scotch ThirdParty.

La première tentative a révélé que `decomposeParDict` impose six domaines; le runner a été corrigé de quatre à six processus. Après correction, Chemkin, `blockMesh`, `setFields` et `decomposePar -force` terminent correctement. Le calcul parallèle atteint `Time=10000 s` et `End` en environ 250 secondes. Les temps écrits `1000` à `10000` sont tous reconstruits avec succès. Les erreurs de continuité restent de l’ordre de `10^-7` à `10^-12` dans les derniers pas et aucun `FOAM FATAL`, échec MPI ou erreur de reconstruction n’est observé.

Statut : **validé OF13 — `End=10000 s`, calcul LTS parallèle et reconstruction réussis**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037. Le nombre de processus à six est une adaptation de configuration du runner, rendue nécessaire par la valeur officielle de `system/decomposeParDict`.
