# Audit OF13 — multicomponentFluid/filter

L’Allrun OpenFOAM 13 exécute `blockMesh`, `createZones`, `createBaffles`, puis `foamRun`. Le cas est mono-région `multicomponentFluid`, avec une zone `filter` définie dans `system/createZonesDict` et des champs possédant les conditions d’entrée uniformes du filtre (`U=(0 50 0)`, `T=450 K` pour les espèces concernées). Le contrôle impose `endTime=5`, `deltaT=0.001`, `writeInterval=0.1` et un `maxDeltaT=1`.

Le runner `198_multicomponentFluid_filter/run.py` importe par FoamPilot les champs, constantes et dictionnaires de référence, puis reproduit exactement les quatre étapes de l’Allrun avec `BaseSolver.run_command` sous l’environnement OpenFOAM 13 explicite. La zone et la topologie de baffle sont créées par les utilitaires OF13 officiels; aucun fichier ou maillage n’est généré hors FoamPilot.

La validation est complète. `blockMesh`, `createZones` et `createBaffles` terminent correctement, puis `foamRun` atteint `Time=5 s` et `End` en environ 12 secondes. Le Courant maximal final est d’environ `0.995`, les erreurs de continuité restent de l’ordre de `10^-6` à `10^-9` et aucun `FOAM FATAL`, problème de baffle ou erreur de zone n’est observé.

Statut : **validé OF13 — `End=5 s`, zone/filter et baffles créés, calcul réussi**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037.
