# Audit OF13 — multicomponentFluid/smallPoolFire2D

L’Allrun OpenFOAM 13 exécute `blockMesh`, `createPatch`, puis `foamRun`. Le dictionnaire `createPatchDict` construit le patch `inlet` depuis une zone box `(-0.0529 -0.001 -0.1)–(0.0529 0.002 0.1)`. Le cas est un feu de nappe 2D `multicomponentFluid` avec réaction, combustion, radiation, soot et champs multi-espèces. Le contrôle impose `endTime=3`, `deltaT=0.001`, `writeInterval=0.1`, ajustement de pas et `maxCo=0.5`.

Le runner `204_multicomponentFluid_smallPoolFire2D/run.py` importe par FoamPilot tous les champs, dictionnaires de système et propriétés de référence, puis reproduit exactement `blockMesh`, `createPatch` et `foamRun` sous l’environnement OF13 explicite. Les modèles de combustion, radiation, réaction et conditions aux limites sont conservés sans réécriture.

`blockMesh` et `createPatch` terminent correctement. Le calcul reste stable, avec un Courant maximal proche de `0.5` et des erreurs de continuité faibles, mais le plafond de 300 secondes intervient vers `Time≈2.684 s` sur `3 s`. Aucun `FOAM FATAL`, problème de patch, erreur de radiation ou divergence n’est observé; la terminaison finale n’est pas atteinte dans le budget.

Statut : **accepté avec réserve — progression stable, limite de temps avant `End=3 s`**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037.
