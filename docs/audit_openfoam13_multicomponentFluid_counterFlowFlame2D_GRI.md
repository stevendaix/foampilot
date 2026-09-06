# Audit OF13 — multicomponentFluid/counterFlowFlame2D_GRI

La référence OpenFOAM 13 fournit `Allrun-parallel` : `blockMesh`, `decomposePar -cellProc`, `foamRun -parallel` puis `reconstructPar -cellProc`. Le cas est counter-flow 2D `multicomponentFluid`, avec chimie GRI détaillée, `endTime=0.5`, `deltaT=1e-6`, `writeInterval=0.05`, ajustement de pas et `maxCo=0.4`. La décomposition est hiérarchique à 12 domaines (`6×2×1`) avec distribution Zoltan RCB.

Le runner `196_multicomponentFluid_counterFlowFlame2D_GRI/run.py` importe par FoamPilot les champs, constantes et dictionnaires de la référence, puis reproduit la chaîne parallèle avec `blockMesh`, `decomposePar -cellProc`, `foamRun -parallel` à 12 processus et `reconstructPar -cellProc`. La chimie GRI, la thermodynamique, le maillage et les conditions aux limites ne sont pas réécrits.

`blockMesh` et la décomposition terminent correctement. Le calcul parallèle reste stable, avec un Courant maximal proche de `0.4` et des erreurs de continuité faibles, mais le plafond de 300 secondes intervient vers `Time≈0.151 s` sur `0.5 s`. La reconstruction finale n’est pas atteinte dans ce budget; aucun `FOAM FATAL`, problème de décomposition ou erreur de bibliothèque n’est observé.

Statut : **accepté avec réserve — coût de la chimie GRI parallèle et limite de temps**.

Le runner utilise `BaseSolver.run_command(environment=...)`, évolution générique API-037.
