# Audit OF13 — multicomponentFluid/smallPoolFire3D

L’Allrun OpenFOAM 13 exécute `blockMesh`, `createPatch`, `decomposePar -force`, `foamRun -parallel`, puis `reconstructPar`. Le dictionnaire `createPatchDict` construit le patch `inlet` depuis une zone box `(-0.1 -0.001 -0.1)–(0.1 0.005 0.1)`. La décomposition est hiérarchique à 4 domaines (`1×2×2`). Le cas est un feu de nappe 3D `multicomponentFluid` avec réaction, combustion, radiation et suie; le contrôle impose `endTime=4`, `deltaT=0.001`, `writeInterval=0.1`, ajustement de pas, `maxCo=0.6` et `maxDeltaT=0.1`.

Le runner `205_multicomponentFluid_smallPoolFire3D/run.py` importe par FoamPilot tous les champs, constantes et dictionnaires de référence, puis reproduit la chaîne parallèle avec `blockMesh`, `createPatch`, `decomposePar -force`, `foamRun -parallel` à 4 domaines et `reconstructPar`. Les modèles de combustion/radiation, les conditions aux limites et la topologie 3D sont conservés sans réécriture.

`blockMesh`, `createPatch` et `decomposePar -force` terminent correctement. Le calcul MPI reste stable jusqu’à `Time≈2.726 s` sur `4 s` au plafond de 300 secondes, avec Courant maximal proche de `0.6` et erreurs de continuité faibles. Aucun `FOAM FATAL`, problème MPI, erreur de patch ou divergence n’est observé; la reconstruction finale n’est pas atteinte dans le budget.

Statut : **accepté avec réserve — calcul 3D stable, limite de temps avant `End=4 s` et reconstruction**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037.
