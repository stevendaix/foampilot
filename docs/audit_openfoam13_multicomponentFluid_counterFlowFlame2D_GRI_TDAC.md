# Audit OF13 — multicomponentFluid/counterFlowFlame2D_GRI_TDAC

Le tutoriel OF13 ne fournit pas d’Allrun de calcul; sa mise en données impose un cas counter-flow 2D `multicomponentFluid` avec chimie GRI sous TDAC, `endTime=0.5`, `deltaT=1e-6`, `writeInterval=0.05`, ajustement de pas et `maxCo=0.4`. Le maillage est `100×40×1`, avec les patches `fuel`, `air`, `outlet` et `frontAndBack` de type `empty`. Le dictionnaire `decomposeParDict` impose une décomposition hiérarchique à 4 domaines (`2×2×1`). La chimie active `ode/Seulex`, une étape chimique initiale de `1e-7`, une réduction de tolérance `1e-4` et une tabulation de tolérance `3e-3`.

Le runner `197_multicomponentFluid_counterFlowFlame2D_GRI_TDAC/run.py` importe les champs, constantes et dictionnaires par FoamPilot, puis reproduit `blockMesh`, `decomposePar`, `foamRun -parallel` à quatre domaines et `reconstructPar`. La chimie GRI/TDAC, la thermodynamique et les conditions aux limites sont conservées depuis la référence.

La validation est complète. La décomposition et le calcul parallèle terminent sans erreur fatale; `foamRun` atteint `Time=0.5 s` et `End`. La reconstruction des temps `0.05` à `0.5 s` atteint également `End`. Le Courant maximal reste proche de `0.4`, les erreurs de continuité sont faibles et aucun `FOAM FATAL`, problème MPI ou erreur de reconstruction n’est observé. Le plafond de session est dépassé principalement à cause du calcul et de la reconstruction GRI/TDAC, mais les journaux montrent une terminaison complète.

Statut : **validé OF13 — `End=0,5 s`, calcul parallèle GRI/TDAC et reconstruction réussis**.

Le runner utilise `BaseSolver.run_command(environment=...)`, évolution générique API-037.
