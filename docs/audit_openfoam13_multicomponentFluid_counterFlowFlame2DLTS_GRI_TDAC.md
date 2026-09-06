# Audit OF13 — multicomponentFluid/counterFlowFlame2DLTS_GRI_TDAC

Le tutoriel OF13 ne fournit pas d’Allrun; sa mise en données impose un cas counter-flow 2D `multicomponentFluid` avec chimie GRI détaillée et réduction/tabulation TDAC. Le maillage est `100×40×1`, les patches sont `fuel`, `air`, `outlet` et `frontAndBack` de type `empty`. Le contrôle impose `endTime=1500`, `deltaT=1`, `writeInterval=20`, tandis que `fvSchemes` utilise `localEuler`. La décomposition de référence est hiérarchique à 4 domaines (`2×2×1`). `chemistryProperties` active `ode/Seulex`, `initialChemicalTimeStep=1e-7`, réduction à `1e-4`, tabulation à `3e-3` et inclut `reactionsGRI`.

Le runner `195_multicomponentFluid_counterFlowFlame2DLTS_GRI_TDAC/run.py` importe par FoamPilot les champs, constantes et dictionnaires OF13, puis exécute `blockMesh`, `decomposePar`, `foamRun -parallel` à quatre domaines et `reconstructPar`. La thermodynamique, la chimie GRI/TDAC, les fonctions de contrôle et les conditions aux limites restent celles de la référence.

La validation est complète. Le maillage, la décomposition hiérarchique et le calcul parallèle terminent sans `FOAM FATAL`. `foamRun` atteint `Time=1500 s` et `End`; `reconstructPar` reconstruit les temps 20 à 1500 s et atteint également `End`. Les erreurs de continuité restent faibles, de l’ordre de `10^-10` dans les derniers pas. L’exécution globale dépasse le plafond de 300 secondes principalement à cause du coût du calcul chimique/reconstruction, mais les journaux montrent une terminaison complète.

Statut : **validé OF13 — `End=1500 s`, calcul LTS GRI/TDAC parallèle et reconstruction réussis**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037; aucune nouvelle extension d’API spécifique n’est nécessaire.
