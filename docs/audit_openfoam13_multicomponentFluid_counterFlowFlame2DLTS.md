# Audit OF13 — multicomponentFluid/counterFlowFlame2DLTS

Le tutoriel OF13 est un cas mono-région `multicomponentFluid` counter-flow avec la même mise en données 2D que `counterFlowFlame2D`, mais avec LTS. Le maillage est `100×40×1`, les patches sont `fuel`, `air`, `outlet` et `frontAndBack` de type `empty`. Le contrôle impose `endTime=1000`, `deltaT=1`, `writeInterval=20`, tandis que `fvSchemes` utilise `localEuler` et `fvSolution` fournit les paramètres LTS associés.

Le runner `194_multicomponentFluid_counterFlowFlame2DLTS/run.py` importe intégralement les champs, constantes et dictionnaires de référence par FoamPilot, puis exécute `blockMesh` et `foamRun` sous l’environnement OpenFOAM 13 explicite. Les réactions méthane, la thermodynamique compressible, les conditions aux limites et les paramètres LTS restent ceux de la référence.

La validation est complète. Le maillage et le solveur démarrent correctement; le calcul LTS atteint `Time=1000 s` et `End` en environ 19 secondes. Les erreurs de continuité restent de l’ordre de `10^-8` à `10^-10` dans les derniers pas et aucun `FOAM FATAL`, problème de chimie ou erreur de champ n’est observé.

Statut : **validé OF13 — `End=1000 s`, calcul counter-flow LTS réussi**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037.
