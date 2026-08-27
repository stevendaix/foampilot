# Audit OF13 — multicomponentFluid/counterFlowFlame2D

Le tutoriel OF13 ne fournit pas d’Allrun. Sa mise en données de référence est un cas 2D `multicomponentFluid` avec réaction méthane simplifiée, maillage `100×40×1`, patches `fuel`, `air`, `outlet` et `frontAndBack` de type `empty`. Le contrôle impose `endTime=0.5`, `deltaT=1e-6`, `writeInterval=0.05`, un pas ajustable et `maxCo=0.4`.

Le runner `193_multicomponentFluid_counterFlowFlame2D/run.py` importe via FoamPilot les champs, constantes et dictionnaires OF13, puis exécute la séquence conventionnelle `blockMesh` et `foamRun` sous OpenFOAM 13. Les réactions, la thermodynamique compressible, le transport et les conditions aux limites sont conservés depuis la référence sans réécriture manuelle.

La validation est complète. Le maillage et le solveur démarrent correctement; le calcul atteint `Time=0.5 s` et `End` en environ 27 secondes. Le Courant maximal observé est d’environ `0.398`, inférieur à la consigne `0.4`, les erreurs de continuité restent faibles et aucun `FOAM FATAL`, problème de chimie ou erreur de champ n’est observé.

Statut : **validé OF13 — `End=0.5 s`, calcul counter-flow réussi**.

L’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037 est utilisée pour rendre explicite l’environnement OpenFOAM 13 pendant l’exécution.
