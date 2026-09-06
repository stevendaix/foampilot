# Audit OF13 — multiphaseEuler/LBend

Le tutoriel OpenFOAM 13 ne fournit pas d’Allrun. Sa mise en données comprend un `blockMeshDict` qui décrit directement la géométrie complète du coude en L, sans étape d’extrusion. Le cas `multiphaseEuler` comporte les phases `solids` et `gas`, des propriétés thermophysiques séparées, un modèle de théorie cinétique granulaire pour les solides, des modèles de traînée et de transfert de quantité de mouvement, une contrainte de pression et les corrections de traînée du solveur. Le contrôle impose `endTime=1.9`, `deltaT=1e-4`, `writeInterval=0.1`, ajustement de pas, `maxCo=0.1` et `maxDeltaT=0.01`.

Le runner `210_multiphaseEuler_LBend/run.py` importe par FoamPilot les champs suffixés de phase (`.solids`, `.gas`), tous les dictionnaires `constant` et `system`, puis reproduit `blockMesh` et `foamRun` sous environnement OF13 explicite. La référence ne comporte pas d’Allrun et aucune étape d’extrusion ou de décomposition n’est ajoutée artificiellement.

`blockMesh` termine correctement. `foamRun` reste stable au plafond de 300 secondes, avec MULES appliqué à `alpha.solids` et `alpha.gas`, fractions bornées, somme des fractions égale à 1, températures proches de 300 K et solveurs de phase convergents. Le dernier état observé est proche de `Time≈0,00` dans la sortie tronquée de validation, sans `FOAM FATAL`, mais le calcul n’atteint pas `End=1,9 s` dans le budget.

Statut : **accepté avec réserve — mise en données et stabilité multiphase validées, limite de temps avant `End=1,9 s`**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037; aucun changement d’API supplémentaire n’a été nécessaire.
