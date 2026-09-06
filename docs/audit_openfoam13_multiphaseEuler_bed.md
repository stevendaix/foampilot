# Audit OF13 — multiphaseEuler/bed

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est un lit triphasique `air/water/solid` avec `solid` stationnaire, modèles de diamètre et de transferts de quantité de mouvement, contraintes de pression et correction de traînée. Le dictionnaire `setFieldsDict` initialise le domaine en air puis la zone `bed` avec `alpha.air=0.4`, `alpha.water=0` et `alpha.solid=0.6`. Le contrôle impose `endTime=20`, `deltaT=0.002`, `writeInterval=0.1`, ajustement de pas et `maxDeltaT=1`.

Le runner `212_multiphaseEuler_bed/run.py` importe par FoamPilot les champs suffixés de phase (`.air`, `.water`, `.solid`), tous les dictionnaires `constant/system`, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. Les propriétés thermophysiques, les modèles de phase et les conditions initiales du lit sont conservés sans réécriture.

La validation est complète. `blockMesh` et `setFields` terminent correctement; le journal confirme l’initialisation de `alpha.air`, `alpha.water` et `alpha.solid` dans la zone `bed`. `foamRun` atteint `Time=20 s` et `End` en environ 196 secondes. Les fractions `air` et `water` restent bornées, les températures des phases restent physiques, les solveurs d’énergie convergent et aucun `FOAM FATAL`, défaut de lit ou erreur de phase n’est observé.

Statut : **validé OF13 — lit triphasique air/eau/solide jusqu’à `End=20 s`**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037; aucun changement d’API supplémentaire n’a été nécessaire.
