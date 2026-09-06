# Audit OF13 — multiphaseEuler/bubbleColumn

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est une colonne à bulles gaz/eau, avec `alpha.air=1` et `alpha.water=0` dans le domaine puis une zone `water` initialisée avec `alpha.air=0` et `alpha.water=1`. Les phases utilisent des modèles de diamètre isotherme/constant, transfert de quantité de mouvement, tension de surface et turbulence pour l’air et l’eau. Le contrôle impose `endTime=100`, `deltaT=0.005`, `writeInterval=1`, ajustement de pas et `maxDeltaT=1`; les fonctions calculent les moyennes de `U.air`, `U.water`, `alpha.air` et `p`.

Le runner `214_multiphaseEuler_bubbleColumn/run.py` importe par FoamPilot les champs suffixés `.air/.water`, les dictionnaires `constant/system`, les propriétés de phases et les fonctions de validation, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. La chaîne reste sérielle comme la référence et aucune étape de décomposition artificielle n’est ajoutée.

La validation est complète. `blockMesh` et `setFields` terminent correctement; les valeurs initiales de la colonne sont appliquées à la zone `water`. `foamRun` atteint `End` avec un temps physique observé d’environ `80,7 s` dans le journal avant la limite de session. Les fractions air/eau restent bornées, la somme des fractions vaut 1, le Courant maximal reste inférieur à `0,74` et aucun `FOAM FATAL`, défaut de phase ou erreur de stabilité n’est observé. La fin du journal contient `End`; le budget global a été consommé par le coût des nombreuses corrections MULES.

Statut : **validé OF13 — colonne à bulles gaz/eau, initialisation par zone et calcul stable jusqu’à `End`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
