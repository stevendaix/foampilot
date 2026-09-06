# Audit OF13 — multiphaseEuler/bubbleColumnIATE

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est une colonne air/eau avec un modèle IATE de population de bulles. Le champ interfacial `kappai.air` est transporté dans la phase air; les coefficients de `phaseProperties` activent notamment les modèles de coalescence par sillage et aléatoire ainsi que la cassure turbulente. Les fonctions de moyenne de champ `U.air`, `U.water`, `alpha.air` et `p` sont conservées. Le domaine est initialisé en air puis la zone `water` en eau. Le contrôle impose `endTime=100`, `deltaT=0.005`, `writeInterval=1` et `maxDeltaT=1`.

Le runner `218_multiphaseEuler_bubbleColumnIATE/run.py` importe par FoamPilot les champs de phase, `kappai.air`, les propriétés de phase, les coefficients IATE, les modèles de transfert de quantité de mouvement, les fonctions et les autres dictionnaires OF13, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. La chaîne reste sérielle comme la référence.

La validation du maillage et de l’initialisation réussit. Pendant `foamRun`, `kappai.air` est résolu à chaque pas et les modèles IATE de coalescence/cassure sont actifs. Les fractions air/eau restent bornées et leur somme vaut 1; les températures restent entre environ 300 et 350 K; le Courant maximal observé reste inférieur à environ `0.95`. Aucun `FOAM FATAL`, défaut de population balance ou instabilité n’est observé. Le plafond de 300 secondes interrompt la validation autour de `Time≈75.75 s` sur `100 s`; aucune reconstruction n’est requise par l’Allrun.

Statut : **accepté avec réserve — mise en données IATE et calcul stables sans erreur, mais `End=100 s` hors budget de validation**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
