# Audit OF13 — multiphaseEuler/bubbleColumnLaminar

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est une colonne air/eau avec modèle laminaire pour les deux phases, sans turbulence LES ni modèle IATE. Le domaine est initialisé en air puis la zone `water` en eau. Les propriétés de phase, drag, tension de surface, schémas de transport, contraintes de pression et fonction de moyenne sont importés depuis la référence OF13. Le contrôle impose `endTime=100`, `deltaT=0.005`, `writeInterval=1` et `maxDeltaT=1`; `fvSolution` conserve les deux sous-cycles de correction des fractions.

Le runner `220_multiphaseEuler_bubbleColumnLaminar/run.py` importe par FoamPilot les champs `.air/.water`, les dictionnaires `constant/system`, les propriétés de phases et les fonctions, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. La chaîne reste sérielle comme la référence.

La validation du maillage et de l’initialisation réussit. Le calcul laminaire résout les phases avec corrections MULES et conserve les fractions bornées; les températures restent physiques autour de 300–350 K. Le Courant maximal observé reste inférieur à environ `0.74`. Le journal contient `End` après une progression observée vers `Time≈84.38 s`; aucun `FOAM FATAL`, problème de phase, de turbulence ou d’instabilité n’est observé. Le plafond global a été consommé par le coût des sous-cycles et du calcul multiphasique, mais la fin du calcul est présente dans le journal.

Statut : **validé OF13 — colonne air/eau laminaire jusqu’à `End`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
