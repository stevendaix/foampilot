# Audit OF13 — multiphaseEuler/bubbleColumnLES

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est une colonne air/eau avec turbulence LES pour les deux phases et deux sous-cycles de correction des fractions (`nSubCycles=2`). Le domaine est initialisé en air puis la zone `water` en eau. Les propriétés de phase, drag, tension de surface, turbulence LES, schémas de transport et fonction de moyennes sont importés sans modification. Le contrôle impose `endTime=100`, `deltaT=0.005`, `writeInterval=1` et `maxDeltaT=1`.

Le runner `219_multiphaseEuler_bubbleColumnLES/run.py` importe par FoamPilot les champs `.air/.water`, les dictionnaires `constant/system`, les configurations de turbulence et les fonctions de validation, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. La chaîne reste sérielle comme la référence.

La validation du maillage et de l’initialisation réussit. Le calcul reste stable avec corrections MULES, fractions air/eau bornées et somme des fractions égale à 1. Le Courant maximal observé est proche de `1.00` sans défaut fatal. Le plafond de 300 secondes interrompt la progression vers `Time≈79.39 s` sur `100 s`; aucun `FOAM FATAL`, problème LES ou défaut de phase n’est observé. Aucune reconstruction n’est requise par l’Allrun.

Statut : **accepté avec réserve — mise en données LES et calcul stables sans erreur, mais `End=100 s` reste hors budget de validation**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
