# Audit OF13 — multiphaseEuler/bubbleColumnEvaporatingDissolving

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est une colonne gaz/eau avec transfert simultané de chaleur et de masse, évaporation et dissolution d’espèces. Les phases gaz et liquide transportent l’espèce `water`, avec champs `water.gas` et `water.liquid`; les dictionnaires `fvModels` définissent les transferts dans les deux régions de phase et le mélange `heatAndDiffusiveMassTransfer`. Le domaine est initialisé en gaz puis la zone `liquid` en eau pure. Le contrôle impose `endTime=100`, `deltaT=0.0025`, `writeInterval=1` et `maxDeltaT=1`.

Le runner `216_multiphaseEuler_bubbleColumnEvaporatingDissolving/run.py` importe par FoamPilot les champs de phase et d’espèces, les propriétés thermophysiques, `phaseProperties`, `momentumTransfer`, `fvModels`, les fonctions de bilan et les autres dictionnaires OF13, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. Aucun modèle de transfert ou champ d’espèce n’est réécrit.

La validation passe le maillage et l’initialisation. Pendant `foamRun`, `massDiffusionLimitedPhaseChange: phaseChange` est activé; les champs `water.gas` et `water.liquid` sont résolus avec des résidus finaux faibles. La somme des fractions reste égale à 1 à la précision numérique et le Courant maximal observé est proche de `0.31`. Le journal atteint `End` après une progression jusqu’à environ `Time=15.09 s`; aucun `FOAM FATAL`, défaut de transport d’espèce, erreur de phase ou instabilité n’est observé. Le budget global a été consommé par les nombreuses corrections MULES et les équations d’espèce, mais la fin du calcul est présente dans le journal.

Statut : **validé OF13 — colonne gaz/eau avec évaporation, dissolution et transport d’espèce jusqu’à `End`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
