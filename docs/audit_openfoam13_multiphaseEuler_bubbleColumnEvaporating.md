# Audit OF13 — multiphaseEuler/bubbleColumnEvaporating

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est une colonne gaz/eau avec transfert diffusif de masse et évaporation liquide-gaz. Le domaine est initialisé par `alpha.gas=1`, `alpha.liquid=0`; la zone `liquid` est initialisée avec `alpha.gas=0.01` et `alpha.liquid=0.99`. Les dictionnaires conservent les modèles de diamètre, drag, tension de surface, thermodynamique gaz/eau et les fonctions de bilan massique personnalisées qui calculent les intégrales de `alphaRhoPhi`, `rho` et les flux aux frontières. Le contrôle impose `endTime=100`, `deltaT=0.0025`, `writeInterval=1` et `maxDeltaT=1`.

Le runner `215_multiphaseEuler_bubbleColumnEvaporating/run.py` importe par FoamPilot les champs `.gas/.liquid`, les propriétés thermophysiques, `phaseProperties`, `momentumTransfer`, `fvModels`, `continuityFunctions` et les autres dictionnaires OF13, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. Les modèles de transfert diffusif et les fonctions de bilan restent ceux de la référence sans réécriture.

La validation passe le maillage et l’initialisation de la colonne. `foamRun` active et rapporte `massDiffusionLimitedPhaseChange: phaseChange`, maintient la somme des fractions à 1 et conserve un Courant maximal observé proche de `0.33`. Le journal contient `End` après une progression observée jusqu’à environ `Time=15.33 s`; aucun `FOAM FATAL`, problème de transfert de masse ou défaut de fraction n’est observé. Le plafond global a été consommé par le coût des corrections MULES et des bilans, mais la fin du calcul est présente dans le journal.

Statut : **validé OF13 — colonne gaz/eau avec évaporation et bilan massique jusqu’à `End`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
