# Audit OF13 — multiphaseEuler/bubbleColumnEvaporatingReacting

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est une colonne gaz/eau avec évaporation liquide-gaz et chimie réactive dans la phase gaz. Les champs réactifs comprennent notamment `CO.gas`, `CO2.gas`, `H2O.gas`, `air.gas` et `Ydefault.gas`; les fonctions demandent aussi `Qdot.gas`. La zone liquide est initialisée à `alpha.gas=0.01` et `alpha.liquid=0.99`. Les propriétés thermophysiques, modèles de transfert de masse, réaction, tension de surface, turbulence et contrôles de correction de volume/drag sont ceux de la référence OF13. Le contrôle impose `endTime=100`, `deltaT=0.001`, `writeInterval=1` et `maxDeltaT=1`.

Le runner `217_multiphaseEuler_bubbleColumnEvaporatingReacting/run.py` importe par FoamPilot les champs de phase et d’espèces, les propriétés thermophysiques, les dictionnaires de chimie et de transfert, les fonctions `Qdot` et les autres assets OF13, puis reproduit `blockMesh`, `setFields` et `foamRun` sous environnement OF13 explicite. Aucun champ réactif ni modèle physique n’est réécrit.

La validation du maillage et de l’initialisation réussit. Pendant `foamRun`, `massDiffusionLimitedPhaseChange: phaseChange` est actif; les équations de `CO.gas`, `CO2.gas` et `H2O.gas` convergent avec des résidus finaux très faibles. Les corrections MULES conservent la somme des fractions à 1 et le Courant maximal observé reste inférieur à environ `0.31`. Le journal contient `End` après une progression observée jusqu’à environ `Time=5.03 s`; aucun `FOAM FATAL`, défaut de chimie, erreur de transfert ou instabilité n’est observé. Le budget de session a été consommé par le coût des boucles de chimie, d’évaporation et de fractions, mais la fin du calcul est présente dans le journal.

Statut : **validé OF13 — colonne gaz/eau réactive avec évaporation et chaleur de réaction jusqu’à `End`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
