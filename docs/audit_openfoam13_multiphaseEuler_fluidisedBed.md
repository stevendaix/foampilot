# Audit OF13 — multiphaseEuler/fluidisedBed

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, `decomposePar`, `foamRun -parallel`, puis `reconstructPar`. Le cas est un lit fluidisé gaz/particules avec phases `air` et `particles`. La zone `bed` est initialisée par `setFields` avec `alpha.air=0.45` et `alpha.particles=0.55`; les propriétés granulaires, la théorie cinétique des particules, les modèles de traînée, la turbulence et les champs `Theta.particles`, `nut.particles`, `alphaMean.particles` sont conservés. Le contrôle impose `endTime=2`, `deltaT=2e-4`, `writeInterval=0.01` et `maxDeltaT=0.01`.

Le runner `223_multiphaseEuler_fluidisedBed/run.py` importe par FoamPilot les champs air/particles, les dictionnaires `constant/system` et les fonctions de moyenne, puis reproduit `blockMesh`, `setFields`, `decomposePar` simple à 4 domaines, `foamRun -parallel` à 4 processus et `reconstructPar` sous environnement OF13 explicite. Aucune étape absente de l’Allrun n’est ajoutée.

La validation est complète. Le maillage et l’initialisation de la zone `bed` réussissent. Le calcul parallèle termine à `Time=2 s` et la reconstruction parcourt les sorties jusqu’à `End`. Les champs fluides et granulaires, les moyennes de phase, la température, la fraction particulaire et les variables de théorie cinétique sont reconstruits; aucun `FOAM FATAL`, défaut MPI ou instabilité terminale n’est observé.

Statut : **validé OF13 — lit fluidisé air/particules jusqu’à `End=2 s`, calcul parallèle et reconstruction réussis**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
