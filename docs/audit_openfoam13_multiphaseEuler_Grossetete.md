# Audit OF13 — multiphaseEuler/Grossetete

L’Allrun OpenFOAM 13 exécute `blockMesh`, `extrudeMesh`, puis `foamRun`. Le maillage 2D est extrudé en coin wedge. Le cas `multiphaseEuler` comporte les phases `gas` et `liquid`, des propriétés thermophysiques séparées, les modèles de transport de quantité de mouvement gaz/liquide, une dispersion et une tension de surface gaz-liquide constante `sigma=0.071`. Le contrôle impose `endTime=2`, `deltaT=0.001`, `writeInterval=1` et `maxDeltaT=0.001`; la résolution utilise les corrections de fraction volumique MULES et des modèles de traînée/échange thermique.

Le runner `209_multiphaseEuler_Grossetete/run.py` importe par FoamPilot les champs suffixés de phase (`.gas`, `.liquid`), tous les dictionnaires `constant` et `system`, puis reproduit `blockMesh`, `extrudeMesh` et `foamRun` sous environnement OF13 explicite. Les propriétés gaz/liquide, le wedge, les modèles de transfert et les fonctions de validation restent ceux de la référence OF13, sans réécriture manuelle de la physique.

La validation est complète. `blockMesh` et `extrudeMesh` terminent correctement. `foamRun` atteint `Time=2 s` et `End` en environ 55 secondes. Les journaux confirment la résolution MULES de `alpha.gas` et `alpha.liquid`, avec fractions gaz et liquide bornées et Courant maximal proche de `0.25`. Les solveurs turbulents gaz/liquide convergent à chaque pas et aucun `FOAM FATAL`, défaut d’extrusion ou problème de phase n’est observé.

Statut : **validé OF13 — multiphaseEuler gaz/liquide avec extrusion wedge jusqu’à `End=2 s`**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037; aucun changement d’API supplémentaire n’a été nécessaire.
