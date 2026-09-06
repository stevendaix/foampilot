# Audit OF13 — multicomponentFluid/verticalChannelSteady

L’Allrun OpenFOAM 13 exécute `blockMesh`, `potentialFoam`, supprime `0/phi`, lance `foamRun`, puis `steadyParticleTracks`. Le cas est un canal vertical 3D stationnaire avec `fvSchemes` en `steadyState`, cloud `reactingMultiphaseCloud` injecté par `inletCentral`, fonctions de moyenne à l’outlet et suivi des champs `d U T`. Le contrôle impose `endTime=500`, `deltaT=1`, `writeInterval=20` et `purgeWrite=10`.

Le runner `208_multicomponentFluid_verticalChannelSteady/run.py` importe par FoamPilot les champs, constantes, `cloudProperties`, positions, fonctions et dictionnaires de suivi, puis reproduit `blockMesh`, `potentialFoam`, la suppression gérée de `0/phi` par `remove_case_asset`, `foamRun` stationnaire et `steadyParticleTracks` sous environnement OF13 explicite. Les modèles multiphasiques, l’injection, les conditions aux limites et les contrôles stationnaires sont conservés sans réécriture.

La validation est complète. `blockMesh` et `potentialFoam` terminent correctement; le solveur stationnaire atteint `End=500 s`. `steadyParticleTracks` traite les temps 0, 320, 340, …, 500 s et écrit les fichiers VTK `particleTracks.vtk`; environ 6 200 à 6 400 particules sont lues par temps. Aucun `FOAM FATAL`, problème de cloud, erreur de stationnarité ou erreur de post-traitement n’est observé.

Statut : **validé OF13 — solveur `steadyState` jusqu’à `End=500 s` et suivi particulaire complet**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037) et `BaseSolver.remove_case_asset(...)` pour la suppression gérée de `0/phi`.
