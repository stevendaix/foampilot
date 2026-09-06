# Audit OF13 — multiRegion/film/cylinderDripping

La référence OpenFOAM 13 utilise l’Allrun sérielle : `blockMesh -region fluid`, `extrudeToRegionMesh -region fluid`, préparation `paraFoam -touchAll`, puis `foamMultiRun`. Le contrôle couple `fluid multicomponentFluid` et `film film`, avec `endTime=1`, `deltaT=1e-2`, `writeInterval=0.01`, `maxCo=0.3` et ajustement automatique du pas. L’extrusion crée une couche film d’épaisseur `0.001` avec les patches mappés `filmWall` et `mappedFilmSurface`.

Le runner `184_multiRegion_film_cylinderDripping/run.py` importe les champs `fluid` et `film`, les espèces `N2/O2/H2O`, les propriétés de particules et les dictionnaires d’extrusion. Il reproduit uniquement avec FoamPilot les étapes de l’Allrun, sans appel direct à un script shell de tutoriel.

La validation confirme le maillage fluide, l’extrusion et le démarrage des solveurs multicomposant/film. Les particules sont correctement transférées : l’extrait final indique `New film detached parcels = 1105`, sans particules échappées ni collées. Les équations fluides et film convergent, les espèces `O2/H2O` sont résolues et les erreurs de continuité restent bornées.

Le calcul atteint `Time=1 s` et `End` en environ 20 secondes, avec `paraFoam -touchAll` exécuté et aucun `FOAM FATAL`.

Statut : **validé OF13**.
