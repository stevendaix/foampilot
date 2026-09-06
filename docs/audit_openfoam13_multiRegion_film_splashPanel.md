# Audit OF13 — multiRegion/film/splashPanel

La référence OpenFOAM 13 exécute `blockMesh -region fluid`, puis `extrudeToRegionMesh -region fluid` avec `system/fluid/extrudeToRegionMeshDict`. Le dictionnaire crée la région `film` à une épaisseur de `0.002`, avec `intrude yes`, `adaptMesh no`, un patch fluid `mappedExtrudedWall`, un patch film `filmWall` et une surface opposée `mappedFilmSurface`. L’Allrun crée ensuite les fichiers ParaView avec `paraFoam -touchAll`, puis lance `foamMultiRun` en série. Le `controlDict` couple `fluid` avec `multicomponentFluid` et `film` avec `film`, à `End=1 s`, `deltaT=1e-4`, `maxCo=0.3` et `maxDeltaT=1e-3`.

Le runner `189_multiRegion_film_splashPanel/run.py` importe les champs `fluid/{H2O,N2,O2,T,U,p,p_rgh}` et `film/{T,U,delta,p}`, les propriétés physiques, les modèles de transport, les propriétés de nuage et les dictionnaires de région via FoamPilot. Il reproduit ensuite la séquence officielle `blockMesh -region fluid`, `extrudeToRegionMesh -region fluid`, `paraFoam -touchAll` et `foamMultiRun` sans opération de fichier ou appel utilitaire direct hors de `BaseSolver.run_command`.

La validation est complète. `blockMesh` crée le maillage fluid de 4 000 cellules et le patch `film`; l’extrusion crée la région film et ajoute deux patches inter-régions. `paraFoam -touchAll` produit les fichiers `fluid` et `film`. Le calcul couplé atteint `End` à `Time=1 s` en environ 7 secondes, avec des solveurs fluid multicomposant et film actifs. Les résultats indiquent notamment 1 819 parcelles absorbées par le film, 1 000 nouvelles parcelles de splash, aucune parcelle échappée ou collée, et une température fluid de `300 K`. Aucun `FOAM FATAL`, aucune erreur de bibliothèque et aucune erreur de région ne sont observés.

Statut : **validé OF13 — `End=1 s`, calcul fluid/film réussi**.

L’extension générique `BaseSolver.run_command(environment=...)`, documentée sous API-037, est utilisée pour charger l’environnement OF13/MPI/ThirdParty dans le processus enfant; la validation montre que l’Allrun reste reproductible depuis un processus hôte non initialisé.
