# Audit OF13 — multiRegion/film/rivuletPanel

La référence OpenFOAM 13 construit la région solide `panel` par `blockMesh -region panel`, puis décompose uniquement le maillage panel avec `decomposePar -region panel -noFields`. Elle exécute ensuite `extrudeToRegionMesh -parallel -dict system/extrudeToRegionMeshDict.film -region panel`, ce qui crée la région `film` à une épaisseur de `0.01` avec une face opposée `empty` et les patches couplés `mappedWall`/`mappedFilmWall`. Après extrusion, la référence reconstruit les maillages avec `reconstructPar -allRegions`, décompose les champs de toutes les régions avec `decomposePar -fields -allRegions`, crée les fichiers ParaView, lance `foamMultiRun` en parallèle sur quatre domaines et reconstruit à nouveau toutes les régions.

Le runner `188_multiRegion_film_rivuletPanel/run.py` importe les fichiers OF13 `0/film`, `0/panel`, `constant` et `system` via les gestionnaires FoamPilot, puis reproduit exactement cette séquence. Les commandes sont exécutées par `BaseSolver.run_command`; l’environnement OF13 est fourni explicitement et le `bashrc` OF13 est sourcé dans le processus enfant pour sélectionner les bibliothèques OpenMPI et Scotch ThirdParty, sans appel shell direct hors du contrôle FoamPilot.

La validation est complète. `blockMesh` termine correctement et produit le panel de 43 200 cellules. `decomposePar` sélectionne Scotch et répartit le maillage sur quatre domaines avec 10 809, 10 808, 10 760 et 10 823 cellules. L’extrusion parallèle termine correctement. La reconstruction initiale et la décomposition des champs de `film` et `panel` réussissent. Le calcul couplé `foamMultiRun` atteint `Time=5 s` et produit les temps écrits `0.1` à `5`; le nombre de Courant maximal observé reste inférieur à `0.2`. La reconstruction finale contient les champs film (`alpha`, `delta`, `U`, `T`, `p`, `phi`, `alphaRhoPhi`) et panneau (`T`). Aucun `FOAM FATAL`, aucune erreur de bibliothèque et aucun échec de reconstruction ne sont observés.

Statut : **validé OF13 — `End=5 s`, reconstruction film/panel réussie**.

## Évolution API

`BaseSolver.run_command` accepte désormais un paramètre optionnel `environment: Dict[str, str]`. Cet environnement est fusionné avec celui du processus avant l’appel du sous-processus. Cette extension générique permet aux runners FoamPilot de déclarer explicitement les environnements OpenFOAM, MPI et bibliothèques ThirdParty nécessaires, notamment lorsque le processus hôte n’a pas sourcé le `bashrc` de la version OpenFOAM de référence.
