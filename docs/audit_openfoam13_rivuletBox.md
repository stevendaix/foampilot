# Audit OF13 — multiRegion/film/rivuletBox

La référence OpenFOAM 13 construit trois régions `box`, `panel` et `film`. L’Allrun exécute `blockMesh -region box`, puis `extrudeToRegionMesh -dict system/extrudeToRegionMeshDict.panel -region box` pour créer le panneau solide et `extrudeToRegionMesh -dict system/extrudeToRegionMeshDict.film -region panel` pour créer le film. Comme le couplage à trois régions n’est pas créé automatiquement par `extrudeToRegionMesh`, l’Allrun modifie ensuite les frontières de `constant/box/polyMesh/boundary` et `constant/film/polyMesh/boundary` avec `foamDictionary`. Elle prépare ParaView, décompose les trois régions à huit domaines, lance `foamMultiRun` parallèle et reconstruit avec `reconstructPar -allRegions`.

Le runner `187_multiRegion_film_rivuletBox/run.py` importe les champs et dictionnaires OF13, transcrit les deux extrusions avec leurs dictionnaires dédiés, applique les quatre propriétés de frontières mappées de la référence avec FoamPilot, puis exécute le pipeline parallèle. Le maillage box couvre `0.75×1×0.02 m`; le panneau a une épaisseur de `0.002` et le film de `0.01`. Le contrôle couple `film`, `panel solid` et `box fluid`, avec `endTime=5`, `deltaT=1e-4`, `maxCo=0.2` et `maxDeltaT=5e-3`.

La validation confirme la création des trois régions, des interfaces `mappedExtrudedWall`/`mappedFilmSurface`, de la décomposition à huit domaines et le démarrage du solveur couplé. Les équations film, panneau et boîte sont résolues; les nombres de Courant restent maîtrisés et les erreurs de continuité restent bornées. Aucun `FOAM FATAL` ni problème de dictionnaire n’est observé.

Le cas est cependant coûteux. Après environ 300 secondes, il atteint `Time≈0,665 s` sur `5 s`; le plafond de validation intervient avant `End`, et la reconstruction finale n’est pas confirmée.

Statut : **accepté avec réserve — limite de temps**.
