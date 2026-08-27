# Audit OF13 — multiRegion/film/VoFToFilm

Le cas OpenFOAM 13 convertit une phase liquide VoF en film de surface. L’Allrun exécute `blockMesh -region VoF`, `extrudeToRegionMesh -region VoF`, `setFields -region VoF`, prépare les fichiers ParaView puis lance `foamMultiRun` sériel. Le dictionnaire d’extrusion crée une région `film` à une couche, d’épaisseur `0.001`, avec `mappedExtrudedWall`, `filmWall` et `mappedFilmSurface`.

Le runner `182_multiRegion_film_VoFToFilm/run.py` importe les champs VoF/film et les propriétés OF13, puis reproduit uniquement avec FoamPilot : `blockMesh -region VoF`, `extrudeToRegionMesh -region VoF`, `setFields -region VoF`, `paraFoam -touchAll` et `foamMultiRun`. Le champ source `alpha.liquid.orig` est pris en charge par le gestionnaire d’import selon les conventions des champs de référence.

La validation confirme la création du maillage VoF et de la région film, l’initialisation de `alpha.liquid` par boîte et le démarrage des solveurs `compressibleVoF` et `film`. Les fractions restent bornées, avec une fraction liquide très faible dans l’extrait final comme attendu après transfert vers le film. Les équations de vitesse, énergie, épaisseur de film, température et pression convergent avec des erreurs de continuité de l’ordre de `1e-12`.

Le calcul atteint `Time=5 s` puis `End` en environ 81 secondes, sans `FOAM FATAL` ni erreur de dictionnaire.

Statut : **validé OF13**.
