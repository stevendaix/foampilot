# Audit OF13 — legacy/incompressible/shallowWaterFoam/squareBump

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/incompressible/shallowWaterFoam/squareBump`.

L’Allrun officielle exécute exactement `blockMesh`, `setFields`, puis `shallowWaterFoam`. Le maillage `blockMesh` contient 400 cellules (`20 x 20 x 1`) dans un domaine 2D, avec patches `sides`, `inlet`, `outlet` et `frontAndBack` de type `empty`.

Le cas utilise les champs shallow-water de la référence : `h.orig` pour la hauteur initiale, `hU.orig` pour le flux de quantité de mouvement, et `hTotal`. La configuration de `setFields` impose par défaut `h0=0`, `h=0,01` et `hU=(0,001 0 0)`. La zone boîte `bump`, `(0,45 0,45 0)` à `(0,55 0,55 0,1)`, impose `h0=0,001`, `h=0,009` et `hU=(0,0009 0 0)`, reproduisant la topographie carrée officielle.

`constant/gravitationalProperties` conserve `g=(0 0 -9,81)`, `rotating true` et `Omega=(0 0 7,292e-5)`. Le contrôle officiel est `endTime=100`, `deltaT=0,1` et écriture toutes les 1 unité de temps. Le runner `160_legacy_shallowWaterFoam_squareBump/run.py` importe les champs et dictionnaires avec FoamPilot, puis exécute uniquement les trois applications de l’Allrun.

La validation OF13 atteint `End=100 s` sans `FOAM FATAL`. Les solveurs `hUx`, `hUy` et `h` convergent, le nombre de Courant maximal est ≈`0,216` et le nombre de Courant d’onde gravitaire maximal ≈`0,627`. Aucune nouvelle fonction d’API n’est nécessaire.
