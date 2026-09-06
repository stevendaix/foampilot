# Audit OF13 — legacy/electromagnetics/mhdFoam/hartmann

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/electromagnetics/mhdFoam/hartmann`.

L’Allrun officielle exécute `blockMesh`, `mhdFoam`, puis `foamPostProcess -func sample`. Le maillage `blockMesh` produit 4 000 cellules dans un canal 2D de 20 par 2, avec une épaisseur d’une cellule et des patches `inlet`, `outlet`, `lowerWall`, `upperWall` et `frontAndBack` de type `empty`.

Les propriétés de référence sont `rho=1`, `nu=1`, `mu=1` et `sigma=1`. Le champ magnétique `B` est initialisé à `(0 20 0)` et imposé sur les parois; `U` entre à `(1 0 0)`, tandis que `p`, `pB`, `Ux`, `Uy` et `Uz` conservent les conditions de la référence. Le contrôle officiel est `endTime 2`, `deltaT 0.005` et écriture toutes les 100 étapes. Le dictionnaire `system/sample` échantillonne `Ux` sur 100 points le long de la ligne centrale.

Le runner `155_legacy_mhdFoam_hartmann/run.py` importe tous les champs, propriétés et dictionnaires par les gestionnaires FoamPilot dédiés, puis exécute `blockMesh`, `mhdFoam` et `foamPostProcess -func sample` via `run_command`. La validation atteint `End=2 s`; le maximum de Courant observé est ≈`0,038`, l’erreur de divergence du flux magnétique reste de l’ordre de `10^-9`, et les profils `postProcessing/sample/*/line_centreProfile.xy` sont produits. Aucun `FOAM FATAL` n’est observé. Aucune nouvelle fonction d’API n’est nécessaire.
