# Audit OF13 — legacy/incompressible/adjointShapeOptimisationFoam/pitzDaily

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/incompressible/adjointShapeOptimisationFoam/pitzDaily`.

L’Allrun officielle exécute `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/pitzDaily`, puis `adjointShapeOptimisationFoam`. La ressource de maillage paramétrée produit le canal pitzDaily avec une épaisseur de cellule et des patches `inlet`, `outlet`, `upperWall`, `lowerWall`, `frontAndBack`.

La mise en données comprend les champs primaux `U`, `p`, `k`, `epsilon`, `nut` et les champs adjoints `Ua`, `pa`, ainsi que `constant/physicalProperties`, `constant/momentumTransport`, `fvSchemes`, `fvSolution` et `fvConstraints`. La turbulence `kEpsilon` et la viscosité cinématique `nu=1e-5` sont conservées. Le contrôle officiel utilise `endTime 1000`, `deltaT 1`, une écriture toutes les 100 étapes et une précision de 12 chiffres.

Le runner `156_legacy_adjointShapeOptimisationFoam_pitzDaily/run.py` importe les fichiers de référence par les gestionnaires FoamPilot dédiés, importe la ressource dans `system/pitzDaily`, puis exécute `blockMesh -dict system/pitzDaily` et `adjointShapeOptimisationFoam` avec `run_command`. La validation atteint `End=1000 s`; les résidus primaux et adjoints restent finis, les erreurs de continuité sont contrôlées et aucun `FOAM FATAL` n’est observé. Aucune nouvelle fonction d’API n’est nécessaire.
