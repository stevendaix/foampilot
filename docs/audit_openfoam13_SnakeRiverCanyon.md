# Audit OF13 — movingMesh/SnakeRiverCanyon

Source locale : `/opt/openfoam13/tutorials/movingMesh/SnakeRiverCanyon`.

Le tutoriel ne fournit pas d’Allrun dans OpenFOAM 13. Sa mise en données contient `blockMeshDict`, `dynamicMeshDict`, `physicalProperties`, `pointDisplacement`, `fvSchemes`, `fvSolution`, `controlDict`, `decomposeParDict` et la géométrie `constant/geometry/AcrossRiver.stl.gz`. Le maillage source est un bloc de `20x60x60` cellules, avec les patches `maxX`, `minX`, `minY`, `maxY`, `minZ` et `maxZ`. Le dictionnaire de mouvement utilise `displacementSBRStress` et une diffusivité `quadratic inverseDistance 1(minZ)`; la surface `AcrossRiver` est projetée avec `surfaceDisplacement`, vitesse `(10 10 10)` et direction fixe `(0 0 1)`. La viscosité cinématique source est `nu=0,01`.

Le contrôle définit le solveur `movingMesh`, `endTime=25`, `deltaT=1` et écriture toutes les 5 étapes. Le runner `168_movingMesh_SnakeRiverCanyon/run.py` importe les dictionnaires, le champ `pointDisplacement` et la surface compressée via FoamPilot, puis exécute `blockMesh`, `decomposePar`, `foamRun -solver movingMesh -parallel` à 2 domaines et `reconstructPar`. La validation OF13 atteint `End` à `Time=25 s`, sans `FOAM FATAL`; la reconstruction produit les temps `5,10,15,20,25`. Le mouvement de maillage est accepté avec succès et aucune nouvelle API n’est nécessaire.
