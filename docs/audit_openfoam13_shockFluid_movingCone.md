# Audit OF13 — shockFluid/movingCone

La référence OpenFOAM 13 exécute `blockMesh`, `decomposePar -cellProc`, puis génère deux maillages de mapping (`1e-05` et `2e-05`) avec `blockMesh -mesh` et des dictionnaires de décomposition propres à chaque temps. Le calcul utilise `foamRun -parallel` avec le maillage mobile défini par `dynamicMeshDict`, puis `reconstructPar -cellProc`. Le cas vise `endTime=2,25e-5 s` avec `maxCo=0,2`.

Le runner `242_shockFluid_movingCone/run.py` importe par FoamPilot les champs, dictionnaires, `dynamicMeshDict` et sous-arbres de maillage temporel. Il remplace les liens symboliques de l’Allrun par l’import explicite des `decomposeParDict` dans `system/`, puis exécute les deux appels `blockMesh -mesh`, les décompositions, `foamRun -parallel` à 4 domaines et la reconstruction.

La validation est complète. Les maillages initial et temporels, les décompositions à 4 domaines, le calcul compressible mobile et la reconstruction atteignent `Time=2,25e-5 s` et `End`. Les champs `cellMotionU`, `meshPhi`, `rho`, `U`, `p` et `T` sont reconstruits. Des avertissements OF13 de planéité des patches wedge et de géométrie de frontière apparaissent pendant la reconstruction, sans `FOAM FATAL`, NaN ou arrêt prématuré.

Statut : **validé OF13 — maillage mobile `shockFluid` parallèle à 4 domaines jusqu’à `End=2,25e-5 s`**, avec avertissements géométriques non bloquants.

Seules les fonctions FoamPilot existantes ont été utilisées; aucune nouvelle API n’a été ajoutée pour ce tutoriel.
