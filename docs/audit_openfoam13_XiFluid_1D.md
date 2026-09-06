# Audit OF13 — XiFluid/1D

Le cas `XiFluid/1D` est un tutoriel OpenFOAM 13 effectivement présent mais non encore inscrit dans la matrice initiale. Son Allrun exécute `blockMesh`, `setFields`, puis `foamRun` avec le solveur `XiFluid`. La mise en données contient les champs de combustion `Xi`, `b`, `Tu`, `T`, `p`, `U`, `alphat`, `epsilon`, `k` et `nut`, ainsi que `combustionProperties`, `fvModels` avec ignition et les dictionnaires thermiques/turbulents.

Le runner `248_XiFluid_1D/run.py` importe par FoamPilot l’ensemble des champs et dictionnaires OF13, en convertissant correctement les fichiers `.orig` lors de l’import des champs, puis exécute `blockMesh`, `setFields` et `foamRun` avec environnement OF13 explicite. Aucune logique shell de cas n’est utilisée.

La validation est complète : le calcul atteint `Time=0,2 s` et `End` en environ 12 secondes. Le Courant maximal reste proche de `0,38`; les erreurs de continuité locales restent faibles et l’erreur cumulative est de l’ordre de `5e-13`. Aucun `FOAM FATAL`, NaN, divergence ou erreur de lecture n’apparaît.

Statut : **validé OF13 — flamme prémélangée 1D `XiFluid` jusqu’à `End=0,2 s`**.

Aucune nouvelle API n’a été ajoutée; le runner utilise les primitives FoamPilot existantes, notamment l’import des champs de référence et l’exécution avec environnement explicite.
