# Audit OF13 — incompressibleFluid/wingMotion

Le tutoriel parent OpenFOAM 13 enchaîne `blockMesh`, `snappyHexMesh` avec l’aile OBJ `wing_5degrees.obj`, `extrudeMesh`, `createPatch`, un calcul stationnaire, `mapFields` vers le cas transitoire, `decomposePar`, `foamRun -parallel` avec maillage mobile, puis `reconstructPar`.

Le runner `251_incompressibleFluid_wingMotion/run.py` reproduit la chaîne complète avec imports FoamPilot des trois sous-cas officiels (`wingMotion_snappyHexMesh`, `wingMotion2D_steady`, `wingMotion2D_transient`), copie FoamPilot du maillage et des champs, et environnement OF13/MPI explicite.

La validation passe le maillage snappy, l’extrusion, `createPatch`, le calcul stationnaire, le mapping et la décomposition transitoire. Le calcul mobile transitoire reste stable jusqu’à `Time≈0,2076 s` sur `End=5 s` au plafond de 300 secondes; Courant maximal proche de `0,897`, erreurs de continuité de l’ordre de `1e-13`, aucun `FOAM FATAL`, NaN ou divergence observée. La reconstruction finale n’est pas atteinte dans le budget.

Statut : **accepté avec réserve — chaîne de préparation et calcul stationnaire validés; transitoire stable à `0,2076/5 s`, hors budget**.

Aucune nouvelle API n’a été ajoutée; le runner réutilise les primitives FoamPilot existantes.
