# Audit OF13 — solidDisplacement/plateHole

La référence OpenFOAM 13 exécute `blockMesh`, `foamRun`, puis `foamPostProcess -func "components(sigma)"` et `foamPostProcess -func graphUniform`. Le cas représente une plaque trouée en contrainte plane, avec traction uniforme `(10000 0 0)` et `endTime=100`.

Le runner `247_solidDisplacement_plateHole/run.py` importe par FoamPilot les champs `D/T`, la géométrie, les propriétés élastiques et les dictionnaires `graphUniform`, puis exécute la chaîne complète sous environnement OF13 explicite. Les fonctions de post-traitement sont appelées par FoamPilot et non par le script shell source.

La validation est complète : `blockMesh` et `foamRun` atteignent `Time=100` et `End`. Les résidus GAMG des composantes `Dx/Dy` diminuent à chaque pas et `Max sigmaEq≈28845,6`. `components(sigma)` produit les six composantes `sigmaxx`, `sigmaxy`, `sigmaxz`, `sigmayy`, `sigmayz`, `sigmazz` aux temps écrits `20`, `40`, `60`, `80` et `100`. Un avertissement non bloquant apparaît au temps initial, avant que le champ `sigma` ne soit disponible; le post-traitement termine ensuite normalement. `graphUniform` termine également sans erreur fatale.

Statut : **validé OF13 — plaque trouée sous traction jusqu’à `End=100`, contraintes et graphe produits**.

Seules les fonctions FoamPilot existantes ont été utilisées; aucune nouvelle API n’a été ajoutée.
