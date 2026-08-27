# Audit OF13 — shockFluid/shockTube

La référence OpenFOAM 13 exécute `blockMesh`, `setFields`, `foamRun`, puis `foamPostProcess -func sample`. Le script `createGraphs` produit ensuite des graphes gnuplot optionnels à partir de `postProcessing/sample/*/data.xy`; cette étape de visualisation externe n’est pas nécessaire à la validation FoamPilot du calcul.

Le runner `244_shockFluid_shockTube/run.py` importe par FoamPilot les champs `T/U/p`, les dictionnaires `system/`, les propriétés physiques et `setFieldsDict`, puis exécute exactement la chaîne OpenFOAM utile : `blockMesh`, `setFields`, `foamRun` et `foamPostProcess -func sample`, sous environnement OF13 explicite.

La validation est complète. `foamRun` atteint `End=0,007 s` en moins d’une seconde de temps CPU et le post-traitement produit les profils `T`, `mag(U)` et `p` aux temps `0` à `0,007 s` dans `postProcessing/sample`. Des avertissements concernant un conflit de nom de groupe `empty` sont signalés par OpenFOAM pendant `setFields` et `sample`, sans `FOAM FATAL`, NaN ou échec de production des données.

Statut : **validé OF13 — tube à choc compressible jusqu’à `End=0,007 s`, échantillonnage produit**.

Seules les fonctions FoamPilot existantes ont été utilisées; aucune nouvelle API n’a été ajoutée.
