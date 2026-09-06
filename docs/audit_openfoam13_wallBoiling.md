# Audit OF13 — multiRegion/CHT/wallBoiling

La référence OpenFOAM 13 utilise deux régions, `fluid` et `solid`, et le solveur régional `multiphaseEuler` pour le fluide. Le cas contient les phases `gas` et `liquid`, les champs `alpha.gas`, `alpha.liquid`, `T.gas`, `T.liquid`, ainsi que le modèle `heatTransferLimitedPhaseChange` et la loi `wallBoiling` sur la paroi. Le solide est résolu thermiquement avec `solid`.

Le runner `181_multiRegion_CHT_wallBoiling/run.py` reproduit l’Allrun : `blockMesh`, `extrudeMesh`, `splitMeshRegions -cellZones`, préparation `paraFoam` des régions, `decomposePar -allRegions`, `foamMultiRun` parallèle à quatre domaines, reconstruction `-latestTime -allRegions` et les deux post-traitements `foamPostProcess` officiels pour la coupe et les propriétés de paroi.

La validation confirme l’extrusion, la séparation des régions et le démarrage du solveur multiphasique. MULES résout `alpha.gas` et `alpha.liquid`; les modèles `heatTransferLimitedPhaseChange` et `wallBoiling` s’exécutent à chaque pas et produisent les champs `mDot`, `wetFraction`, `dDeparture`, `fDeparture` et `nucleationSiteDensity`. Aucun `FOAM FATAL` ni erreur de dictionnaire n’est observé.

Le cas est très coûteux : après environ 278 secondes, il atteint `Time≈1,93 s` sur `8 s`, puis le plafond de 300 secondes interrompt la validation avant `End`. Les post-traitements et la reconstruction n’ont donc pas pu être confirmés dans cette exécution.

Statut : **accepté avec réserve — limite de temps**.
