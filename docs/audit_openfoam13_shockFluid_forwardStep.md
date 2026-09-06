# Audit OF13 — shockFluid/forwardStep

La référence OpenFOAM 13 ne fournit pas d’Allrun à la racine, mais ses dictionnaires définissent une chaîne minimale déterministe `blockMesh → foamRun`. Le cas compressible `shockFluid` utilise les champs `T/U/p`, un pas initial `deltaT=0,002 s`, un pas maximal `maxDeltaT=1 s` et `endTime=4 s`.

Le runner `241_shockFluid_forwardStep/run.py` importe par FoamPilot tous les champs et dictionnaires de la référence, puis exécute `blockMesh` et `foamRun` avec l’environnement OF13 explicite. Aucune commande shell de préparation de cas n’est utilisée.

La validation est complète : le maillage et le calcul compressible atteignent `Time=4 s` et `End` en environ 31 secondes. Le Courant maximal reste proche de `0,20`, les variables `rho`, `T`, `U` et `p` sont résolues sans divergence, et aucun `FOAM FATAL`, NaN ou erreur de lecture n’apparaît.

Statut : **validé OF13 — calcul `shockFluid` sériel jusqu’à `End=4 s`**.

Seules les fonctions FoamPilot existantes ont été utilisées; aucune nouvelle API n’a été ajoutée.
