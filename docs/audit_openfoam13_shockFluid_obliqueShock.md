# Audit OF13 — shockFluid/obliqueShock

La référence OpenFOAM 13 ne fournit pas d’Allrun à la racine, mais ses dictionnaires définissent une chaîne minimale `blockMesh → foamRun`. Le cas compressible `shockFluid` utilise les champs `T/U/p`, un pas initial `deltaT=0,0025 s`, un ajustement automatique limité par `maxCo=0,2` et `endTime=10 s`.

Le runner `243_shockFluid_obliqueShock/run.py` importe par FoamPilot les champs et dictionnaires OF13, puis exécute `blockMesh` et `foamRun` avec l’environnement OF13 explicite. Aucune logique shell de préparation de cas n’est utilisée.

La validation est complète : le calcul atteint `Time=10 s` et `End` en environ 2,4 secondes. Le Courant maximal reste proche de `0,20`, les variables compressibles sont résolues sans divergence et aucun `FOAM FATAL`, NaN ou erreur de lecture n’apparaît.

Statut : **validé OF13 — calcul `shockFluid` sériel jusqu’à `End=10 s`**.

Seules les fonctions FoamPilot existantes ont été utilisées; aucune nouvelle API n’a été ajoutée.
