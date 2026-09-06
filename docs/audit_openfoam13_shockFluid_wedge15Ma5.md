# Audit OF13 — shockFluid/wedge15Ma5

La référence OpenFOAM 13 ne fournit pas d’Allrun à la racine, mais ses dictionnaires définissent un cas compressible `shockFluid` avec maillage wedge et la chaîne minimale `blockMesh → foamRun`. Le calcul utilise les champs `T/U/p`, `endTime=0,2 s`, `deltaT=1e-4 s`, `maxCo=1` et `maxDeltaT=1e-6 s`.

Le runner `245_shockFluid_wedge15Ma5/run.py` importe par FoamPilot les champs et dictionnaires OF13, puis exécute `blockMesh` et `foamRun` sous l’environnement OF13 explicite. Aucune logique shell de préparation de cas n’est utilisée.

La validation est complète : le calcul atteint `Time=0,2 s` et `End` en environ 3 secondes. Le Courant maximal reste inférieur à `0,19`, les variables compressibles sont résolues sans divergence, et aucun `FOAM FATAL`, NaN ou erreur de lecture n’apparaît.

Statut : **validé OF13 — calcul compressible wedge sériel jusqu’à `End=0,2 s`**.

Seules les fonctions FoamPilot existantes ont été utilisées; aucune nouvelle API n’a été ajoutée.
