# Audit OF13 — solidDisplacement/beamEndLoad

La référence OpenFOAM 13 ne fournit pas d’Allrun à la racine, mais ses dictionnaires définissent un cas solide sériel `solidDisplacement` avec la chaîne `blockMesh → foamRun`. Le champ mécanique `D` applique une traction uniforme `(0 10000 0)` sur `tractionEnd`; le champ `T` est importé et le modèle élastique utilise `planeStress yes` et `thermalStress no`. Le calcul vise `endTime=10000` avec `deltaT=1`.

Le runner `246_solidDisplacement_beamEndLoad/run.py` importe par FoamPilot les champs `D/T`, les propriétés physiques et les dictionnaires OF13, configure le solveur `solidDisplacement` et exécute `blockMesh` puis `foamRun` avec environnement OF13 explicite. Aucune logique shell de cas n’est utilisée.

La validation est complète : le calcul atteint `Time=10000` et `End` en environ 2 secondes. Les solveurs GAMG des composantes de déplacement convergent à chaque pas avec résidus finaux très faibles; aucune erreur de lecture, divergence, NaN ou `FOAM FATAL` n’apparaît.

Statut : **validé OF13 — poutre sous traction d’extrémité jusqu’à `End=10000`**.

Seules les fonctions FoamPilot existantes ont été utilisées; aucune nouvelle API n’a été ajoutée.
