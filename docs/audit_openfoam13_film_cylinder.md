# Audit OF13 — multiRegion/film/cylinder

La référence OpenFOAM 13 propose une Allrun sérielle et une Allrun parallèle. La variante parallèle exécute `blockMesh -region fluid`, `decomposePar -region fluid -noFields`, `extrudeToRegionMesh -region fluid` en parallèle, `decomposePar -fields -copyZero`, `foamMultiRun` parallèle, `reconstructPar -allRegions` et `paraFoam -touchAll`.

Le runner `183_multiRegion_film_cylinder/run.py` reproduit cette variante avec FoamPilot. Il importe les champs fluides et film, les espèces `N2/O2/H2O`, les propriétés physiques, le nuage de particules et les dictionnaires d’extrusion. L’extrusion crée une région film d’une couche et d’épaisseur `0.01`, avec les patches mappés OF13.

Une première exécution a révélé que l’appel MPI devait transmettre explicitement l’option OpenFOAM `-parallel` à `extrudeToRegionMesh` et `foamMultiRun`; sans cette option, le maillage film restait global et `reconstructPar` ne pouvait pas reconstruire la région. Le helper parallèle du runner a été corrigé pour ajouter `-parallel`, sans nouvelle fonction de cœur nécessaire.

La validation corrigée confirme la présence du maillage `fluid` et `film` dans les quatre processeurs. Le calcul multicomposant/film atteint `Time=20 s` et `End`; les espèces fluides sont résolues, les particules sont transférées au film et les équations de vitesse, énergie et épaisseur de film convergent. Après nettoyage du stockage temporaire, `reconstructPar -allRegions` reconstruit les champs `fluid` et `film` jusqu’à `Time=20 s`, et `paraFoam -touchAll` s’exécute sans erreur. Aucun `FOAM FATAL` ne subsiste.

Statut : **validé OF13**.
