# Audit OF13 — legacy/lagrangian/mdEquilibrationFoam/periodicCubeArgon

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/lagrangian/mdEquilibrationFoam/periodicCubeArgon`.

L’Allrun officielle exécute uniquement `blockMesh`, `mdInitialise` et `mdEquilibrationFoam`. Le maillage est un cube unique de `12 x 12 x 12` cellules, converti à l’échelle `2,462491658e-9 m`, avec six patches `cyclic` appariés selon X, Y et Z. Le dictionnaire `decomposeParDict` officiel existe avec deux domaines, mais n’est pas appelé par l’Allrun et reste donc importé sans exécution.

`mdInitialiseDict` conserve une zone `liquid` de densité massique `1220`, température `300 K`, vitesse nulle et réseau `Ar` sur une cellule `(1 1 1)`. `moleculeProperties` est importé intégralement. `potentialDict` conserve le potentiel pair `maitlandSmith` Ar–Ar, le potentiel électrostatique amorti, les tables, la limite d’énergie potentielle `1e-18`, le tethering et la gravité nulle.

Le contrôle officiel est `endTime=5e-11 s`, `deltaT=1e-14 s` et écriture toutes les `5e-12 s`. Le runner `165_legacy_mdEquilibrationFoam_periodicCubeArgon/run.py` utilise uniquement les gestionnaires FoamPilot `fields_manager`, `constant`, `system` et `run_command`, sans commande shell dans le runner. La validation OF13 démarre correctement les trois étapes, crée le maillage et initialise exactement `2197` molécules d’argon. `mdEquilibrationFoam` reste stable, conserve une densité massique d’environ `1219,9999997` et atteint `Time=4,675e-11 s` sur un plafond de 300 s, sans `FOAM FATAL`; le temps officiel complet n’est pas atteint dans cette limite. Statut : accepté avec réserve pour coût d’exécution, aucune nouvelle API nécessaire.
