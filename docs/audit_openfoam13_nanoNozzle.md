# Audit OF13 — legacy/lagrangian/mdFoam/nanoNozzle

Source locale : `/opt/openfoam13/tutorials/legacy/lagrangian/mdFoam/nanoNozzle`.

L’Allrun officielle exécute `blockMesh`, `decomposePar`, `mdInitialise -parallel`, `mdFoam -parallel`, puis `reconstructPar`. Le maillage est un nano-nozzle multi-section avec zones `sectionA`, `sectionB` et `sectionC`, des parois `front/back/top/bottom`, une entrée `sectionAEnd` et une sortie murale `sectionCEnd`. `blockMesh` produit `27 136` cellules. La décomposition source impose `4` processeurs; le runner a été corrigé après un premier contrôle qui avait détecté une tentative à 2 domaines.

Le contrôle OF13 définit `endTime=2e-13 s`, `deltaT=1e-15 s` et écriture toutes les `5e-14 s`. Les trois sections sont initialisées à une densité massique `1004`, une température `298 K` et une vitesse nulle. `mdEquilibrationDict` fixe la température cible à `298 K`. Les propriétés multi-sites de la molécule `water`, le potentiel Lennard-Jones, l’électrostatique amortie, le tethering et la géométrie source sont importés intégralement.

Le runner `167_legacy_mdFoam_nanoNozzle/run.py` utilise exclusivement les managers FoamPilot et `solver.run_command`; les étapes MPI ajoutent `-parallel` et utilisent quatre domaines. La validation OF13 corrigée réussit `blockMesh`, `decomposePar` et `mdInitialise -parallel`, avec `110 197` molécules initialisées. `mdFoam -parallel` démarre sans `FOAM FATAL`, conserve environ `109 669` molécules et atteint `Time=7e-15 s` après environ 27 secondes. Le calcul a été arrêté proprement pour coût disproportionné avant `2e-13 s`; `reconstructPar` n’a donc pas été exécuté dans cette validation. Statut : accepté avec réserve. Aucune nouvelle API nécessaire.
