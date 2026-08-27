# Audit OF13 — shockFluid/LadenburgJet60psi

La référence OpenFOAM 13 est un cas compressible `shockFluid` de test dont l’état initial est fourni dans `0/T`, `0/U` et `0/p`; les fichiers `0.orig` documentent les champs sources. Le calcul est défini jusqu’à `endTime=2e-05 s`, avec pas initial `1e-10 s`, ajustement automatique et `maxCo=0,5`. La décomposition de référence utilise 8 domaines simples.

Le runner `238_shockFluid_LadenburgJet60psi/run.py` importe par FoamPilot les champs, dictionnaires et propriétés physiques, puis exécute `blockMesh`, `decomposePar`, `foamRun -parallel` avec 8 processus MPI et `reconstructPar -latestTime`, sous environnement OF13 explicite.

La validation est complète. Le calcul parallèle atteint `Time=2e-05 s` et `End`, puis la reconstruction du dernier temps termine correctement. Le Courant maximal reste proche de `0,5`, la densité est résolue et aucune divergence, erreur MPI ou `FOAM FATAL` n’apparaît.

Statut : **validé OF13 — calcul compressible `shockFluid` parallèle à 8 domaines jusqu’à `End=2e-05 s`**.

Le runner utilise `BaseSolver.run_command(environment=...)`; aucune nouvelle API publique n’a été ajoutée.
