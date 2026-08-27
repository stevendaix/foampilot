# Audit OF13 — multicomponentFluid/parcelInBox

Le tutoriel OpenFOAM 13 ne fournit pas d’Allrun de calcul. Sa mise en données est un cas mono-région `multicomponentFluid` avec `blockMesh` puis `foamRun`. Les particules sont configurées dans `constant/cloudProperties` et `constant/cloudPositions`; `constant/fvModels` active le modèle `clouds` avec `liblagrangianParcel.so` et la force de flottabilité. Le contrôle impose `endTime=0.5`, `deltaT=1e-3`, `writeInterval=0.1`, ajustement de pas, `maxCo=5` et `maxDeltaT=1e-3`. Les champs fluides comprennent notamment `air`, `H2O`, `T`, `U`, `p` et `G`.

Le runner `202_multicomponentFluid_parcelInBox/run.py` importe intégralement par FoamPilot les champs, constantes et dictionnaires de référence, notamment `cloudProperties`, `cloudPositions` et `fvModels`, puis exécute `blockMesh` et `foamRun` avec l’environnement OpenFOAM 13 explicite. Aucun fichier de nuage n’est généré manuellement et aucune commande shell hors FoamPilot n’est utilisée.

La validation est complète. `blockMesh` termine correctement. `foamRun` atteint `Time=0.5 s` et `End`; le journal confirme la résolution d’un nuage 3D `cloud` avec un parcel et le transport de l’espèce `H2O`. Le Courant fluide reste quasi nul conformément à la configuration, les erreurs de continuité sont de l’ordre de `10^-9` à `10^-12` et aucun `FOAM FATAL`, problème de cloud ou erreur de champ n’est observé.

Statut : **validé OF13 — nuage particulaire et calcul multicomposant jusqu’à `End=0,5 s`**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037; aucun changement d’API supplémentaire n’a été nécessaire.
