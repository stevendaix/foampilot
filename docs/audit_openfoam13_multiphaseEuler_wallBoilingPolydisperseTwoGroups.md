# Audit OF13 — multiphaseEuler/wallBoilingPolydisperseTwoGroups

La référence OpenFOAM 13 est une variante d’ébullition pariétale polydisperse comprenant une phase liquide et deux populations gazeuses `gas` et `gas2`. Elle exécute `blockMesh`, `extrudeMesh`, `decomposePar` à 4 domaines, `foamRun` en parallèle, `reconstructPar -latestTime`, puis `graphCell` et `patchSurface`.

Le runner `235_multiphaseEuler_wallBoilingPolydisperseTwoGroups/run.py` importe par FoamPilot tous les champs, dictionnaires et données de validation, les ressources thermodynamiques OF13 compressées et les champs de diagnostic adaptés aux deux populations gazeuses. Il reproduit la chaîne complète sous environnement OF13 explicite avec 4 processus MPI.

La validation passe `blockMesh`, `extrudeMesh` et `decomposePar`, puis démarre correctement `foamRun -parallel`. Le calcul atteint `Time≈2,14 s` sur `4 s` au plafond de 300 secondes. Les fractions `gas/gas2/liquid` restent bornées avec une somme volumique proche de 1, les distributions de tailles des deux populations sont résolues, les diamètres moyens de Sauter sont calculés et les propriétés de changement de phase sur la paroi sont mises à jour. Les températures restent dans la plage observée `341–362 K`; les flux de changement de phase et de paroi sont produits sans erreur fatale. La reconstruction et les post-traitements finaux ne sont pas atteints dans le budget.

Statut : **accepté avec réserve — ébullition pariétale à deux groupes gazeux stable jusqu’à `Time≈2,14 s` sur `4 s`, reconstruction hors budget**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037) et `BaseSolver.import_reference_asset`; aucune nouvelle API publique n’a été ajoutée.
