# Audit OF13 — multiphaseEuler/wallBoilingPolydisperse

La référence OpenFOAM 13 est la variante polydisperse de l’ébullition pariétale. Elle exécute `blockMesh`, `extrudeMesh`, `decomposePar` à 4 domaines, `foamRun` en parallèle, `reconstructPar -latestTime`, puis les post-traitements `graphCell` et `patchSurface`. Le cas utilise un mélange eau/gaz avec quatre groupes de tailles de bulles.

Le runner `234_multiphaseEuler_wallBoilingPolydisperse/run.py` importe par FoamPilot tous les champs, dictionnaires et données de validation, ainsi que les ressources thermodynamiques OF13 compressées (`wallBoiling-liquid.gz`, `wallBoiling-vapour.gz`, `wallBoiling-saturation.csv`). Il reproduit la chaîne complète avec environnement OF13 explicite et 4 processus MPI.

La validation progresse sans erreur fatale mais dépasse le plafond de temps. `blockMesh`, `extrudeMesh` et `decomposePar` terminent avec succès. `foamRun -parallel` atteint `Time≈2.74 s` sur `4 s` au plafond de 300 secondes. Les fractions `gas/liquid` restent normalisées, les quatre groupes de tailles sont résolus, le diamètre moyen de Sauter évolue normalement et les propriétés d’ébullition pariétale (`wetFraction`, `dDeparture`, `fDeparture`, `nucleationSiteDensity`) sont calculées à chaque pas. Les températures restent dans la plage physique observée `341–362 K`. Aucun `FOAM FATAL`, arrêt MPI ou divergence n’apparaît. La reconstruction et les post-traitements finaux ne sont pas atteints dans le budget.

Statut : **accepté avec réserve — ébullition pariétale polydisperse stable jusqu’à `Time≈2.74 s` sur `4 s`, reconstruction hors budget**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037) et `BaseSolver.import_reference_asset`; aucune nouvelle API publique n’a été ajoutée.
