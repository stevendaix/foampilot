# Audit OF13 — multiphaseEuler/wallBoilingIATE

La référence OpenFOAM 13 exécute `blockMesh`, `extrudeMesh`, `decomposePar` à 4 domaines, `foamRun` en parallèle, `reconstructPar -latestTime`, puis les post-traitements `graphCell` et `patchSurface` des propriétés d’ébullition sur la paroi `wall`. Le cas est un écoulement eau/gaz avec modèle IATE et ébullition pariétale.

Le runner `233_multiphaseEuler_wallBoilingIATE/run.py` importe par FoamPilot tous les champs, dictionnaires et données de validation, ainsi que les ressources thermodynamiques OF13 `wallBoiling-liquid.gz`, `wallBoiling-vapour.gz` et `wallBoiling-saturation.csv`. Il reproduit la chaîne complète avec environnement OF13 explicite et 4 processus MPI. La première tentative a détecté l’absence des ressources décompressées; le runner a été corrigé pour importer les fichiers compressés aux chemins attendus par le lecteur OpenFOAM.

La validation corrigée est complète. `blockMesh`, `extrudeMesh`, `decomposePar`, `foamRun -parallel`, `reconstructPar -latestTime`, `graphCell` et `patchSurface` terminent avec succès. Le calcul atteint `Time=4 s` et `End`. Les champs `wallBoiling:wetFraction`, `wallBoiling:dDeparture`, `wallBoiling:fDeparture` et `wallBoiling:nucleationSiteDensity` sont générés sur la paroi, de même que les champs gaz/liquide demandés par `graphCell`. Aucun `FOAM FATAL`, défaut de décomposition ou échec de post-traitement n’apparaît.

Statut : **validé OF13 — ébullition pariétale IATE jusqu’à `End=4 s`, avec reconstruction et post-traitements complets**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037) et `BaseSolver.import_reference_asset`; aucune nouvelle API publique n’a été ajoutée.
