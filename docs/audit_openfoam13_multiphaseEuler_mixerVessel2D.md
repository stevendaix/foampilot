# Audit OF13 — multiphaseEuler/mixerVessel2D

La référence OpenFOAM 13 est un mélangeur 2D multiphasique avec phases `air`, `water`, `oil` et `mercury`. Elle construit un maillage depuis la ressource partagée `resources/blockMesh/mixerVessel2D`, crée les baffles, les sépare, crée les couples non conformes `nonCouple1` et `nonCouple2`, puis exécute `foamRun` jusqu’à `endTime=5`. La zone `rotor` et le mouvement solide associé sont conservés dans `constant/dynamicMeshDict`; les fonctions de suivi incluent `phaseMap`.

Le runner `227_multiphaseEuler_mixerVessel2D/run.py` importe par FoamPilot tous les champs `0/`, les dictionnaires `constant/system` et la ressource partagée de maillage. Il reproduit ensuite exactement `blockMesh -dict system/mixerVessel2D`, `createBaffles`, `splitBaffles`, `createNonConformalCouples nonCouple1 nonCouple2` et `foamRun`, sous environnement OF13 explicite. Lors de la première exécution, le passage littéral de `$FOAM_TUTORIALS/...` à travers le quoting de la commande a été identifié; la solution généralisable est d’importer la ressource par `solver.import_reference_asset` dans le cas et d’utiliser un chemin local stable, sans logique shell additionnelle.

La validation finale est complète. `blockMesh`, `createBaffles`, `splitBaffles` et `createNonConformalCouples` terminent avec succès. `foamRun` atteint `Time=5 s` et `End` en environ 223 secondes. Les quatre fractions restent bornées et proches des valeurs de mélange, le Courant maximal observé est proche de `0.389`, les erreurs de conservation de volume sont de l’ordre de la précision machine et le flux des couples non conformes est nul. Aucun `FOAM FATAL`, défaut de maillage mobile ou problème de couple n’apparaît.

Statut : **validé OF13 — mélangeur multiphasique 2D avec rotor et couples non conformes jusqu’à `End=5 s`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucune nouvelle API n’a été ajoutée. La correction de ressource partagée est réalisée par l’API existante `BaseSolver.import_reference_asset`.
