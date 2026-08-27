# Audit OF13 — multiphaseEuler/titaniaSynthesis

La référence OpenFOAM 13 exécute `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/titaniaSynthesis`, `decomposePar` à 4 domaines, `foamRun` en parallèle et `reconstructPar`, puis les graphes de validation. Le runner FoamPilot importe la ressource de maillage dans `system/titaniaSynthesis`, ainsi que tous les champs, dictionnaires et fichiers de validation référencés par les conditions aux limites.

Le cas comprend les phases `particles` et `vapour`, une population balance `aggregates` avec 29 groupes de tailles et une géométrie fractale (`Df=1.8`), la coalescence DahnekeInterpolation et une phase vapeur réactive. Les espèces vapeur incluent `O2`, `TiCl4`, `TiO2` et `Cl2`; `reactionDrivenPhaseChange` produit la phase particulaire TiO2. Les champs de composition, température, fractions, tailles et conductivités de réaction sont conservés.

La première validation a révélé que la ressource de maillage partagée devait être importée explicitement dans le chemin local `system/titaniaSynthesis`. Elle a également révélé que `decomposePar` dépend d’une table externe `validation/exptData/wallTemperature`; l’import FoamPilot a été généralisé à l’arborescence `validation` pour préserver les données de référence.

La seconde validation passe `blockMesh`, la décomposition à 4 domaines et démarre correctement `foamRun -parallel`. Le calcul progresse jusqu’à `Time≈9,583 s` sur `10 s` au plafond de 300 secondes. Les fractions `particles/vapour` restent bornées avec une somme de volume égale à 1, la population balance `aggregates` et le champ réactif `TiO2.vapour` sont résolus, le Courant maximal observé est proche de `0,278` et aucun `FOAM FATAL`, défaut MPI ou problème de table externe n’apparaît. La reconstruction n’est pas atteinte dans le budget de validation.

Statut : **accepté avec réserve — calcul réactif et population balance stable jusqu’à `Time≈9,583 s` sur `10 s`, reconstruction hors budget**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037). L’import de données de validation est une généralisation locale du mécanisme existant d’import de ressources et ne nécessite pas de nouvelle API publique.
