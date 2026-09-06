# Audit OF13 — multiphaseEuler/titaniaSynthesisSurface

La référence OpenFOAM 13 reprend la chaîne de `titaniaSynthesis` : `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/titaniaSynthesis`, `decomposePar` à 4 domaines, `foamRun` en parallèle et `reconstructPar`, suivis des graphes de validation. Le runner FoamPilot importe la ressource de maillage partagée, les champs, tous les dictionnaires `constant/system` et les données `validation` référencées par les conditions aux limites.

Le cas conserve les phases `particles` et `vapour`, la population balance `aggregates` et les 29 groupes fractals. Il ajoute une seconde réaction de croissance de surface : `surfaceReactionDrivenPhaseChange` avec production de `TiO2_s`, en complément de `reactionDrivenPhaseChange` pour la formation volumique de `TiO2`. Les espèces `O2`, `TiCl4`, `TiO2`, `TiO2_s` et `Cl2`, les fonctions `graphCell`, `populationBalanceSizeDistribution` et `writeObjects` sur les deux taux de changement de phase sont conservées.

Le runner `232_multiphaseEuler_titaniaSynthesisSurface/run.py` reproduit uniquement les commandes FoamPilot de la chaîne de référence, avec environnement OF13 explicite et 4 processus MPI. Le maillage partagé est importé localement dans `system/titaniaSynthesis`; l’import des données `validation/` permet à `decomposePar` et aux conditions aux limites de retrouver les tables externes nécessaires.

La validation passe `blockMesh`, la décomposition et le démarrage du calcul réactif parallèle. Elle progresse jusqu’à environ `Time=7,5 s` sur `10 s` au plafond de 300 secondes. Les fractions `particles/vapour` restent normalisées, la population balance résout les 29 groupes, le diamètre moyen de Sauter évolue et le champ de croissance `TiO2_s.vapour` est calculé. Les températures restent dans la plage observée `296–1382 K`. Des messages récurrents `solution singularity` apparaissent sur `N2.vapour`, mais aucun `FOAM FATAL`, arrêt MPI ou divergence globale n’est observé. La reconstruction n’est pas atteinte dans le budget.

Statut : **accepté avec réserve — calcul réactif de surface stable jusqu’à environ `7,5/10 s`, avec singularités locales répétées sur `N2.vapour` et reconstruction hors budget**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucune nouvelle API publique n’a été ajoutée.
