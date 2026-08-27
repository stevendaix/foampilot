# Audit OF13 — multiphaseEuler/steamInjection

Le dossier OpenFOAM 13 `multiphaseEuler/steamInjection` ne contient pas d’`Allrun` ni de script de lancement à la racine. Sa mise en données fournit toutefois une chaîne minimale déterministe `blockMesh → foamRun`, un `controlDict` avec `solver=multiphaseEuler`, `endTime=10`, `deltaT=1e-3`, adaptation de pas avec `maxCo=0.25` et `maxDeltaT=1e-2`, ainsi que les dictionnaires complets de transfert de chaleur et changement de phase.

Le cas comprend les phases `steam` et `water`. La vapeur utilise un diamètre isotherme dépendant de la pression (`d0=3e-3`, `p0=1e5`), l’eau est la phase continue, la traînée est Schiller–Naumann et la masse virtuelle possède `Cvm=0.5`. Le modèle `heatTransferLimitedPhaseChange` est activé dans `constant/fvModels`; `fvConstraints` limite les températures des deux phases entre 270 et 2000 K. Les fonctions `cellMin(T.steam,T.water,p)` et `cellMax(T.steam,T.water,p)` sont conservées.

Le runner `230_multiphaseEuler_steamInjection/run.py` importe par FoamPilot tous les champs `0/` et tous les dictionnaires `constant/system`, puis exécute `blockMesh` et `foamRun` sous environnement OF13 explicite. L’absence d’Allrun est documentée plutôt que masquée; aucune étape non présente dans la mise en données n’a été ajoutée.

La validation est complète. `blockMesh` et `foamRun` atteignent `Time=10 s` et `End` en environ 91 secondes. Le changement de phase est effectivement actif; `mDot` reste positif et borné, avec une moyenne observée proche de `0.0667` et un maximum proche de `51.17`. Les fractions `steam/water` restent bornées, les températures observées restent dans les limites physiques et aucun `FOAM FATAL`, défaut de maillage ou divergence n’apparaît.

Statut : **validé OF13 — injection de vapeur dans l’eau jusqu’à `End=10 s`; absence d’Allrun explicitement documentée**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
