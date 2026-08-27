# Audit OF13 — multiphaseEuler/mixerVessel2DMRF

La référence OpenFOAM 13 est la variante MRF du mélangeur multiphasique 2D. Elle conserve les phases `air`, `water`, `oil` et `mercury`, la zone `rotor`, le maillage partagé `resources/blockMesh/mixerVessel2D` et les fonctions `phaseMap`. Son Allrun exécute uniquement `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/mixerVessel2D`, puis `foamRun`; contrairement à `mixerVessel2D`, elle n’exécute ni baffles ni couples non conformes. Le dictionnaire `constant/MRFProperties` impose une vitesse de rotation de `60 rpm` autour de la zone `rotor`.

Le runner `228_multiphaseEuler_mixerVessel2DMRF/run.py` importe par FoamPilot tous les champs `0/`, les dictionnaires `constant/system` et la ressource partagée de maillage dans `system/mixerVessel2D`, puis exécute `blockMesh` et `foamRun` sous environnement OF13 explicite. L’import local de la ressource évite le passage littéral de `$FOAM_TUTORIALS` dans un argument shell quoté et reste entièrement réalisé par FoamPilot.

La validation est complète. `blockMesh` et `foamRun` terminent avec succès. Le calcul atteint `Time=5 s` et `End` en environ 120 secondes. Les fractions `water/oil/mercury/air` restent bornées, le Courant maximal observé est proche de `0.376`, et les résidus de phase sont traités normalement avec le MRF à `60 rpm`. Aucun `FOAM FATAL`, défaut de maillage ou erreur de rotation n’apparaît.

Statut : **validé OF13 — mélangeur multiphasique MRF jusqu’à `End=5 s`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucune nouvelle API n’a été ajoutée. La ressource partagée est importée via `BaseSolver.import_reference_asset`.
