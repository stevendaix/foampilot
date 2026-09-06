# Audit OF13 — multicomponentFluid/simplifiedSiwek

Le tutoriel OpenFOAM 13 ne fournit pas d’Allrun de calcul explicite. Sa mise en données est un cas `multicomponentFluid` avec `blockMesh` puis `foamRun`, incluant chimie simplifiée méthane/hydrogène, transfert radiatif et deux nuages réactifs `coalCloud` et `limestoneCloud`. Les dictionnaires `coalCloudProperties`, `limestoneCloudProperties`, les fichiers de positions, `radiationProperties`, `fvModels` et les champs multi-espèces constituent la mise en données de référence. Le contrôle impose `endTime=0.5`, `deltaT=1e-4`, `writeInterval=0.0025`, ajustement de pas et `maxDeltaT=1`.

Le runner `203_multicomponentFluid_simplifiedSiwek/run.py` importe intégralement par FoamPilot les champs, constantes, dictionnaires, positions de particules et propriétés de radiation, puis exécute `blockMesh` et `foamRun` sous l’environnement OF13 explicite. Aucune configuration de cloud, de radiation ou de réaction n’est réécrite manuellement.

La validation est complète. `blockMesh` termine correctement. `foamRun` atteint `Time=0.5 s` et `End` en environ 19 secondes. Les journaux confirment la résolution des clouds 2D `coalCloud` et `limestoneCloud`, avec respectivement 27 et 18 parcels courants, ainsi que les transferts thermiques et la réaction de surface du charbon. Le Courant maximal reste inférieur à `0.044`, les erreurs de continuité restent faibles et aucun `FOAM FATAL`, problème de cloud ou erreur de radiation n’est observé.

Statut : **validé OF13 — clouds coal/limestone, radiation et calcul jusqu’à `End=0,5 s`**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037; aucun changement d’API supplémentaire n’a été nécessaire.
