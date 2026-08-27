# Audit OF13 — multicomponentFluid/SandiaD_LTS

L’Allrun OpenFOAM 13 exécute `chemkinToFoam` avec les fichiers GRI30, puis `blockMesh` et `setFields`. Il lance ensuite une phase de développement sans chimie : `startTime=0`, `writeInterval=1500`, `endTime=1500` et `chemistry=off`, suivie d’une phase avec chimie : `startTime=1500`, `writeInterval=100`, `endTime=5000` et `chemistry=on`. Une dernière invocation `foamRun` est utilisée avec l’option de journalisation de la référence. Le cas est LTS avec `localEuler`, un modèle `multicomponentFluid`, les espèces GRI et les entrées `inletCH4`, `inletPilot` et `inletAir`.

Le runner `191_multicomponentFluid_SandiaD_LTS/run.py` importe les champs `.orig`, constantes, dictionnaires système et fichiers Chemkin par FoamPilot. Il reproduit les conversions et les deux phases de l’Allrun avec `BaseSolver.run_command`, en modifiant les entrées ciblées de `system/controlDict` et `constant/chemistryProperties` par `foamDictionary`. L’environnement OF13 est chargé explicitement dans les processus enfants pour rendre l’exécution indépendante de l’état du shell hôte.

La préparation est validée : `chemkinToFoam`, `blockMesh` et `setFields` terminent correctement. La phase sans chimie atteint `End` à `Time=1500 s` en environ 99 secondes. La phase avec chimie reste stable, avec des erreurs de continuité bornées et sans `FOAM FATAL`; elle atteint `Time≈2869 s` sur `5000 s` lorsque le plafond de validation de 300 secondes intervient. La phase chimique n’atteint donc pas son terme dans le budget de validation et la dernière invocation/reconstruction de la référence n’est pas exécutée dans ce run.

Statut : **accepté avec réserve — limite de temps pendant la phase chimique**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037. Aucune nouvelle extension d’API spécifique au cas n’est nécessaire.
