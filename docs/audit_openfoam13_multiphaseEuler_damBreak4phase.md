# Audit OF13 — multiphaseEuler/damBreak4phase

La référence OpenFOAM 13 standard exécute `blockMesh`, `setFields`, puis `foamRun`. Elle propose aussi une variante `AllrunFine` séparée utilisant `fineBlockMeshDict`, décomposition et calcul MPI; le runner présent couvre la variante standard correspondant à l’entrée `damBreak4phase` de la matrice. Le cas standard modélise une rupture de barrage à quatre phases `water`, `oil`, `mercury` et `air`. Les champs de fractions contiennent les conditions interfaciales entre les phases; la fonction de validation `phaseMap` est importée. Le contrôle impose `endTime=6`, `deltaT=1e-4`, `writeInterval=0.02`, `maxCo=0.5`, `maxAlphaCo=0.5` et `maxDeltaT=1`.

Le runner `222_multiphaseEuler_damBreak4phase/run.py` importe par FoamPilot tous les champs suffixés et dictionnaires `constant/system`, y compris les propriétés des quatre phases, les transferts de quantité de mouvement, les conditions de tension de surface, les contraintes de pression et la fonction de phase map, puis reproduit exactement la chaîne standard `blockMesh → setFields → foamRun` sous environnement OF13 explicite. Aucune décomposition ou variante fine n’est ajoutée à cette entrée.

La validation est complète. `blockMesh` et `setFields` créent le domaine et initialisent successivement les zones `water`, `oil` et `mercury` avec les quatre fractions. `foamRun` atteint `Time=6 s` et `End` en environ 141 secondes. Les fractions `water/oil/mercury/air` restent bornées, le Courant maximal observé est proche de `0.48`, les corrections MULES terminent normalement et aucun `FOAM FATAL`, défaut de phase ou problème de stabilité n’est observé.

Statut : **validé OF13 — rupture de barrage à quatre phases jusqu’à `End=6 s`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire. La variante `AllrunFine` devra être suivie séparément uniquement si elle est distinguée dans la matrice.
