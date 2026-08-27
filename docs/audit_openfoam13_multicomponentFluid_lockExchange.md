# Audit OF13 — multicomponentFluid/lockExchange

L’Allrun OpenFOAM 13 exécute `blockMesh`, `setFields`, puis `foamRun`. Le cas est mono-région `multicomponentFluid` avec les espèces `water` et `sludge`, un champ de gravité et un échange gravitaire de type lock-exchange. Le dictionnaire `setFieldsDict` fixe par défaut `sludge=0`, `water=1`, puis initialise la zone box `(0 0 0)–(5 2 2)` avec `sludge=1` et `water=0`. Le contrôle impose `endTime=100`, `deltaT=0.05`, `writeInterval=1` et `maxDeltaT=1`.

Le runner `199_multicomponentFluid_lockExchange/run.py` importe les champs `.orig`, constantes et dictionnaires par FoamPilot, puis reproduit exactement la séquence de l’Allrun avec `BaseSolver.run_command` dans l’environnement OpenFOAM 13 explicite. Les propriétés physiques, les conditions aux limites et les schémas de transport multi-espèces sont conservés sans réécriture manuelle.

La validation est complète. `blockMesh` et `setFields` terminent correctement; les journaux confirment l’initialisation des champs `sludge` et `water`, y compris la zone `sludge`. `foamRun` atteint `Time=100 s` et `End` en environ 20 secondes. Le Courant maximal final est inférieur à `0.47`, les erreurs de continuité restent faibles et aucun `FOAM FATAL`, problème de champ ou erreur de zone n’est observé.

Statut : **validé OF13 — `End=100 s`, lock-exchange multi-espèces réussi**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037.
