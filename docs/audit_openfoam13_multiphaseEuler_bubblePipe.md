# Audit OF13 — multiphaseEuler/bubblePipe

L’Allrun OpenFOAM 13 exécute `blockMesh`, `createZones`, `decomposePar`, `foamRun -parallel`, puis `reconstructPar`. Le cas représente une conduite à bulles avec phases `water`, `air1`, `air2` et une population balance `bubbles`. Les dictionnaires conservent les fractions initiales, les distributions de groupes de taille `f*.air1/air2`, les modèles de diamètre et de transport, ainsi que les forces de lift, wall lubrication et dispersion turbulente appliquées à l’eau. Les fonctions de validation calculent les résidus, les forces de phase, les fractions, la distribution de taille à l’injection et à l’outlet, et les densités de probabilité en diamètre. Le contrôle impose `endTime=4`, `deltaT=1e-4` et `writeInterval=0.5`.

Le runner `221_multiphaseEuler_bubblePipe/run.py` importe par FoamPilot tous les champs multi-phase et groupes de taille, les dictionnaires `constant/system` et les fonctions de validation, puis reproduit exactement `blockMesh`, `createZones`, `decomposePar` simple à 4 domaines, `foamRun -parallel` à 4 processus et `reconstructPar`. L’extrusion et `setFields` ne sont pas ajoutés car absents de l’Allrun OF13. L’option `-latestTime` a également été supprimée de la reconstruction pour respecter la référence.

La validation du maillage, de la zone outlet et de la décomposition réussit. Le calcul parallèle active la population balance `bubbles`, les groupes de taille, le calcul des diamètres moyens de Sauter et les quatre phases. Les fractions `air1`, `air2` et `water` restent bornées; la somme des fractions reste à 1 à la précision numérique. Le Courant maximal observé reste proche de `0.80`. Le plafond de 300 secondes interrompt la progression autour de `Time≈3.366 s` sur `4 s`; la reconstruction finale n’a pas démarré. Aucun `FOAM FATAL`, défaut de population balance, erreur MPI ou instabilité n’est observé.

Statut : **accepté avec réserve — chaîne multi-phase/population balance stable, mais `End=4 s` et `reconstructPar` restent hors budget de validation**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
