# Audit OF13 — XiFluid/kivaTest

Le cas `XiFluid/kivaTest` est un tutoriel OpenFOAM 13 découvert lors de la comparaison complète des chemins non suivis. Son Allrun exécute `kivaToFoam -file otape17`, puis `foamRun`. La référence démarre à `-180 CAD` et vise `60 CAD`, avec dynamique moteur définie par `dynamicMeshDict`, ignition dans `constant/fvModels`, et fonctions de pas de temps dans `system/functions`.

Le runner `249_XiFluid_kivaTest/run.py` importe par FoamPilot l’asset KIVA racine `otape17`, les champs de démarrage du répertoire `-180/` vers leur véritable temps utilisateur, ainsi que les dictionnaires `constant/` et `system/`. Il exécute ensuite `kivaToFoam` et `foamRun` avec l’environnement OF13 explicite. L’import du temps `-180/` a été corrigé après vérification de l’ancrage de `import_reference_field`; les fichiers non standards sont importés par `import_reference_asset` au chemin exact attendu.

La validation dépasse le plafond de 300 secondes pendant le calcul, sans erreur fatale visible. Le calcul atteint environ `42,575 CAD` sur `60 CAD`; le Courant maximal reste proche de `0,029`, les erreurs de continuité restent faibles et diminuent, et les champs de moteur, combustion et maillage mobile sont traités sans `FOAM FATAL`, NaN ou divergence observée.

Statut : **accepté avec réserve — calcul XiFluid moteur stable jusqu’à `42,575/60 CAD` au plafond de temps**.

Aucune nouvelle API n’a été ajoutée; le runner utilise `BaseSolver.import_reference_asset` pour préserver le temps utilisateur non standard et `BaseSolver.run_command(environment=...)` pour l’environnement OF13.
