# Audit OF13 — shockFluid/diffuserIntake

La référence OpenFOAM 13 exécute `blockMesh`, `foamRun`, puis `foamPostProcess -latestTime -func graphFace(name=graph,start=(0 0 0),end=(0.5 0 0),fields=(Cf p),axis=x)`. Le cas compressible `shockFluid` possède les fonctions `MachNo` et `wallShearStress`, et vise `endTime=2e-3 s` avec pas `1e-7 s`.

Le runner `240_shockFluid_diffuserIntake/run.py` importe par FoamPilot les champs, dictionnaires, fonctions et propriétés physiques OF13, puis exécute la chaîne sérielle complète prévue, avec environnement OF13 explicite. Le post-traitement `graphFace` est passé comme dictionnaire FoamPilot, sans appel au script `createGraphs` de validation.

La validation atteint `Time≈1,34e-3 s` sur `2e-3 s` au plafond de 180 secondes. `blockMesh` termine, `foamRun` démarre correctement, et les fonctions `MachNo` et `wallShearStress` sont actives. Le Courant maximal observé reste proche de `0,254`; aucune divergence, valeur NaN, erreur fatale ou erreur de lecture n’est observée. Le post-traitement `graphFace` n’est pas atteint dans le budget.

Statut : **accepté avec réserve — calcul compressible stable jusqu’à `Time≈1,34e-3/2e-3 s`, post-traitement final hors budget**.

Le runner utilise `BaseSolver.run_command(environment=...)` et les fonctions d’import existantes; aucune nouvelle API n’a été ajoutée pour ce tutoriel.
