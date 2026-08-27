# Audit OF13 — incompressibleFluid/pitzDailySteadyMappedToRefined

La référence OpenFOAM 13 est un cas multi-étapes conçu pour tester l’interpolation entre maillages. L’Allrun prépare et exécute d’abord le cas source `incompressibleFluid/pitzDailySteady` en parallèle, construit ensuite un maillage cible légèrement raffiné à partir de `resources/blockMesh/pitzDaily`, décompose ce maillage, exécute `mapFieldsPar -consistent -sourceTime latestTime`, relance `foamRun`, puis reconstruit avec `reconstructPar -withZero`.

Le runner `250_incompressibleFluid_pitzDailySteadyMappedToRefined/run.py` remplace les copies/fusions shell par des imports FoamPilot du cas source et des dictionnaires, importe la ressource de maillage officielle, applique les transformations de raffinement et de décalage via `SystemDirectory.replace_file_text`, puis conserve les étapes de décomposition, mapping et reconstruction. L’environnement OF13 est explicite, notamment `PATH`, `LD_LIBRARY_PATH`, Scotch ThirdParty, `FOAM_MPI`, `WM_MPLIB` et `MPI_BUFFER_SIZE=20000000`.

La validation sous OpenFOAM 13 démarre correctement le cas source : `blockMesh`, `decomposePar` et `foamRun` parallèle terminent jusqu’à `End=300 s`, sans `FOAM FATAL` dans le calcul source. Le plafond d’exécution de 300 secondes est toutefois atteint à ce stade; le maillage cible, `mapFieldsPar`, le second calcul et `reconstructPar -withZero` n’ont donc pas encore été exécutés dans cette validation.

Statut : **validation partielle — source OF13 validée jusqu’à `End=300 s`; cible et mapping à compléter, réserve de temps explicite**.

Aucune nouvelle API n’a été ajoutée dans cette étape; les transformations utilisent les méthodes FoamPilot existantes `import_reference_dict`, `import_reference_file`, `replace_file_text`, `update_dictionary_entries` et `run_command`.
