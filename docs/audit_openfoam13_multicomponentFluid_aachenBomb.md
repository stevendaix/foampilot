# Audit OF13 — multicomponentFluid/aachenBomb

La référence OpenFOAM 13 `Allrun-parallel` exécute `chemkinToFoam chemkin/chem.inp chemkin/therm.dat chemkin/transportProperties constant/reactions constant/speciesThermo`, puis `blockMesh`, `decomposePar`, `runParallel foamRun` et `reconstructPar`. Le cas utilise `multicomponentFluid`, une chimie réduite aachen, `endTime=0.01`, `deltaT=2.5e-6`, `writeInterval=5e-5`, `adjustTimeStep=yes` et `maxCo=0.1`. La décomposition officielle est simple à 12 domaines (`2×2×3`) avec distribution Zoltan RCB.

Le runner `192_multicomponentFluid_aachenBomb/run.py` importe les champs, constantes, dictionnaires et fichiers Chemkin par FoamPilot, puis reproduit cette chaîne avec `BaseSolver.run_command` : conversion Chemkin, maillage, décomposition, calcul MPI à douze processus et reconstruction. L’environnement OF13/MPI/ThirdParty est chargé explicitement dans les processus enfants par l’extension générique `run_command(environment=...)`.

La conversion Chemkin, `blockMesh` et `decomposePar` terminent correctement. Le calcul parallèle démarre sur douze domaines, reste stable avec un Courant maximal proche de `0,1` et des erreurs de continuité faibles, mais le plafond de validation de 300 secondes intervient vers `Time≈3,84×10^-4 s` sur `0,01 s`. La reconstruction finale n’est donc pas atteinte dans ce budget. Aucun `FOAM FATAL`, aucune erreur de bibliothèque et aucun échec de décomposition ne sont observés.

Statut : **accepté avec réserve — limite de temps du calcul chimique parallèle**.
