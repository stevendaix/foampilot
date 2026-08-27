# Audit OF13 — legacy/basic/laplacianFoam/flange

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/basic/laplacianFoam/flange`.

L’Allrun officielle ne contient pas de `blockMeshDict`. Elle importe le maillage fourni `flange.ans` avec `ansysToFoam flange.ans -scale 0.001`, lance `laplacianFoam`, puis exporte les résultats avec `foamToEnsight`, `foamToEnsightParts` et `foamToVTK`.

La mise en données comprend le champ scalaire thermique `T`, `constant/physicalProperties` avec `DT = 4e-05 m²/s`, ainsi que les dictionnaires `controlDict`, `fvSchemes` et `fvSolution`. Le champ `T` est initialisé à `273 K`; les patches imposés utilisent `273 K` et `573 K`, avec deux patches à gradient nul. Le contrôle officiel est `endTime 3`, `deltaT 0.005`, `writeControl runTime` et `writeInterval 0.1`.

Le runner FoamPilot `151_legacy_laplacianFoam_flange/run.py` importe tous les fichiers de référence, convertit le maillage Ansys avec `run_command`, puis exécute les mêmes solveur et exporteurs. La validation OF13 atteint `End=3 s`; les sorties `Ensight/Ensight.case` et `VTK` sont générées, sans `FOAM FATAL`. Aucune nouvelle fonction d’API n’est nécessaire : `BaseSolver.import_reference_asset`, `CaseFieldsManager.import_reference_field` et `BaseSolver.run_command` suffisent.
