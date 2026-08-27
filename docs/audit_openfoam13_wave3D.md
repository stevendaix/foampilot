# Audit OF13 — incompressibleVoF/wave3D

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/incompressibleVoF/wave3D`.

La référence contient une séquence de prétraitement de vague analogue à `wave` : `blockMesh`, `extrudeMesh`, `refineMesh`, `setWaves`, `decomposePar`, calcul parallèle `foamRun`, puis `reconstructPar`. Le cas est tridimensionnel avec patches `frontAndBack`, `inlet`, `inletSide`, `outletSide` et `outlet`.

Les paramètres relevés sont `solver incompressibleVoF`, `deltaT 0.05`, `writeInterval 1`, `adjustTimeStep no`, `maxCo 1`, `maxAlphaCo 1`, `maxDeltaT 1`, et `numberOfSubdomains 18` avec décomposition hiérarchique. `constant/waveProperties` décrit une vague Airy; `system/setWavesDict` cible `alpha.water`. Les champs `U.orig` et `alpha.water.orig` utilisent les conditions `waveVelocity` et `waveAlpha` de `libwaves.so` sur `(inlet|inletSide)`. Le cas utilise également `system/functions` avec une iso-surface de `alpha.water`.

Runner créé : `foampilot/tutorials/146_incompressibleVoF_wave3D/run.py`, avec `blockMesh`, `refineMesh`, `setWaves`, `decomposePar`, `mpirun -np 18 foamRun -solver incompressibleVoF -parallel` et `reconstructPar -newTimes`. L’Allrun officielle ne contient pas `extrudeMesh`; une première version erronée a été corrigée avant la relance. La validation corrigée reproduit `blockMesh` à 353 440 cellules puis les zones de `refineMesh` jusqu’à 1 024 024 cellules, mais le processus reçoit `SIGTERM` du sandbox avant la fin du raffinement, sans `FOAM FATAL` dans le journal. `setWaves`, le calcul et la reconstruction ne sont donc pas validés dans cet environnement pour cet ordre. API utilisées : `BaseSolver.import_reference_asset`, `CaseFieldsManager.import_reference_field`, `BaseSolver.run_command`; aucune nouvelle API.
