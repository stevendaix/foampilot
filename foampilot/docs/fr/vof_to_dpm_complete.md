# Cartographie complète du projet VOF–DPM

L’implémentation complète est regroupée dans `examples/openfoam13/vof_to_dpm/`.

| Domaine | Emplacement |
|---|---|
| Convertisseur Python | `src/foampilot/utilities/vof_to_dpm.py` |
| Tests Python | `test/test_vof_to_dpm.py` |
| Exemple pédagogique | `examples/course_vof_to_dpm.py` |
| Générateur PDF | `examples/generate_vof_to_dpm_technical_note.py` |
| Extracteur C++ offline | `examples/openfoam13/vof_to_dpm/applications/vofToDpm/` |
| Pont incompressible | `examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds/` |
| Pont compressible | `examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds/` |
| Sources statisticalDPMFoam | `examples/openfoam13/vof_to_dpm/statisticalDPMFoam/` |
| Tests OpenFOAM 13 | `examples/openfoam13/vof_to_dpm/test/openfoam13/` |
| Note technique et bibliographie | `docs/fr/vof_to_dpm_technical_note.pdf`, `docs/fr/vof_to_dpm.bib` |

Les cas exécutables principaux sont `vofToDpmSingleCell`, `vofToDpmParcelInBox`, `incompressibleVoFCloudsDamBreak` et `compressibleVoFCloudsDamBreak`. Commencer par le [guide d’installation et d’exécution](vof_to_dpm_openfoam13.md), puis exécuter les tests Python avant de compiler les composants OpenFOAM.

Les versions anglaise et chinoise sont disponibles dans `docs/en/vof_to_dpm_openfoam13.md` et `docs/zh/vof_to_dpm_openfoam13.md`.
