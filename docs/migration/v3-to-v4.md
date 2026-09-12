# Migration FoamPilot v3 → v4.0

## Objectif
Supprimer tous les shims et consolidons la structure `src/foampilot/` comme seul point d'entrée.

## Anciens vs Nouveaux Imports

| Ancien | Nouveau | Statut |
|--------|---------|--------|
| `foampilot.openfoam13` | `foampilot.core.physics.openfoam13` / `foampilot.workflows.urban` | ⏳ |
| `foampilot.model_addon` | `foampilot.workflows.medical.windkessel` | ⏳ |
| `foampilot.utilities` | `foampilot.core.*` (divers) | ⏳ |
| `foampilot.mesh` | `foampilot.core.meshing` | ⏳ |
| `foampilot.base.openFOAMFile` | `foampilot.core.base` | ⏳ |
| `foampilot.postprocess` | `foampilot.core.postprocessing` | ⏳ |
| `foampilot.report` | `foampilot.core.reporting` | ⏳ |

## Fichiers à migrer

### `openfoam13` (3 fichiers)
- `examples/urbanclimate/run.py`
- `foampilot/test/openfoam13/test_urbanclimate.py`
- `foampilot/test/openfoam13/test_physics.py`

### `utilities` (~30 fichiers)
- `examples/tobias_tutorial/*` (28 fichiers)
- `examples/thermoregulation/human_test.py`
- `examples/building_geo/generate_wind_cases.py`

### `mesh` (4 fichiers)
- `examples/marine_config/manoeuvring/run_simu.py`
- `examples/marine_config/propeller_mrf/run_simu.py`
- `examples/marine_config/dtc_moving/run_simu.py`
- `examples/openfoam13/build_realistic_dtc_foampilot.py`

### `base.openFOAMFile` (~20 fichiers)
- `foampilot/src/foampilot/core/reporting/simulation_report.py`
- `foampilot/src/foampilot/core/reporting/mesh_report.py`
- `foampilot/src/foampilot/system/*` (8 fichiers)
- `foampilot/src/foampilot/constant/*` (10 fichiers)
- `foampilot/test/test_openfoam14_features.py`