# Post 3 : Configuration solver & BCs inspirée de generate_wind_cases.py

## Pourquoi
`generate_wind_cases.py` est la référence pour les cas de vent urbains dans foampilot. On s'en est inspiré pour :
- le solver `incompressibleFluid`,
- le modèle de turbulence `kEpsilon`,
- le profil logarithmique à l'entrée.

## Détails
- **U inlet** : `codedFixedValue` avec profil log en z.
- **k inlet** : `codedFixedValue` basé sur l'intensité de turbulence.
- **epsilon inlet** : `codedFixedValue` issu de la théorie de la couche limite.
- **p outlet** : `fixedValue 0`.
- **top** : `symmetry`.
- **ground + bâtiments** : `wall`.
- **côtés** : `noFrictionWall`.

## Piège évité
`pressureOutlet` n'existe pas en tant que `polyPatch` dans OpenFOAM 13. On garde `patch` dans `boundary` et on applique `pressureInletOutletVelocity` dans `0/U` et `fixedValue 0` dans `0/p`.

## Code
`examples/building_geo/voxcity_vector_example.py` → `setup_openfoam_case()`
