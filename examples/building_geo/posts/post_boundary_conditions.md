# Conditions aux limites pour CFD urbaine avec foampilot

## Objectif
Disposer d'une configuration de BCs réaliste pour des simulations de vent autour de bâtiments, directement inspirée de `generate_wind_cases.py`.

## Configuration retenue
- **Solver** : `incompressibleFluid` (OpenFOAM 13)
- **Turbulence** : `kEpsilon`
- **Profil d'entrée** : logarithmique en `z`, codé en `codedFixedValue` pour `U`, `k`, `epsilon`
- **Sortie** : `pressureInletOutletVelocity` + `fixedValue 0` pour `p`
- **Parois** : `wall` pour bâtiments et sol, `noFrictionWall` pour les côtés
- **Top** : `symmetry`

## Points de vigilance
- `pressureOutlet` n'est pas un type de `polyPatch` valide dans OpenFOAM 13. On garde `patch` dans `constant/polyMesh/boundary`, et on applique le type de condition aux limites dans les fichiers de champ (`0/p`, `0/U`, etc.).
- `symmetryPlane` dans `boundary` → BC `symmetry` dans les champs.
- Dimensions de `k` et `epsilon` vérifiées via `checkMesh` et les erreurs OpenFOAM.

## Code
Voir `examples/building_geo/voxcity_vector_example.py` → `setup_openfoam_case()`.
