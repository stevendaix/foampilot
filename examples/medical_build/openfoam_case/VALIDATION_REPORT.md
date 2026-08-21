# Validation OpenFOAM 13

Le cas est généré avec `./Allrun` après chargement de `/opt/openfoam13/etc/bashrc`.

Résultats validés le 21 août 2026 :

| Étape | Résultat |
|---|---|
| `surfaceCheck` | surface fermée, aucun triangle illégal |
| `blockMesh` | 21 504 cellules de fond |
| `snappyHexMesh` | terminé sans erreur |
| `checkMesh` | `Mesh OK` |
| Cellules finales | 231 628 |
| Points finaux | 308 222 |
| Patches | 11, dont `outer` et 10 patches anatomiques |
| Non-orthogonalité maximale | 69,4337 degrés |
| Skewness maximale | 2,11906 |

Les patches anatomiques sont `aorta_surface_inlet`, `aorta_surface_outlet_0`, `aorta_surface_outlet_1`, `aorta_surface_outlet_2`, `aorta_surface_outlet_3`, `aorta_surface_outlet_5`, `aorta_surface_outlet_6`, `aorta_surface_outlet_7`, `aorta_surface_outlet_8` et `aorta_surface_wall`.
