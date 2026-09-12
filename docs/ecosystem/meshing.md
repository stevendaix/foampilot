# Meshing — Outils

## Candidates

| Outil | URL | Licence | Python | Linux | Décision |
|-------|-----|---------|--------|-------|----------|
| `Gmsh` | https://gitlab.onelab.info/gmsh/gmsh | GPL | ✅ | ✅ | **USE** |
| `meshio` | https://github.com/nschloe/meshio | MIT | ✅ | ✅ | **IGNORE** |
| `pygalmesh` | https://github.com/nschloe/pygalmesh | MIT | ✅ | ✅ | **IGNORE** |
| `trimesh` | https://github.com/mikedh/trimesh | MIT | ✅ | ✅ | **IGNORE** |
| `snappyHexMesh` | OpenFOAM natif | GPL | ❌ | ✅ | **WRAP** |

## Recommandation

- **Maillage MVP** : Gmsh TetGen via `GmshMesher` existant
- **Maillage avancé** : `snappyHexMesh` plus tard
- **Pas d'outil mesh Python intermédiaire** pour l'instant

## Notes

- Gmsh est déjà utilisé dans foampilot.
- `meshio` / `pygalmesh` / `trimesh` peuvent être réévalués si besoin d'interopérabilité.
