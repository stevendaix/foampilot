# Building Reconstruction — Outils

## Candidates

| Outil | URL | Licence | Python | Linux | Décision |
|-------|-----|---------|--------|-------|----------|
| `osmnx` | https://github.com/gboeing/osmnx | MIT | ✅ | ✅ | **WRAP** |
| `citysim` | - | - | - | - | **IGNORE** |
| `py3dtiles` | https://github.com/py3dtiles/py3dtiles | BSD-3-Clause | ✅ | ✅ | **IGNORE** |

## Recommandation

- **OSM** : `osmnx` pour extraction bâtiments/routes
- **Reconstruction 3D** : pas d'outil Python mature retenu pour MVP
- **Approche retenue** : données IGN + simplification CFD dans `foampilot.urban`

## Notes

- `osmnx` permet d'extraire les polygones OSM, mais qualité très variable
- Pour CFD, mieux vaut BD TOPO / LiDAR HD IGN
