# Écosystème existant — Benchmark

## Objectif

Éviter de réimplémenter des briques existantes. Déterminer quelles parties développer vs réutiliser pour le module `foampilot.urban`.

## Méthode

Pour chaque outil/capacité :
- URL
- Licence
- Dernière activité
- Python / Linux
- Input / Output
- Fonction
- Performance
- Qualité
- Décision : **USE / WRAP / BUILD / IGNORE**

## Résumé des décisions

| Domaine | Outil | Décision |
|---------|-------|----------|
| LiDAR | `lazrs` + `laspy` | **WRAP** |
| LiDAR | `PDAL` | **WRAP** (pipelines complexes) |
| GIS | `shapely` | **USE** |
| GIS | `geopandas` | **USE** |
| GIS | `pyproj` | **USE** |
| GIS | `rasterio` | **USE** |
| OSM | `osmnx` | **WRAP** |
| CityGML | `citygml4py` / `pycitygml` | **IGNORE** (pas de besoin immédiat) |
| Mesh I/O | `meshio` | **IGNORE** (pas besoin pour l'instant) |
| Mesh | `pygalmesh` | **IGNORE** (Gmsh suffit pour MVP) |
| Mesh | `trimesh` | **IGNORE** |
| Gmsh | API Python | **USE** (via `foampilot.mesh.gmsh_mesher`) |
| OpenFOAM | `foampilot` | **USE** |
| Unités | `pint` / `manageunits.py` | **USE** |

## Détails par fichier

- `lidar.md` — outils LiDAR
- `gis.md` — outils GIS
- `building_reconstruction.md` — reconstruction bâtiments
- `city_models.md` — CityGML / modèles urbains
- `urban_cfd.md` — outils CFD urbains
- `meshing.md` — outils maillage
- `decisions.md` — décisions finales
