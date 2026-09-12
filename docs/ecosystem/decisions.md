# Décisions écosystème

## Résumé

| Domaine | Outil | Décision | Raison |
|---------|-------|----------|--------|
| LiDAR | `lazrs` + `laspy` | WRAP | Lecture LAS/LAZ native Python |
| LiDAR | `PDAL` | WRAP | Pipelines complexes, classification |
| GIS | `shapely` | USE | Validation, cleanup, simplification |
| GIS | `geopandas` | USE | Lecture SHP/GPKG, reprojection |
| GIS | `pyproj` | USE | CRS |
| GIS | `rasterio` | USE | MNT GeoTIFF |
| OSM | `osmnx` | WRAP | Extraction bâtiments/routes |
| CityGML | `citygml4py` / `pycitygml` | IGNORE | Pas de besoin immédiat |
| Mesh | `meshio` / `pygalmesh` / `trimesh` | IGNORE | Gmsh suffit pour MVP |
| Gmsh | API Python | USE | Via `foampilot.mesh.gmsh_mesher` |
| OpenFOAM | `foampilot` | USE | Déjà intégré |
| Unités | `pint` / `manageunits.py` | USE | Déjà dans foampilot |

## Non décidés / À réévaluer

- `PDAL` binding Python : nécessite install système, évaluer plus tard
- `osmnx` : qualité OSM variable, à utiliser avec précaution
- `snappyHexMesh` : backend alternatif, évaluer Phase 3+

## Règles

1. Ne pas ajouter de dépendance sans justification
2. Préférer `shapely` + `geopandas` pour GIS
3. Ne pas réinventer : utiliser `ValueWithUnit` pour les unités
4. Garder Gmsh comme backend principal pour Phase 1-2
