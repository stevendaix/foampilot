# VoxCity vers OpenFOAM : pipeline vectorielle sans STL

## Résumé
On a remplacé la voie STL/snappyHexMesh par une génération directe du maillage Gmsh à partir des données vectorielles VoxCity (bâtiments + DEM), puis un export OpenFOAM par `DirectOpenFOAMExporter`.

## Pourquoi cette voie ?
- Plus rapide que le téléchargement + export STL + snappy
- Pas de tolérance/feature-angle à régler
- Maillage tétraédrique valide directement dans `constant/polyMesh`

## Ce qui a été fait
### Backend foampilot
- `foampilot/urban/model/terrain.py` : correction de `CFDTerrain.get_elevation` pour 2/3 points et garde-fous NaN.
- `foampilot/mesh/snappymesh.py` : support multi-STL.
- `foampilot/openfoam/snappy_case_builder.py` : méthodes complétées.
- `foampilot/urban/geometry/building_extruder.py` : bug PyVista sur `faces` corrigé.

### Pipeline VoxCity
- Lecture VoxCity via `VoxCityReader` → `UrbanModel` + `CFDTerrain`.
- Construction Gmsh : domaine fluide + bâtiments, suppression des volumes bâtiments, assignation des patches.
- Export direct OpenFOAM : `DirectOpenFOAMExporter`.

## Résultats
- Cas synthétique 3 bâtiments : `checkMesh` OK.
- Cas VoxCity réel 16 bâtiments : `checkMesh` OK, calcul `foamRun -solver incompressibleFluid` convergé.

## Fichiers clés
- `examples/building_geo/voxcity_vector_example.py`
- `examples/building_geo/voxcity_export_work/src/vector_builder.py`
- `examples/building_geo/voxcity_export_work/src/read_voxcity.py`

## Prochaines étapes
- Tester la voie OBJ.
- Benchmark vs snappyHexMesh.
- Nettoyer les BCs par défaut.
