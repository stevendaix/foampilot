# Building Geo — Suivi de session

## Contexte
- Projet : `foampilot` — automatisation CFD urbain avec Gmsh + OpenFOAM
- Répertoire : `/home/steven/foampilot/examples/building_geo/`
- Module : `foampilot.urban` (`src/foampilot/urban/`)
- Scripts clés :
  - `generate_wind_cases.py` — génération multi-directions
  - `run_single_wind_case.py` — génération + exécution d'un cas
  - `osm_neighborhood_example.py` — test sur données réelles OSM
  - `wind_postprocess.py` — post-traitement avec OpenFOAMDirectReader

## Objectifs
1. Maillage Gmsh robuste pour quartiers urbains
2. BCs réalistes (profil logarithmique, kEpsilon par défaut)
3. Données réelles OSM + terrain
4. Post-processing complet (images, Cp, |U|)
5. Backend snappyHexMesh via VoxCity + terrain

## États

### ✅ Fait
- `.gitignore` créé (`cases/`, `post/`, `*.md`, pycache, logs)
- `wind_postprocess.py` fonctionne avec `OpenFOAMDirectReader`
- Bugfix `openfoam_direct.py` :
  - `_build_cells_from_faces` : bounds check sur `neighbour`
  - `OpenFOAMDirectReader._ensure_mesh_loaded` : `n_cells = max(owner, neighbour)`
  - `n_faces` déprécié → `n_cells` dans extraction patch
- Pipeline améliorée portée de `building_aero` vers `building_geo` :
  - `algorithm_3d=4`, `Mesh.Optimize`, `Mesh.OptimizeNetgen`
  - Patchs lowercase : `inlet`, `outlet`, `side_left`, `side_right`, `top`, `ground`
  - BCs : `pressureOutlet`, `noFrictionWall`, `symmetry`
  - `codedFixedValue` pour U, k, epsilon, omega (profil log)
  - `nNonOrthogonalCorrectors=2`, `purgeWrite=1`, relaxation factors
- Default turbulence model : `kOmegaSST` → `kEpsilon`
- Background mesh sizing pour faces hors buildings/ground
- `osm_neighborhood_example.py` amélioré :
  - `simplify_buildings()` (bounding boxes)
  - Terrain GeoTIFF support
  - `osmnx.projection.project_gdf` corrigé
  - `min_size` dans `MeshConfig`
- Plan VoxCity/snappy validé et intégré dans `plan_voxcity_integration.md`
- Nouveaux modules backend snappy créés (côté `foampilot/src` seulement) :
  - `foampilot/urban/snappy_config.py` : `TerrainConfig`, `BuildingConfig`, `DomainConfig`, `SnappyMeshConfig`
  - `foampilot/urban/readers/voxcity_reader.py` : lecture VoxCity → `UrbanModel` + `CFDTerrain`
  - `foampilot/urban/terrain/processor.py` : DEM → terrain STL fermé
  - `foampilot/urban/geometry/building_extruder.py` : footprints → bâtiments STL
  - `foampilot/openfoam/snappy_case_builder.py` : wiring STL + SnappyMesher + Solver
- Tests unitaires ajoutés :
  - `test_snappy_config.py`
  - `test_terrain_processor.py`
  - `test_building_extruder.py`
  - `test_snappy_case_builder.py`
- `pyproject.toml` corrigé : virgule manquante dans `dependencies`

### 🚧 En cours / Bloqué
- OSM Paris 50m : timeout Gmsh fragment avec footprints complexes
- Simplification OSM nécessaire mais pas encore testée dans pipeline complète
- Tests snappy : collection pytest fonctionne, reste à valider le run complet `blockMesh → snappyHexMesh`
- VoxCityReader dépend de `voxcity` + Google Earth Engine : pas encore installé/exécuté

### ❌ À faire
- Tester cas OSM complet (génération + maillage + simulation + post)
- Vérifier mesh quality sur cas réel
- Ajouter visualisations supplémentaires (Cp buildings, rose des vents)
- Implémenter lecture BD TOPO (Phase 3)
- Tester terrain réel sur cas OSM
- Finaliser backend snappy :
  - brancher `SnappyMesher` existant dans `SnappyCaseBuilder`
  - lancer `blockMesh` / `surfaceFeatures` / `snappyHexMesh`
  - valider `checkMesh` sur cas simple
  - intégrer `foampilot.Solver` pour la physique

## Notes
- `checkMesh` cas synthétique 25 bâtiments : 8803 cellules, 7 patches, convergence OK
- OSM reader nécessite `osmnx` + `rasterio` (tous deux installés)
- Backend snappy utilise `foampilot.mesh.snappymesh.SnappyMesher` existant
- `cases/` et `post/` sont ignorés par git
