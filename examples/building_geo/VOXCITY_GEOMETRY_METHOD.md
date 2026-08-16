# Méthode de reconstruction géométrique de VoxCity

## Vue d'ensemble

VoxCity (v1.6.2) est une bibliothèque Python de modélisation urbaine 3D basée sur des voxels. Sa pipeline géométrique est **unidirectionnelle** :

```
Sources de données (vectorielles ou raster)
    ↓
Grilles 2D (hauteur de bâtiments, DEM, occupation du sol, canopée)
    ↓
Grille de voxels 3D (codes sémantiques par cellule)
    ↓
Export HDF5 / simulation
```

Contrairement à une attente intuitive, VoxCity ne possède pas de fonction native de « reconstruction » de bâtiments à partir de la grille de voxels (inversion voxel → vecteur). Au lieu de cela, il préserve les données vectorielles originales dans les métadonnées (`extras['building_gdf']`) et encode les informations de hauteur par cellule dans des grilles 2D dédiées. Les consommateurs (ex. : foampilot) utilisent ces grilles et le GDF pour reconstruire la géométrie CFD.

---

## 1. Conversion des sources brutes en grilles de voxels

### 1.1 Sources de données supportées

Le module `voxcity.downloader` fournit des adaptateurs pour chaque source :

| Source | Module | Type |
|--------|--------|------|
| OpenStreetMap (OSM) | `voxcity/downloader/osm.py` | Vectoriel (GDF) |
| Overture | `voxcity/downloader/overture.py` | Vectoriel (GDF) |
| Microsoft Building Footprints | `voxcity/downloader/mbfp.py` | Vectoriel (GDF) |
| EUBUCCO v0.1 | `voxcity/downloader/eubucco.py` | Vectoriel (GDF) |
| Global Building Atlas (GBA) | `voxcity/downloader/gba.py` | Vectoriel (GDF) |
| Open Building 2.5D Temporal | `voxcity/downloader/gee.py` | Raster (GeoTIFF) |
| England 1m DSM-DTM | `voxcity/downloader/gee.py` | Raster (GeoTIFF) |
| ESA WorldCover / ESRI / Dynamic World | `voxcity/downloader/gee.py` | Raster (GeoTIFF) |

### 1.2 Pipeline d'assemblage

Le point d'entrée est `VoxCityPipeline.run()` dans `voxcity/generator/pipeline.py` :

```python
# voxcity/generator/pipeline.py:169
def run(self, cfg: PipelineConfig, building_gdf=None, terrain_gdf=None, **kwargs) -> VoxCity:
    # 1. Téléchargement parallèle (4 threads)
    land_cover_grid, bh, bmin, bid, building_gdf_out, canopy_top, canopy_bottom, dem, lc_src_effective = \
        self._run_parallel_downloads(...)

    # 2. Aplatissement du DEM sur les zones d'eau
    dem, water_dem_info = _flatten_water_dem_by_component(dem, land_cover_grid, ...)

    # 3. Voxelisation
    voxelizer = Voxelizer(voxel_size=cfg.meshsize, ...)
    vox = voxelizer.generate_combined(
        building_height_grid_ori=bh,
        building_min_height_grid_ori=bmin,
        building_id_grid_ori=bid,
        land_cover_grid_ori=land_cover_grid,
        dem_grid_ori=dem,
        tree_grid_ori=canopy_top,
        canopy_bottom_height_grid_ori=canopy_bottom,
    )

    # 4. Assemblage du objet VoxCity
    return self.assemble_voxcity(vox, bh, bmin, bid, land_cover_grid, dem, ...)
```

### 1.3 Téléchargement parallèle

Les 4 sources (occupation du sol, bâtiments, canopée, DEM) sont téléchargées en parallèle via `ThreadPoolExecutor` (`_run_parallel_downloads`, ligne 398 du pipeline). Chaque source utilise une stratégie :

- `LandCoverSourceStrategy` → `get_land_cover_grid()`
- `BuildingSourceStrategy` → `get_building_height_grid()`
- `CanopySourceStrategy` → `get_canopy_height_grid()`
- `DemSourceStrategy` → `get_dem_grid()`

---

## 2. Reconstruction des empreintes et hauteurs de bâtiments

VoxCity ne reconstruit pas les bâtiments à partir des voxels. Il préserve deux représentations parallèles :

### 2.1 Grilles 2D de bâtiments

La fonction `create_building_height_grid_from_gdf_polygon()` (`voxcity/geoprocessor/raster/buildings.py:62`) produit trois grilles 2D de même taille `(ny, nx)` :

| Grille | Type | Signification |
|--------|------|---------------|
| `building_height_grid` | `float64` | Hauteur maximale du bâtiment par cellule (m) |
| `building_min_height_grid` | `object` (liste par cellule) | Liste de segments `[min_height, max_height]` pour chaque bâtiment intersectant la cellule |
| `building_id_grid` | `float64` | Identifiant du bâtiment « gagnant » par cellule |

### 2.2 Deux modes de rasterisation

La fonction `_decide_auto_mode()` (ligne 106) choisit automatiquement entre deux algorithmes selon la densité et le chevauchement des bâtiments :

#### Mode rapide : `_process_with_rasterio`

```python
# voxcity/geoprocessor/raster/buildings.py:199
def _process_with_rasterio(filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, rectangle_vertices, complement_height):
    # 1. Construction d'une transformée Affine rasterio
    transform = Affine(du * u_vec[0], dv * v_vec[0], float(origin[0]),
                       du * u_vec[1], dv * v_vec[1], float(origin[1]))

    # 2. Rasterisation des hauteurs par bâtiment (last-wins)
    height_shapes = [(mapping(geom), height) for geom, height in zip(valid_buildings.geometry, valid_buildings['height'])]
    height_raster = features.rasterize(height_shapes, out_shape=(grid_size[1], grid_size[0]), transform=transform, ...)

    # 3. Rasterisation des IDs (last-wins)
    id_raster = features.rasterize(id_shapes, ...)

    # 4. Rasterisation des min_heights (last-wins)
    min_heights_raster = features.rasterize(min_height_shapes, ...)
```

#### Mode précis : `_process_with_geometry_intersection`

```python
# voxcity/geoprocessor/raster/buildings_precise.py:134
def _process_with_geometry_intersection(filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, complement_height):
    # 1. Collecte des polygones de bâtiments et de leurs bbox
    building_polygons = _collect_building_polygons(filtered_gdf, complement_height)

    # 2. Pour chaque cellule, calcul des bâtiments candidats via inversion de l'affine
    candidates = _candidate_cells_by_building(building_polygons, ...)

    # 3. Traitement cellule par cellule : intersection géométrique exacte
    for (i, j), cand_ks in candidates.items():
        cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
        for k in cand_ks:
            inter_area = cell.intersection(polygon).area
            if (inter_area / cell_area) > _CELL_INTERSECTION_THRESHOLD:  # 0.3
                # Le bâtiment couvre >30% de la cellule → on l'affecte
                building_min_height_grid[i, j].append([min_height, height])
                building_id_grid[i, j] = feature_id
```

Le seuil `_CELL_INTERSECTION_THRESHOLD = 0.3` signifie qu'un bâtiment doit couvrir au moins 30 % de la surface de la cellule pour y être écrit.

### 2.3 Stockage dans `building_gdf`

Le GeoDataFrame filtré et traité est stocké dans `extras['building_gdf']` sous forme de **GeoParquet compressé** dans le groupe HDF5 `voxcity/extras_gdf/` (`voxcity/io.py:538-565`).

Colonnes typiques du `building_gdf` (observé sur le fichier de démo) :

```
id, names, sources, level, height, min_height, is_underground,
num_floors, num_floors_underground, min_floor, subtype, class,
facade_color, facade_material, roof_material, roof_shape,
roof_direction, roof_orientation, roof_color, roof_height,
geometry, has_parts, version, bbox, building_id, is_inner, height_estimated
```

---

## 3. Méthode de traitement par chevauchement (`process_building_footprints_by_overlap`)

Fichier : `voxcity/geoprocessor/overlap.py:9`

```python
def process_building_footprints_by_overlap(filtered_gdf, overlap_threshold=0.5):
    """
    Merge overlapping buildings based on area overlap ratio, assigning the ID
    of the larger building to smaller overlapping ones.
    """
```

### Algorithme

1. **Projection** : les géométries sont projetées en `EPSG:3857` (mètres) pour calculer des surfaces cohérentes.
2. **Tri par aire** : les bâtiments sont triés par aire décroissante (`sort_values(by='area', ascending=False)`).
3. **Index spatial R-tree** : construction d'un index sur les bounding boxes des bâtiments valides.
4. **Itération** : pour chaque bâtiment `i` (du plus grand au plus petit) :
   - Recherche des bâtiments `j < i` dont la bbox intersecte celle de `i`.
   - Pour chaque candidat `j` :
     - Calcul de l'intersection géométrique.
     - Ratio de chevauchement : `overlap.area / current_area`.
     - Si `ratio > overlap_threshold` (défaut `0.5`) : le bâtiment `i` hérite de l'ID du bâtiment `j` (plus grand).
5. **Application** : les IDs modifiés sont reportés dans le GeoDataFrame original.

### Effet sur la voxelisation

En fusionnant les IDs des bâtiments qui se chevauquent à plus de 50 %, la rasterisation/rasterisation précise affecte tous les voxels de la zone chevauchée au bâtiment dominant (le plus grand). Cela évite les artefacts de « dernière écriture » (last-wins) lors de la rasterisation.

---

## 4. Construction du `building_gdf` et contenu

### 4.1 Origine

Le `building_gdf` est créé dans `get_building_height_grid()` (`voxcity/generator/grids.py:178`) et provient de l'une des sources suivantes :

- **Paramètre utilisateur** : `building_gdf` passé directement.
- **Téléchargement automatique** : `load_gdf_from_openstreetmap()`, `load_gdf_from_overture()`, `get_mbfp_gdf()`, `load_gdf_from_eubucco()`, `load_gdf_from_gba()`.

### 4.2 Enrichissement par source complémentaire

Si `building_complementary_source` est défini, VoxCity fusionne les hauteurs :

- **Extraction depuis GDF complémentaire** : `extract_building_heights_from_gdf()` (`voxcity/geoprocessor/heights.py:19`) — affecte la hauteur moyenne pondérée par intersection.
- **Complétion de footprints** : `complement_building_heights_from_gdf()` (`voxcity/geoprocessor/heights.py:96`) — ajoute les bâtiments présents uniquement dans la source complémentaire.
- **Extraction depuis GeoTIFF** : `extract_building_heights_from_geotiff()` (`voxcity/geoprocessor/heights.py:169`) — moyenne des pixels du raster sous le polygone.

### 4.3 Patch final

`VoxCityPipeline._patch_building_gdf()` (`pipeline.py:330`) remplace les hauteurs NaN ou nulles par `building_complement_height` (défaut `10 m`) et ajoute la colonne booléenne `height_estimated`.

---

## 5. Reconstruction du terrain (DEM)

### 5.1 Téléchargement / chargement

`get_dem_grid()` (`voxcity/generator/grids.py:441`) supporte :

- **Local file** : GeoTIFF passé via `dem_path`.
- **GSI DEM Japan** : téléchargement via `voxcity/downloader/gsi.py`.
- **Sources Google Earth Engine** : `USGS 3DEP 1m`, `England 1m DTM`, `DEM France 1m/5m`, `AUSTRALIA 5M DEM`, `Netherlands 0.5m DTM`.
- **Flat / None** : retourne une grille de zéros (`FlatDemStrategy`).

Le DEM est toujours interpolé sur la grille via `create_dem_grid_from_geotiff_polygon()`.

### 5.2 Aplatissement des zones d'eau

`_flatten_water_dem_by_component()` (`pipeline.py:28`) :

1. Détecte les cellules d'eau selon la classification d'occupation du sol (classe standard 9).
2. Utilise `scipy.ndimage.label` avec une connectivité 4 ou 8 pour identifier les corps d'eau connexes.
3. Pour chaque corps d'eau, remplace le DEM par la valeur minimale du DEM à l'intérieur du composant connexe.

```python
# pipeline.py:72-80
for water_body_id in range(1, water_body_count + 1):
    component_mask = labels == water_body_id
    finite_mask = component_mask & np.isfinite(dem_grid)
    water_min = float(np.min(dem_grid[finite_mask]))
    flattened[component_mask] = water_min
```

### 5.3 Intégration dans la grille de voxels

Dans `Voxelizer.generate_combined()` (`voxcity/generator/voxelizer.py:158`), le DEM est d'abord recentré :

```python
dem_grid = dem_grid_ori.copy() - np.min(dem_grid_ori)  # ancrage au sol à 0
dem_grid = process_grid(building_id_grid, dem_grid)      # lissage par bâtiment
```

Puis, pour chaque colonne `(i, j)` :

```python
ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + 1
voxel_grid[i, j, :ground_level] = GROUND_CODE  # -1
voxel_grid[i, j, ground_level - 1] = land_cover_grid[i, j]  # code d'occupation du sol
```

---

## 6. Génération de la grille de voxels 3D

Fichier : `voxcity/generator/voxelizer.py`

### 6.1 Codes sémantiques

| Code | Signification |
|------|---------------|
| `-1` | Sol / DEM |
| `-2` | Canopée arborée |
| `-3` | Bâtiment |
| `0` | Inoccupé (air) |
| `1..N` | Occupation du sol (selon source) |

### 6.2 Voxelisation des bâtiments (Numba JIT)

Les segments de bâtiments par cellule sont d'abord aplatis :

```python
# voxelizer.py:91
def _flatten_building_segments(building_min_height_grid, voxel_size):
    # Convertit la liste [[min_h, max_h], ...] par cellule en 4 arrays plats :
    # seg_starts, seg_ends, seg_offsets, seg_counts
```

Puis le noyau Numba remplit les voxels :

```python
# voxelizer.py:38
@jit(nopython=True, parallel=True)
def _voxelize_kernel(voxel_grid, land_cover_grid, dem_grid, tree_grid,
                     canopy_bottom_grid, has_canopy_bottom,
                     seg_starts, seg_ends, seg_offsets, seg_counts,
                     trunk_height_ratio, voxel_size):
    for i in prange(rows):
        for j in range(cols):
            ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + 1

            # Sol et occupation du sol
            voxel_grid[i, j, :ground_level] = GROUND_CODE
            voxel_grid[i, j, ground_level - 1] = land_cover_grid[i, j]

            # Arbres
            tree_height = tree_grid[i, j]
            if tree_height > 0.0:
                crown_base_level = int(crown_base_height / voxel_size + 0.5)
                crown_top_level = int(tree_height / voxel_size + 0.5)
                voxel_grid[i, j, tree_start:tree_end] = TREE_CODE

            # Bâtiments (segments multiples possibles par cellule)
            base = seg_offsets[i, j]
            count = seg_counts[i, j]
            for k in range(count):
                s = seg_starts[base + k]
                e = seg_offsets[base + k]
                start = ground_level + s
                end = ground_level + e
                voxel_grid[i, j, start:end] = BUILDING_CODE
```

Si Numba n'est pas disponible, une boucle Python pure (`voxelizer.py:219`) assure la même logique.

---

## 7. Structure du fichier HDF5

Fichier observé : `neighborhood_demo/output/voxcity.h5`

### Racine

```
Attributs :
  __format__      = 'voxcity_results.v3'
  axes            = 'north,east,up'
  rotation_angle  = 0.0
  crs             = 'EPSG:4326'
  meshsize        = 5.0
  bounds          = [2.3225, 48.8515, 2.3245, 48.8528]
Dataset :
  rectangle_vertices  shape=(4, 2) float64  # [SW, NW, NE, SE] en (lon, lat)
```

### Groupe `voxcity`

```
Datasets :
  voxel_grid       shape=(29, 29, 7)   int8    # codes voxels 3D
  building_height  shape=(29, 29)      float64 # hauteur max par cellule
  building_id      shape=(29, 29)      float64 # ID bâtiment par cellule
  dem              shape=(29, 29)      float64 # DEM par cellule
  land_cover       shape=(29, 29)      int64   # classe occupation du sol

Groupes :
  building_min_heights/
    offsets  shape=(842,) int64
    values   shape=(878,) float64
    attrs : n_cols=2, shape=[29, 29]

  canopy/
    top   shape=(29, 29) float64
    bottom shape=(29, 29) float64

  extras_gdf/
    building_gdf  dataset scalaire (bytes) → GeoParquet

  extras_np/
    canopy_bottom shape=(29, 29) float64
    canopy_top    shape=(29, 29) float64

Attribut :
  extras_json : JSON avec rectangle_vertices, sources, flatten_water_dem, etc.
```

### Désérialisation des `min_heights`

Les `building_min_height_grid` (tableau `object` de listes) sont sérialisés en 3 datasets plats (`voxcity/io.py:300-362`) :

- **offsets** : `int64`, longueur `ny*nx + 1` ; la cellule `i` possède `values[offsets[i] : offsets[i+1]]`.
- **values** : `float64`, tous les segments aplatis `[min, max, min, max, ...]`.
- **n_cols** : nombre de colonnes par tuple (généralement `2`).

---

## 8. Flux de données complet

```
┌─────────────────────────────────────────────────────────────────────┐
│  Sources brutes (OSM, Overture, Microsoft, EUBUCCO, GEE, ...)     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │  Téléchargement parallèle (x4) │
                │  LandCover | Building | Canopy | DEM │
                └───────────────┬────────────────┘
                                │
                ┌───────────────▼────────────────────────┐
                │  create_building_height_grid_from_gdf_ │
                │  polygon()                              │
                │  ├─ process_building_footprints_by_    │
                │  │  overlap() → fusion des IDs          │
                │  ├─ _process_with_rasterio()           │
                │  │  OU                                 │
                │  └─ _process_with_geometry_intersection │
                │     (seuil 30 % par cellule)           │
                └───────────────┬────────────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │  Grilles 2D (ny × nx)             │
              │  • building_height_grid            │
              │  • building_min_height_grid        │
              │  • building_id_grid                │
              │  • land_cover_grid                 │
              │  • dem_grid                        │
              │  • canopy_top / bottom             │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │  Voxelizer.generate_combined()     │
              │  ├─ _flatten_building_segments()  │
              │  ├─ _voxelize_kernel() [Numba]    │
              │  │  ground_level = DEM / voxel_size│
              │  │  voxel_grid[:,:,:ground] = -1   │
              │  │  voxel_grid[:,:,ground-1] = LC  │
              │  │  voxel_grid[:,:,tree_range] = -2│
              │  │  voxel_grid[:,:,bldg_range] = -3│
              │  └─ (fallback Python pur)          │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │  assemble_voxcity()                │
              │  └─ VoxCity(voxels, buildings,     │
              │             land_cover, dem,       │
              │             tree_canopy, extras)   │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │  save_results_h5() / save_h5()     │
              │  ├─ Grille voxels (gzip)           │
              │  ├─ Grilles 2D (gzip)              │
              │  ├─ building_min_heights (flat)    │
              │  ├─ canopy                         │
              │  └─ extras_gdf/building_gdf        │
              │     (GeoParquet bytes)             │
              └────────────────────────────────────┘
```

---

## 9. Paramètres de configuration clés

### 9.1 Paramètres de grille et voxels

| Paramètre | Défaut | Description | Source |
|-----------|--------|-------------|--------|
| `meshsize` | — | Taille de cellule en mètres (ex. `5.0`) | `PipelineConfig` |
| `voxel_size` | = `meshsize` | Taille du voxel en mètres | `Voxelizer.__init__()` |
| `voxel_dtype` | `np.int8` | Type des codes voxels | `Voxelizer` |
| `max_voxel_ram_mb` | `None` | Limite mémoire pour la grille 3D | `Voxelizer` |

### 9.2 Paramètres de chevauchement

| Paramètre | Défaut | Description | Source |
|-----------|--------|-------------|--------|
| `overlap_threshold` | `0.5` | Ratio de chevauchement pour fusionner deux bâtiments (50 %) | `process_building_footprints_by_overlap()` |
| `overlapping_footprint` | `"auto"` | `"auto"`, `True` (précis) ou `False` (rapide) | `create_building_height_grid_from_gdf_polygon()` |
| `_CELL_INTERSECTION_THRESHOLD` | `0.3` | Fraction minimale de cellule couverte pour attribuer un bâtiment | `buildings_precise.py` |
| `_CANDIDATE_CELL_MARGIN` | `2` | Marge en cellules autour de chaque bâtiment pour la recherche de candidats | `buildings_precise.py` |

Seuils de décision automatique (`_decide_auto_mode`) :

```python
_OVERLAP_HIGH = 0.15      # >15 % de bâtiments qui se chevauchent → mode précis
_OVERLAP_MEDIUM = 0.08    # >8 % en zone dense (>15% densité) → mode précis
_DENSITY_MEDIUM = 0.15    # seuil de densité associé
_OVERLAP_LOW_SMALL = 0.05 # >5 % pour ≤200 bâtiments → mode précis
```

### 9.3 Paramètres de terrain

| Paramètre | Défaut | Description | Source |
|-----------|--------|-------------|--------|
| `flatten_water_dem` | `True` | Aplatir le DEM sur les plans d'eau | `_flatten_water_dem_by_component()` |
| `water_dem_connectivity` | `4` | Connectivité pour la détection des plans d'eau (`4` ou `8`) | `_flatten_water_dem_by_component()` |
| `dem_interpolation` | `None` | Méthode d'interpolation du DEM | `create_dem_grid_from_geotiff_polygon()` |

### 9.4 Paramètres de canopée

| Paramètre | Défaut | Description | Source |
|-----------|--------|-------------|--------|
| `trunk_height_ratio` | `11.76/19.98 ≈ 0.59` | Ratio hauteur du tronc / hauteur totale de l'arbre | `Voxelizer` |
| `static_tree_height` | `10.0` | Hauteur statique des arbres (source `"Static"`) | `StaticCanopyStrategy` |
| `default_top_height` | `10.0` | Hauteur par défaut des arbres OSM | `OSMCanopyStrategy` |
| `default_trunk_height` | `4.0` | Hauteur de tronc par défaut OSM | `OSMCanopyStrategy` |
| `default_crown_ratio` | `0.6` | Ratio de la couronne | `OSMCanopyStrategy` |

### 9.5 Paramètres de construction

| Paramètre | Défaut | Description | Source |
|-----------|--------|-------------|--------|
| `building_source` | — | Source de données bâtiments | `PipelineConfig` |
| `building_complementary_source` | `None` | Source complémentaire pour les hauteurs manquantes | `get_building_height_grid()` |
| `building_complement_height` | `10.0` | Hauteur par défaut pour bâtiments sans donnée | `_patch_building_gdf()` |
| `complement_building_footprints` | `None` | Si `True`, ajoute les footprints manquants depuis la source complémentaire | `create_building_height_grid_from_gdf_polygon()` |
| `floor_height` | `3.0` | Hauteur par étage (OSM / Overture) | downloaders |
| `remove_perimeter_object` | `None` | Largeur de la bande périphérique à éliminer (en fraction de grille) | `VoxCityPipeline.run()` |

---

## 10. Notes sur la « reconstruction » pour CFD

Pour les cas d'usage comme foampilot, la géométrie CFD est reconstruite **non pas à partir des voxels**, mais à partir des grilles 2D et du `building_gdf` original. Le schéma dans `generate.py` (ligne 75) illustre ce flux :

```python
gdf = getattr(voxcity, "extras", {}).get("building_gdf")
for idx, row in gdf.iterrows():
    height = float(getattr(row, "height", 9.0) or 9.0)
    urban.add_building(Building(
        id=f"vox_{idx}",
        footprint=row.geometry,
        ground_z=0.0,
        roof_z=height,
        source="voxcity",
    ))
```

Les grilles `building_height_grid` et `building_id_grid` servent quant à elles aux simulateurs internes (solaire, visibilité) pour associer chaque cellule de sol ou façade à un bâtiment identifié.

---

## Références des fichiers sources

- Pipeline principale : `voxcity/generator/pipeline.py`
- Grilles 2D : `voxcity/generator/grids.py`
- Voxelisation : `voxcity/generator/voxelizer.py`
- Rasterisation rapide : `voxcity/geoprocessor/raster/buildings.py`
- Rasterisation précise : `voxcity/geoprocessor/raster/buildings_precise.py`
- Chevauchements : `voxcity/geoprocessor/overlap.py`
- Hauteurs complémentaires : `voxcity/geoprocessor/heights.py`
- Géométrie de grille : `voxcity/geoprocessor/raster/core.py`
- Entrées/sorties HDF5 : `voxcity/io.py`
- Modèles de données : `voxcity/models.py`
