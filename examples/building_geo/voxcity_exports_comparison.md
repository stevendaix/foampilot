# VoxCity → OpenFOAM : comparaison des exports pour CFD urbaine

## Contexte
- Zone test : Paris 15e, ~150 m × 150 m, `meshsize = 5 m`
- Grille VoxCity : `(29, 22, 6)` → ~0.5 MB
- Bâtiments : OSM + Microsoft Footprints
- DEM : IGN RGE ALTI 1m
- Coût EE : téléchargement DEM + canopy + buildings + land cover

---

## 1. Exports natifs VoxCity

| Format | Fonction | Contenu | Cas d’usage |
|--------|----------|---------|-------------|
| **HDF5 / NPY** | `city.voxels.classes`, `city.dem.elevation`, `city.to_xarray()` | Grille voxel brute + DEM | Pipeline custom, traitement direct |
| **OBJ** | `export_obj(city, ...)` | Maillage voxelisé des bâtiments + terrain | Blender/Rhino, visualisation |
| **VOX** | `export_magicavoxel_vox(city, ...)` | Voxels éditables | Prototypage visuel |
| **INX/EDB** | `export_inx(city, ...)` | Modèle ENVI-met | Microclimat, pas CFD directement |

### 1.1 Accès direct aux données brutes (sans export fichier)
```python
city = get_voxcity(rectangle_vertices, meshsize=5, output_dir='output')

# Grille sémantique 3D : (north, east, up)
voxels = city.voxels.classes          # numpy.ndarray int8
dem = city.dem.elevation              # numpy.ndarray 2D

# Building GeoDataFrame vectoriel
gdf = city.extras['building_gdf']

# via xarray
ds = city.to_xarray()
```

---

## 2. Chemins de conversion vers OpenFOAM/snappyHexMesh

### Option A — Données vectorielles extraites (`building_gdf` + DEM)
**Pipeline** :
1. `building_gdf` → extraction footprints + hauteurs
2. `CFDTerrain.from_grid(dem_grid)` → terrain STL fermé
3. `BuildingExtruder` → bâtiments STL
4. `SnappyCaseBuilder` → cas OpenFOAM complet

**Avantages** :
- géométrie propre, pas de voxelisation
- contrôle total sur simplification/ancrage
- pas de dépendance à la résolution voxel
- compatible avec la pipeline Gmsh existante

**Inconvénients** :
- nécessite de parser `building_gdf`
- perte de détails fins si `meshsize` grand

**Matûrité** : ✅ Codé dans `VoxCityReader` + `TerrainProcessor` + `BuildingExtruder`

---

### Option B — Export OBJ + retraitement
**Pipeline** :
1. `export_obj(city, ...)` → `voxcity.obj`
2. Ouvrir OBJ dans PyVista / trimesh
3. Séparer bâtiments / terrain par matériaux
4. Nettoyer, fermer, simplifier
5. Exporter `terrain.stl` + `buildings.stl`

**Avantages** :
- utilise un export officiel VoxCity
- garde la voxelisation intacte

**Inconvénients** :
- OBJ voxelisé = faces carrées, beaucoup de triangles
- fichiers lourds, maillage snappyHexMesh explosé
- séparation bâtiments/terrain délicate
- risque de surfaces non fermées

**Matûrité** : ⚠️ À implémenter

---

### Option C — Grille voxel brute → extraction surfaces
**Pipeline** :
1. `city.voxels.classes` + `city.dem.elevation`
2. Seuiller les classes `13` (Building) → masque 3D
3. Marchings cubes / surface nets → maillage bâtiments
4. DEM → surface terrain + jupes
5. STL

**Avantages** :
- pas de dépendance aux exports VoxCity
- contrôle total sur la résolution
- potentiellement plus rapide que l’OBJ

**Inconvénients** :
- marching cubes sur voxels → maillage “blocky”
- nécessite `scikit-image` ou `pyvista`/`trimesh`
- artefacts aux transitions voxel/terrain

**Matûrité** : ⚠️ À implémenter

---

### Option D — ENVI-met INX → conversion vers OpenFOAM
**Pipeline** :
1. `export_inx(city, ...)` → `voxcity.inx`
2. Parser INX (XML)
3. Extraire géométries + surfaces
4. Convertir vers STL / snappyHexMesh

**Avantages** :
- format structuré, documenté
- contient déjà un domaine CFD-like

**Inconvénients** :
- format ENVI-met propriétaire
- perte d’information / conversion approximative
- temps de dev important

**Matûrité** : ❌ Non pertinent pour OpenFOAM direct

---

## 3. Tableau de décision

| Critère | A — `building_gdf` + DEM | B — OBJ | C — Voxel brut |
|---------|---------------------------|---------|----------------|
| Qualité géométrie | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| Performance snappyHexMesh | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| Facilité implémentation | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Fidélité au modèle VoxCity | ★★★☆☆ | ★★★★★ | ★★★★★ |
| Coût de traitement | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| Dépendances | OSM/MS/IGN | PyVista | scikit-image |

**Recommandation** : commencer par **Option A** (déjà codée), puis ajouter **Option B** comme alternative si l’OBJ montre des avantages pour des zones très denses.

---

## 4. Plan de test

### Test 1 — Option A (vectorielle)
- [ ] Télécharger VoxCity réel avec `VoxCityReader`
- [ ] Extraire `building_gdf` + DEM
- [ ] Générer `terrain.stl` + `buildings.stl`
- [ ] Lancer `blockMesh → surfaceFeatures → snappyHexMesh`
- [ ] Vérifier `checkMesh`

### Test 2 — Option B (OBJ)
- [ ] `export_obj(city, ...)`
- [ ] Ouvrir avec PyVista
- [ ] Séparer par matériaux / labels
- [ ] Nettoyer et fermer
- [ ] Exporter STL et comparer le nombre de triangles

### Test 3 — Option C (voxel brut)
- [ ] Extraire `city.voxels.classes`
- [ ] Marching cubes sur classe `13` (Building)
- [ ] Comparer le maillage obtenu avec Option A

### Métriques de comparaison
- Nombre de triangles dans `buildings.stl`
- Temps de `snappyHexMesh`
- Qualité mesh (`checkMesh` : max non-ortho, skewness)
- Nombre de cellules finales
- Mémoire RAM utilisée

---

## 5. Fichiers de référence

| Fichier | Rôle |
|---------|------|
| `examples/building_geo/voxcity_snappy_example.py` | Exemple synthétique / VoxCity |
| `examples/building_geo/voxcity_residential_test.py` | Test zone résidentielle réelle |
| `foampilot/src/foampilot/urban/readers/voxcity_reader.py` | Lecture VoxCity → UrbanModel |
| `foampilot/src/foampilot/urban/terrain/processor.py` | DEM → terrain STL |
| `foampilot/src/foampilot/urban/geometry/building_extruder.py` | Bâtiments → STL |
| `foampilot/src/foampilot/openfoam/snappy_case_builder.py` | Wiring complet |

---

## 6. Prochaine étape recommandée

1. **Valider Test 1** sur une petite zone résidentielle avec `building_gdf`
2. Si ça marche, ajouter un mode OBJ dans `VoxCityReader` pour comparaison
3. Benchmarker les 3 options sur la même zone
4. Choisir la meilleure voie et documenter le choix
