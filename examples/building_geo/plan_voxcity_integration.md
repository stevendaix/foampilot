# Plan d'intégration VoxCity → OpenFOAM via snappyHexMesh avec terrain

## Objectif

Mettre en place une **pipeline alternative** à Gmsh, basée sur VoxCity + snappyHexMesh, pour :
- exploiter VoxCity comme source agrégée de bâtiments + DEM,
- produire un cas OpenFOAM prêt à mailler avec `blockMesh` + `snappyHexMesh`,
- conserver **intacte** la pipeline Gmsh existante.

```text
VoxCity
  ↓
UrbanModel
  ↓
Terrain + bâtiments STL
  ↓
Cas OpenFOAM prêt à mailler
  ↓
blockMesh → surfaceFeatures → snappyHexMesh → OpenFOAM
```

---

## Architecture cible

### Pipeline recommandée

```text
VoxCityReader
    │
    ├─ building_gdf
    ├─ DEM
    └─ metadata/sources
    │
    ▼
Projection / normalisation CRS
    │
    ▼
UrbanModel
    │
    ├─ buildings
    ├─ terrain
    └─ domain
    │
    ▼
SnappyCaseBuilder  (module léger de wiring)
    │
    ├─ constant/triSurface/terrain.stl        ← TerrainProcessor
    ├─ constant/triSurface/buildings.stl      ← BuildingExtruder
    ├─ system/blockMeshDict                   ← SnappyMesher existant
    ├─ system/surfaceFeaturesDict             ← SnappyMesher existant
    ├─ system/snappyHexMeshDict               ← SnappyMesher existant
    ├─ 0/*                                    ← foampilot.Solver
    ├─ constant/*                             ← foampilot.Solver
    └─ system/controlDict                     ← foampilot.Solver
    │
    ▼
OpenFOAM case
```

### Principe important

- **VoxCity + snappyHexMesh** = nouveau backend, totalement séparé de `GmshQuarterBuilder`.
- On ne modifie **pas** la pipeline Gmsh existante.
- Le cas OpenFOAM final reste compatible avec `foampilot.Solver`, `FoamPostProcessing`, etc.
- Les commandes OpenFOAM sont lancées depuis `SnappyMesher` ou `SnappyCaseBuilder`.

---

## 1. Brique existante à réutiliser : `foampilot.mesh.snappymesh.SnappyMesher`

Fichier : `foampilot/src/foampilot/mesh/snappymesh.py`

### Ce qu’elle sait déjà faire

- Écrire `system/snappyHexMeshDict`
- Écrire `system/surfaceFeaturesDict`
- Écrire `system/blockMeshDict`
- Lancer `blockMesh`, `surfaceFeatures`, `snappyHexMesh`

### Ce qu’elle ne sait pas faire (et que nous allons ajouter)

- Générer les STL `terrain.stl` et `buildings.stl`
- Configurer automatiquement `locationInMesh` depuis une `UrbanModel`
- Gérer plusieurs STL avec régions distinctes
- Intégrer le terrain non-plat
- Préparer les BCs via `foampilot.Solver`

### Décision

**Ne pas dupliquer `SnappyMesher`.**  
On l’utilise tel quel, et on crée un module `SnappyCaseBuilder` par-dessus pour :
1. générer les STL,
2. instancier/configurer `SnappyMesher`,
3. écrire la physique via `foampilot.Solver`.

---

## 2. Module `VoxCityReader`

Fichier : `foampilot/urban/readers/voxcity_reader.py`

### Responsabilités

- Télécharger/voxéliser la zone via VoxCity,
- extraire `building_gdf`,
- extraire le DEM,
- tout projeter en CRS métrique local,
- produire un `UrbanModel` + un `CFDTerrain`.

### API cible

```python
class VoxCityReader:
    def __init__(
        self,
        meshsize: float = 5.0,
        building_source: str | None = None,
        dem_source: str | None = None,
        land_cover_source: str | None = None,
        canopy_height_source: str | None = None,
    ):
        ...

    def read(self, rectangle_vertices: list) -> Tuple[UrbanModel, CFDTerrain]:
        ...
```

### Points d’attention

- VoxCity nécessite Google Earth Engine + authentification.
- Les géométries arrivent en WGS84 → projection locale via `osmnx.projection.project_gdf()`.
- Les hauteurs peuvent être manquantes → fallback `levels * 3` ou hauteur par défaut.
- Le DEM est une grille numpy → directement utilisable par `CFDTerrain.from_grid()`.

---

## 3. Module `TerrainProcessor`

Fichier : `foampilot/urban/terrain/processor.py`

### Responsabilités

- Recevoir un `CFDTerrain` ou une grille DEM numpy,
- construire une surface fermée pour snappyHexMesh :
  - surface supérieure `z = DEM(x, y)`,
  - jupes latérales,
  - fond inférieur à `z_bottom`,
- simplifier la surface pour garder un STL léger,
- exporter `terrain.stl`.

### API cible

```python
class TerrainProcessor:
    def __init__(self, terrain: CFDTerrain, config: TerrainConfig):
        ...

    def build_closed_surface(self) -> pv.PolyData:
        ...

    def export_stl(self, output_path: Path) -> Path:
        ...
```

### Points d’attention

- Le STL doit être **fermé** pour snappyHexMesh.
- Les normales doivent être cohérentes.
- Éviter les faces dupliquées et les trous.
- Simplification via `pymesh` / `trimesh` / `PyVista`.

---

## 4. Module `BuildingExtruder`

Fichier : `foampilot/urban/geometry/building_extruder.py`

### Responsabilités

- Recevoir les bâtiments d’`UrbanModel`,
- nettoyer/simplifier les footprints,
- calculer l’ancrage sur le terrain (`base_z = ground_z - foundation_depth`),
- extruder en volumes fermés,
- exporter `buildings.stl`.

### API cible

```python
class BuildingExtruder:
    def __init__(self, buildings: List[CFDBuilding], terrain: CFDTerrain, config: BuildingConfig):
        ...

    def build_solids(self) -> List[pv.PolyData]:
        ...

    def export_stl(self, output_path: Path) -> Path:
        ...
```

### Points d’attention

- Utiliser `PyVista` ou `trimesh` pour construire les solides fermés.
- Vérifier que chaque bâtiment est un volume fermé.
- Gérer les footprints invalides / trop petites.

---

## 5. Module `SnappyCaseBuilder`

Fichier : `foampilot/openfoam/snappy_case_builder.py`

### Responsabilités

- Instancier/configurer `SnappyMesher` existant,
- générer `terrain.stl` et `buildings.stl`,
- écrire `blockMeshDict` avec une bbox adaptée au terrain + bâtiments,
- écrire `snappyHexMeshDict` avec :
  - `locationInMesh` dans le fluide,
  - raffinement terrain / bâtiments,
  - surfaces STL référencées,
- lancer la séquence OpenFOAM via `SnappyMesher`,
- réutiliser `foampilot.Solver` pour la physique, les BCs, les champs initiaux.

### API cible

```python
class SnappyCaseBuilder:
    def __init__(
        self,
        case_dir: Path,
        urban: UrbanModel,
        terrain: CFDTerrain,
        solver: Solver,
        domain_config: DomainConfig,
        terrain_config: TerrainConfig,
        building_config: BuildingConfig,
        mesh_config: SnappyMeshConfig,
    ):
        ...

    def write_stl(self) -> tuple[Path, Path]:
        """Generate terrain.stl and buildings.stl."""
        ...

    def configure_snappy(self) -> SnappyMesher:
        """Configure SnappyMesher from urban model and STL bounds."""
        ...

    def write(self) -> Path:
        """Generate all OpenFOAM files without running mesh commands."""
        ...

    def build_mesh(self) -> None:
        """Run full mesh pipeline: blockMesh → surfaceFeatures → snappyHexMesh."""
        ...
```

### Points clés

- `SnappyCaseBuilder` **ne réinvente pas** la physique du cas.
- Il ajoute seulement :
  - la géométrie STL,
  - les dictionnaires de maillage.
- `foampilot.Solver` gère :
  - `controlDict`
  - `fvSchemes`
  - `fvSolution`
  - `0/*` initial fields
  - boundary conditions

---

## 6. Gestion du terrain

### Entrée

DEM depuis VoxCity → grille numpy.

### Traitement

- Clip sur la zone d’intérêt,
- reprojection,
- résampling à `dem_resolution`,
- fermeture du terrain en volume pour STL :
  - surface supérieure `z = DEM(x, y)`,
  - jupes latérales,
  - fond inférieur à `z_bottom`.

### Export

- `constant/triSurface/terrain.stl`

### Paramètres

```python
@dataclass
class TerrainConfig:
    dem_resolution: float = 5.0
    horizontal_extension: float = 50.0
    bottom_offset: float = 20.0
    smoothing_iterations: int = 1
    simplify_tolerance: float | None = 0.5
    fill_nodata: bool = True
    nodata_threshold: float = -9999.0
```

---

## 7. Gestion des bâtiments

### Entrée

Footprints 2D + hauteurs depuis VoxCity `building_gdf`.

### Nettoyage

- Supprimer géométries invalides,
- découper `MultiPolygon`,
- filtrer surfaces trop petites,
- simplifier les contours.

### Hauteur

Hiérarchie :
1. hauteur explicite,
2. roof height,
3. `levels * level_height`,
4. médiane locale,
5. hauteur par défaut.

### Ancrage terrain

```python
ground_z = dem.sample_at_footprint(footprint, method="min_or_centroid")
base_z = ground_z - foundation_depth
roof_z = ground_z + building_height
```

### Export

- `constant/triSurface/buildings.stl`

### Paramètres

```python
@dataclass
class BuildingConfig:
    min_area: float = 10.0
    simplify_tolerance: float = 0.25
    default_height: float = 9.0
    level_height: float = 3.0
    foundation_depth: float = 1.0
```

---

## 8. Domaine CFD

### Calcul de la bbox

```python
xmin = min(buildings_x) - margin_x
xmax = max(buildings_x) + margin_x
ymin = min(buildings_y) - margin_y
ymax = max(buildings_y) + margin_y
zmin = min(terrain_z) - bottom_margin
zmax = max(terrain_z + building_height) + top_margin
```

### Marges

```python
@dataclass
class DomainConfig:
    margin_x: float = 100.0
    margin_y: float = 100.0
    top_margin: float = 100.0
    bottom_margin: float = 20.0
    base_cell_size: float = 5.0
```

---

## 9. Configuration du maillage snappyHexMesh

```python
@dataclass
class SnappyMeshConfig:
    base_cell_size: float = 5.0
    terrain_refinement_level: int = 2
    building_refinement_level: int = 3
    n_cells_between_walls: int = 4
    max_global_cells: int = 5_000_000
    add_layers: bool = False
```

### Logique de raffinement

- `terrain` : `level (2 2)`
- `buildings` : `level (3 3)`
- `locationInMesh` : centre du domaine, z au-dessus du point le plus haut.

---

## 10. Arborescence OpenFOAM produite

```text
case/
├── 0/
│   ├── U
│   ├── p
│   ├── k
│   ├── epsilon
│   └── nut
│
├── constant/
│   └── triSurface/
│       ├── terrain.stl
│       └── buildings.stl
│
└── system/
    ├── blockMeshDict
    ├── surfaceFeatureExtractDict
    ├── snappyHexMeshDict
    ├── createPatchDict
    ├── controlDict
    ├── fvSchemes
    └── fvSolution
```

### Règles de construction

- Tout ce qui est physique/BCs/initial conditions → `foampilot.Solver`.
- Tout ce qui est géométrie/maillage → `SnappyCaseBuilder` + `SnappyMesher`.
- Pas de doublon, pas de sortie de route.

---

## 11. Tests de validation

### Tests unitaires

| Test | Description |
|------|-------------|
| `test_voxcity_reader_paris` | Lecture d’un quartier parisien |
| `test_voxcity_reader_lyon` | Lecture d’un quartier lyonnais |
| `test_voxcity_terrain` | DEM correctement projeté |
| `test_voxcity_to_urban` | Conversion VoxCity → UrbanModel |
| `test_terrain_stl_closed` | STL terrain fermé |
| `test_buildings_stl_closed` | STL bâtiments fermés |
| `test_snappy_case_structure` | Arborescence OpenFOAM correcte |
| `test_block_mesh_dict` | Dictionnaire valide |
| `test_snappy_dict` | Dictionnaire valide |

### Tests d’intégration

| Test | Description |
|------|-------------|
| `test_snappy_flat_terrain` | Terrain plat + snappy passe |
| `test_snappy_real_terrain` | DEM réel + bâtiments ancrés |
| `test_check_mesh` | Qualité mesh acceptable |

---

## 12. Dépendances

### Nouvelle dépendance optionnelle

```toml
voxcity = {version = ">=0.4", optional = true}
```

### Dépendances existantes déjà utilisées

- `osmnx` (projection)
- `shapely`
- `numpy`
- `gmsh` potentiellement pour certains traitements géométriques
- `foampilot.mesh.snappymesh.SnappyMesher` (existant)
- `foampilot.Solver` (existant)

### Prérequis système

- Google Earth Engine authentication
- GDAL

---

## 13. Roadmap d’implémentation

### Phase 0 — Cadrage (0.5 jour)
- [x] Valider le plan
- [ ] Choisir une zone test petite
- [ ] Préparer fixtures locales VoxCity si possible

### Phase 1 — Reader VoxCity (1-2 jours)
- [ ] Créer `voxcity_reader.py`
- [ ] Extraire bâtiments + DEM
- [ ] Projection CRS → mètres
- [ ] Construire `UrbanModel` + `CFDTerrain`

### Phase 2 — STL terrain (1-2 jours)
- [ ] `TerrainProcessor`
- [ ] Surface fermée avec jupes
- [ ] Export `terrain.stl`

### Phase 3 — STL bâtiments (1 jour)
- [ ] `BuildingExtruder`
- [ ] Ancrage sur terrain
- [ ] Export `buildings.stl`

### Phase 4 — SnappyCaseBuilder (1-2 jours)
- [ ] Wiring de `SnappyMesher` existant
- [ ] Génération `blockMeshDict` adaptée au terrain
- [ ] Génération `snappyHexMeshDict` avec plusieurs STL
- [ ] Intégration `foampilot.Solver` pour la physique

### Phase 5 — Commandes OpenFOAM (0.5 jour)
- [ ] `blockMesh` via `SnappyMesher`
- [ ] `surfaceFeatures` via `SnappyMesher`
- [ ] `snappyHexMesh` via `SnappyMesher`
- [ ] `createPatch` si nécessaire
- [ ] `checkMesh`

### Phase 6 — Validation (1-2 jours)
- [ ] Cas test plat
- [ ] Cas test DEM réel
- [ ] Vérifications qualité

### Phase 7 — Intégration foampilot (1 jour)
- [ ] Exports dans `foampilot.urban`
- [ ] Exemple
- [ ] Documentation courte

---

## 14. Fichiers à créer / modifier

### À créer

| Fichier | Rôle |
|---------|------|
| `foampilot/urban/readers/voxcity_reader.py` | Lecture VoxCity → UrbanModel + terrain |
| `foampilot/urban/terrain/processor.py` | DEM → terrain STL fermé |
| `foampilot/urban/geometry/building_extruder.py` | Footprints → bâtiments STL |
| `foampilot/openfoam/snappy_case_builder.py` | Wiring : STL + SnappyMesher + Solver |
| `foampilot/urban/snappy_config.py` | Configs dataclasses |
| `examples/building_geo/voxcity_snappy_example.py` | Exemple bout en bout |
| `tests/test_snappy_terrain.py` | Tests unitaires/integration |

### À modifier

| Fichier | Action |
|---------|--------|
| `foampilot/mesh/snappymesh.py` | Adapter pour supporter plusieurs STL + locationInMesh automatique |
| `foampilot/urban/model/urban_model.py` | Vérifier compatibilité terrain |
| `foampilot/urban/__init__.py` | Exporter nouveaux symboles si besoin |
| `pyproject.toml` | Dépendance optionnelle `voxcity` |
| `SESSION.md` | Mettre à jour le suivi |

### À ne pas modifier

- `foampilot/urban/geometry/gmsh_backend.py`
- `foampilot/urban/mesh/gmsh_quarter_builder.py`
- `foampilot/urban/patches/patch_assigner.py`

---

## 15. Points de vigilance

1. **Ne pas casser Gmsh** : nouveau backend strictement séparé.
2. **Réutiliser foampilot** : la physique, les BCs, les champs initiaux passent par `Solver`.
3. **STL fermés** : snappyHexMesh échoue vite sur des surfaces ouvertes.
4. **Terrain réaliste mais pas trop lourd** : éviter les DEM trop denses sans simplification.
5. **VoxCity/EE instable** : prévoir fallback clair si Earth Engine n’est pas dispo.
6. **Utiliser `SnappyMesher` existant** : ne pas réimplémenter ce qui existe déjà dans `foampilot/mesh/snappymesh.py`.

---

## 16. Durée estimée

- MVP minimal : **4 à 6 jours**
- Version robuste : **9 à 12 jours**
