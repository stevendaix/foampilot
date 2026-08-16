# Plan — Construction géométrique 3D d'un quartier pour CFD

## 1. Objectif

Ajouter à foampilot une **couche urbaine dédiée** pour générer automatiquement des cas CFD de quartier, en s'appuyant sur les briques existantes (`GmshMesher`, `DirectOpenFOAMExporter`, `Solver`, `WindAnalysis`) et en intégrant proprement les données GIS (BD TOPO, LiDAR, cadastre, OSM) via un modèle urbain canonique.

Le cas d'usage de référence est `examples/building_aero/generate_wind_cases.py`.

## 2. Pipeline retenu

```
                    DONNÉES TERRAIN
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       BD TOPO        LiDAR HD       Cadastre
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                 GIS NORMALIZATION
                         ↓
              URBAN MODEL CANONIQUE
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
        CFD GEOMETRY            VISUAL 3D
              ↓
        GEOMETRY CLEANUP
              ↓
      TOPOLOGICAL GEOMETRY
              ↓
       MESH GENERATION
        ┌─────┴─────┐
        ↓           ↓
      Gmsh       snappyHexMesh
        ↓           ↓
        └─────┬─────┘
              ↓
       OpenFOAM case
              ↓
        checkMesh
              ↓
      CFD validation
```

**Principe :** on ne part pas de Gmsh OCC comme source de vérité. On construit d'abord un modèle urbain Python pur, puis on l'injecte dans Gmsh via l'API existante.

## 3. Architecture dans foampilot

### 3.1 Nouveau module

```
foampilot/src/foampilot/urban/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── urban_model.py      ← UrbanModel, Building, Terrain, Road, RoofType, CFDLOD
│   └── domain.py           ← CFDDomain, WindFrame
├── coordinates/
│   ├── __init__.py
│   ├── wind_frame.py       ← WindFrame (world ↔ local CFD)
│   └── transforms.py       ← LocalTransform
├── simplification/
│   ├── __init__.py
│   ├── cfd_simplifier.py   ← CFDSimplifier, SimplificationOptions
│   ├── cleanup.py          ← GeometryCleanup, CleanupOptions
│   └── lod.py              ← CFDLOD enum + helpers
├── geometry/
│   ├── __init__.py
│   ├── cfd_geometry.py     ← CFDGeometry, CFDBuilding, CFDTerrain
│   ├── gmsh_backend.py     ← GmshQuarterBuilder
│   └── surface_backend.py  ← SnappyQuarterBuilder (futur)
├── mesh/
│   ├── __init__.py
│   ├── sizing.py           ← MeshConfig, WakeRefinement, RefinementRegion
│   ├── wake.py             ← WakeRefinement impl
│   ├── boundary_layers.py  ← BoundaryLayerConfig
│   └── gmsh_mesh_builder.py ← GmshMeshBuilder
├── patches/
│   ├── __init__.py
│   └── patch_assigner.py   ← PatchAssigner
├── bc/
│   ├── __init__.py
│   ├── patch_types.py      ← PatchTypes
│   ├── boundary_config.py  ← BoundaryConditionConfig, FieldBoundaryConditions
│   └── abl_profiles.py     ← ABLProfile, log/power law
├── validation/
│   ├── __init__.py
│   ├── geometry_checks.py  ← GeometryValidator
│   └── mesh_checks.py      ← MeshValidator
├── readers/
│   ├── __init__.py
│   ├── base_reader.py
│   ├── bdtopo.py
│   ├── cadastre.py
│   ├── lidar.py
│   ├── mnt.py
│   └── osm.py
└── utils/
    ├── __init__.py
    ├── geo_utils.py
    └── shapely_utils.py
```

### 3.2 Données de référence pour la Phase 3

| Source | Rôle | Format | Accès |
|--------|------|--------|-------|
| **GeoZones** | Contours administratifs INSEE | GeoJSON | `data.gouv.fr` |
| **IGN BD TOPO** | Bâtiments, routes, terrain | SHP/GPKG | https://geoservices.ign.fr/bdtopo |
| **OSM via osmnx** | Bâtiments, voirie | GeoJSON | OpenStreetMap |
| **LiDAR HD** | Toitures, terrain | LAZ/LAS | https://www.ign.fr/geoportail |

**Stratégie Phase 3** : commencer par OSM via `osmnx` car c’est immédiatement testable sans téléchargement manuel, puis ajouter BD TOPO quand les données seront disponibles.

### 3.2 Intégration avec l'existant

| Existant foampilot | Rôle dans urban |
|-------------------|------------------------|
| `GmshMesher` | Construction géométrie, booléens, maillage TetGen |
| `DirectOpenFOAMExporter` | Export `constant/polyMesh` |
| `Solver` | Configuration simpleFoam, BC, turbulence |
| `WindAnalysis` | `WindRose`, `WindCaseResult`, `LawsonProcessor` |
| `WeatherFileEPW` | Rose des vents EPW |
| `FoamPostProcessing` | VTK, slices, Q-criterion, vorticité |
| `ValueWithUnit` | **Gestion des unités physique pour toutes les grandeurs dimensionnées** |

**Règle :** ne pas réimplémenter ce qui existe déjà. Le nouveau module est une **couche d'orchestration urbaine** au-dessus de ces briques.

### 3.3 Gestion des unités

Toutes les grandeurs physiques dimensionnées utilisent `ValueWithUnit` (`foampilot.utilities.manageunits`).

**Règle de distinction :**
- **coordonnées / distances géométriques internes** : `float` en mètres (repère local CFD)
- **grandeurs physiques externes / configuration** : `ValueWithUnit`

Exemples :

```python
from foampilot.utilities.manageunits import ValueWithUnit

# MeshConfig
global_size = ValueWithUnit(15.0, "m")
building_size = ValueWithUnit(2.0, "m")
wake_size = ValueWithUnit(4.0, "m")

# CFDDomain
upstream = ValueWithUnit(8.0, "href")  # multiple de H
downstream = ValueWithUnit(15.0, "href")

# WindProfile
u_ref = ValueWithUnit(10.0, "m/s")
z0 = ValueWithUnit(0.3, "m")

# BoundaryLayerConfig
first_layer_height = ValueWithUnit(0.05, "m")
growth_rate = ValueWithUnit(1.2, "")  # sans unité
```

**Note :** `ValueWithUnit` supporte les conversions, la sérialisation JSON, et la génération des dimensions OpenFOAM. Il doit être utilisé dès que la valeur peut être fournie par l'utilisateur avec une unité différente, ou qu'elle est exposée dans une API publique.

## 4. Modèle urbain canonique

### 4.1 `Building`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from shapely.geometry import Polygon


class RoofType(str, Enum):
    FLAT = "flat"
    GABLE = "gable"
    HIP = "hip"
    PYRAMID = "pyramid"
    UNKNOWN = "unknown"


class CFDLOD(str, Enum):
    LOD0 = "lod0"
    LOD1 = "lod1"
    LOD2 = "lod2"
    LOD3 = "lod3"
    LOD4 = "lod4"


@dataclass
class Building:
    id: str
    footprint: Polygon
    ground_z: float
    roof_z: float
    roof_type: RoofType = RoofType.FLAT
    lod: CFDLOD = CFDLOD.LOD1
    source: str = "manual"
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.footprint.is_valid:
            raise ValueError(f"Building {self.id}: footprint is not valid")

        if self.roof_z <= self.ground_z:
            raise ValueError(
                f"Building {self.id}: roof_z must be greater than ground_z"
            )

    @property
    def height(self) -> float:
        return self.roof_z - self.ground_z

    @property
    def area(self) -> float:
        return self.footprint.area
```

**Règle :** `height` est une propriété calculée. Une seule source de vérité.

### 4.2 `UrbanModel`

```python
@dataclass
class UrbanModelMetadata:
    crs: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    description: Optional[str] = None
    units: str = "meters"


class UrbanModel:
    def __init__(self, crs: Optional[str] = None, metadata: Optional[UrbanModelMetadata] = None):
        self.crs = crs
        self.metadata = metadata or UrbanModelMetadata(crs=crs)
        self._buildings: dict[str, Building] = {}
        self._terrain: Optional[Terrain] = None
        self._roads: dict[str, Road] = {}

    def add_building(self, building: Building) -> None
    def add_terrain(self, terrain: Terrain) -> None
    def add_road(self, road: Road) -> None
    def buildings(self) -> List[Building]
    def building_count(self) -> int
    def bbox(self) -> Tuple[float, float, float, float, float, float]
    def center_xy(self) -> Tuple[float, float, float]
    def to_geojson(self, path: Path) -> None
    @classmethod
    def from_geojson(cls, path: Path) -> "UrbanModel"
    @classmethod
    def from_dict(cls, data: dict) -> "UrbanModel"
```

### 4.3 `CFDDomain`

```python
from typing import Literal

ReferenceHeightMethod = Literal["Hmax", "Hmean", "H90", "H95", "custom"]
ExtentUnits = Literal["href", "meters"]


@dataclass
class CFDDomain:
    upstream: float = 8.0
    downstream: float = 15.0
    lateral: float = 4.0
    top: float = 2.5
    extent_units: ExtentUnits = "href"
    reference_height_method: ReferenceHeightMethod = "Hmax"
    custom_reference_height: Optional[float] = None

    def compute_reference_height(self, urban: UrbanModel) -> float
    def compute_box(self, urban: UrbanModel, wind_frame: Optional[WindFrame] = None) -> Tuple[float, float, float, float, float, float]
```

**Note :** `upstream`, `downstream`, `lateral`, `top` sont des floats sans unité car ce sont des **multiples** de la hauteur de référence, ou des valeurs en mètres selon `extent_units`. La conversion vers des longueurs réelles se fait dans `compute_box()`.

### 4.4 `WindFrame`

```python
@dataclass
class WindFrame:
    """
    Convention:
        direction_deg = 0 -> flow along world +X
        direction_deg = 90 -> flow along world +Y
        +Z local = vertical, identical to world
    """
    direction_deg: float
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_local(self, x: float, y: float, z: float) -> Tuple[float, float, float]
    def to_world(self, x: float, y: float, z: float) -> Tuple[float, float, float]
```

**Principe :** les données restent dans leur système de coordonnées monde. Le `WindFrame` définit le repère local CFD. Les bâtiments ne sont jamais tournés individuellement.

## 5. `CFDSimplifier` — simplification CFD

### 5.1 Principe

Séparer **géométrie réelle** de **géométrie CFD**.

```python
@dataclass
class SimplificationOptions:
    simplify_tolerance: Optional[ValueWithUnit] = None  # None = auto
    min_building_area: ValueWithUnit = ValueWithUnit(1.0, "m^2")
    min_building_height: ValueWithUnit = ValueWithUnit(0.5, "m")
    min_gap: ValueWithUnit = ValueWithUnit(0.5, "m")
    merge_overlapping_buildings: bool = True
    remove_small_holes: bool = True
    hole_area_threshold: ValueWithUnit = ValueWithUnit(0.5, "m^2")
    snap_tolerance: Optional[ValueWithUnit] = None


class CFDSimplifier:
    def __init__(
        self,
        urban: UrbanModel,
        lod: CFDLOD = CFDLOD.LOD1,
        options: Optional[SimplificationOptions] = None,
    ):
        ...

    def simplify(self, wind_frame: Optional[WindFrame] = None) -> CFDGeometry
```

### 5.2 LOD CFD

| Niveau | Description | Usage |
|--------|-------------|-------|
| **CFD_LOD0** | Boîtes simples | Tests, debugging |
| **CFD_LOD1** | Empreinte réelle + hauteur moyenne | Référence majorité études |
| **CFD_LOD2** | Empreinte + toiture simplifiée | Bâtiments critiques |
| **CFD_LOD3** | Géométrie LiDAR simplifiée | Bâtiments critiques seulement |
| **CFD_LOD4** | Détails spécifiques | Localement uniquement |

## 6. `CFDGeometry`

```python
@dataclass
class CFDBuilding:
    id: str
    footprint_local: Polygon
    ground_z_local: float
    roof_z_local: float
    height: float
    source_building_id: str
    attributes: dict = field(default_factory=dict)


@dataclass
class CFDTerrain:
    ...


@dataclass
class CFDGeometry:
    buildings: List[CFDBuilding]
    terrain: Optional[CFDTerrain]
    domain_box: Tuple[float, float, float, float, float, float]
    lod: CFDLOD
    wind_frame: WindFrame
    metadata: dict = field(default_factory=dict)
```

**Important :** `domain_box` est dans le repère local CFD.

## 7. `GmshQuarterBuilder`

### 7.1 Principe

Utiliser `GmshMesher` existant. Ne pas créer de nouveau moteur Gmsh.

```python
class GmshQuarterBuilder:
    def __init__(self, case_path: Path, geometry: CFDGeometry):
        self.case_path = case_path
        self.geometry = geometry
        self._meshing = Meshing(case_path, mesher="gmsh")
        self._patch_assigner = PatchAssigner()
        self._built = False
        self._patches_assigned = False
        self._meshed = False

    def build(self) -> None
    def assign_patches(self) -> None
    def build_mesh(self, config: MeshConfig) -> None
    def export_openfoam(self) -> Path
```

### 7.2 Algorithme `build()`

1. Initialiser Gmsh
2. Créer la boîte fluide
3. Pour chaque bâtiment CFD :
   - Créer surface 2D depuis footprint
   - Extruder selon `height`
   - Affecter physical group 3D
4. `fragment()` sur tous les volumes
5. `cut()` fluide − bâtiments
6. Synchroniser

### 7.3 Robustesse

- Éviter les faces parfaitement coplanaires : `base_z = ground_z - eps`
- Tester `cut()` vs `fragment()`
- Pour 1000+ bâtiments : prévoir export STL + snappyHexMesh

## 8. `PatchAssigner` et `BoundaryConditionConfig`

### 8.1 Séparation stricte

```python
class PatchAssigner:
    def assign(self, builder: GmshQuarterBuilder, domain_bbox: Tuple) -> None
    # Définit les noms de patches : INLET, OUTLET, GROUND, TOP, SIDE_LEFT, SIDE_RIGHT, BUILDINGS
```

```python
@dataclass
class PatchTypes:
    inlet: str = "patch"
    outlet: str = "patch"
    top: str = "symmetryPlane"
    side_left: str = "symmetryPlane"
    side_right: str = "symmetryPlane"
    ground: str = "wall"
    buildings: str = "wall"
```

```python
@dataclass
class FieldBoundaryConditions:
    U_inlet: str = "fixedValue"
    U_outlet: str = "pressureInletOutletVelocity"
    p_inlet: str = "zeroGradient"
    p_outlet: str = "fixedValue"
    k_inlet: str = "fixedValue"
    k_outlet: str = "inletOutlet"
    omega_inlet: str = "fixedValue"
    omega_outlet: str = "inletOutlet"
    nut_ground: str = "nutkRoughWallFunction"
    nut_buildings: str = "nutkRoughWallFunction"
```

**Règle :** la géométrie définit les patches. La physique définit les BC. Même quartier, plusieurs cas CFD possibles.

### 8.2 `UrbanWindCase`

```python
class UrbanWindCase:
    def __init__(
        self,
        geometry: CFDGeometry,
        solver: str = "simpleFoam",
        turbulence: str = "kOmegaSST",
        wind_profile: str = "log",
        bc_config: Optional[BoundaryConditionConfig] = None,
    ):
        pass

    def setup(self, case_path: Path) -> None
    def run(self, nb_proc: int = 1) -> None
```

Permet d'avoir :
- `UrbanWindCase` — RANS
- `UrbanLESCase` — LES
- `UrbanThermalCase` — thermique

sans polluer `QuarterBuilder`.

## 9. `MeshSizing` — sizing intelligent

### 9.1 Composants

```python
@dataclass
class MeshConfig:
    global_size: ValueWithUnit = ValueWithUnit(15.0, "m")
    building_size: ValueWithUnit = ValueWithUnit(2.0, "m")
    wake_size: ValueWithUnit = ValueWithUnit(4.0, "m")
    ground_size: ValueWithUnit = ValueWithUnit(2.0, "m")
    top_size: Optional[ValueWithUnit] = None
    side_size: Optional[ValueWithUnit] = None
    min_size: ValueWithUnit = ValueWithUnit(0.1, "m")
    max_size: ValueWithUnit = ValueWithUnit(50.0, "m")
    grading_factor: float = 1.2
    wake_refinement: Optional[WakeRefinement] = None
    refinement_regions: List[RefinementRegion] = field(default_factory=list)
    boundary_layers: Optional[BoundaryLayerConfig] = None
    algorithm_2d: int = 6
    algorithm_3d: int = 1
```

### 9.2 `WakeRefinement`

```python
@dataclass
class WakeRefinement:
    length: float = 10.0      # × H
    width: float = 4.0        # × H
    height: float = 2.0       # × H
    target_size: ValueWithUnit = ValueWithUnit(2.0, "m")
    distance_threshold: Optional[ValueWithUnit] = None
```

### 9.3 Distance fields

```
distance aux bâtiments
distance au sol
courbure
sillage
```

## 10. `GeometryCleanup`

```python
@dataclass
class CleanupOptions:
    tolerance: ValueWithUnit = ValueWithUnit(0.05, "m")
    simplify_tolerance: Optional[ValueWithUnit] = None  # auto si None
    min_building_area: ValueWithUnit = ValueWithUnit(1.0, "m^2")
    min_building_height: ValueWithUnit = ValueWithUnit(0.5, "m")
    min_gap: ValueWithUnit = ValueWithUnit(0.5, "m")
    merge_overlapping_buildings: bool = True
    make_valid: bool = True
    remove_holes_below_area: ValueWithUnit = ValueWithUnit(0.5, "m^2")
```

Étapes :
1. Reprojection en CRS métrique si nécessaire
2. Validation / réparation des polygones (`shapely.make_valid()`)
3. Suppression des bâtiments trop petits
4. Simplification des empreintes
5. Suppression des trous trop petits
6. Fusion ou résolution des chevauchements
7. Snap des sommets sur grille si nécessaire
8. Suppression des gaps trop fins

## 11. Intégration `examples/building_aero`

### 11.1 Structure cible

```
examples/building_aero/
├── README.md
├── minimal_example.py         ← 1 bâtiment, 1 cas, validation
├── wind_rose_example.py       ← rose des vents + Lawson
├── buildings_config.json
├── wind_rose.json
├── generate_wind_cases.py     ← refactorisé pour utiliser QuarterBuilder
├── run_all_cases.py
├── wind_postprocess.py
├── wind_profile.py            ← déplacé vers foampilot utilities
├── mesh_experiment.py
├── mesh_quality.py
├── openfoam_quality.py
├── convergence_monitor.py
├── adaptive_mesher.py
├── cases/
├── data/
├── experiments/
└── old/                       ← archive du travail existant (WIP)
```

### 11.2 Refactoring `generate_wind_cases.py`

```python
# Après refactoring
from foampilot.urban import (
    Building, UrbanModel, CFDDomain, WindFrame, CFDLOD,
    CFDSimplifier, CFDGeometry,
    GmshQuarterBuilder, MeshConfig, WakeRefinement,
    BoundaryConditionConfig, UrbanWindCase,
)
from foampilot.utilities.manageunits import ValueWithUnit
from shapely.geometry import Polygon


def main():
    urban = UrbanModel(crs="EPSG:2154")
    urban.add_building(Building(
        id="B001",
        footprint=Polygon([(0, 0), (42, 0), (42, 18), (0, 18)]),
        ground_z=0.0,
        roof_z=12.5,
        source="manual",
    ))
    # ...

    wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
    domain = CFDDomain(
        upstream=ValueWithUnit(8.0, "href"),
        downstream=ValueWithUnit(15.0, "href"),
        lateral=ValueWithUnit(4.0, "href"),
        top=ValueWithUnit(2.5, "href"),
        extent_units="href",
        reference_height_method="Hmax",
    )
    geometry = CFDSimplifier(
        urban,
        lod=CFDLOD.LOD1,
    ).simplify(wind_frame=wind_frame)

    case_path = Path("cases/wind_270")

    builder = GmshQuarterBuilder(case_path, geometry)
    builder.build()
    builder.assign_patches()
    builder.build_mesh(MeshConfig(
        global_size=ValueWithUnit(15.0, "m"),
        building_size=ValueWithUnit(2.0, "m"),
        wake_size=ValueWithUnit(4.0, "m"),
        ground_size=ValueWithUnit(2.0, "m"),
        wake_refinement=WakeRefinement(
            length=10.0,
            width=4.0,
            height=2.0,
            target_size=ValueWithUnit(2.0, "m"),
        ),
    ))
    builder.export_openfoam()

    bc_config = BoundaryConditionConfig(
        top="symmetryPlane",
        sides="symmetryPlane",
        ground="wall",
        buildings="wall",
    )
    case = UrbanWindCase(
        geometry=geometry,
        solver="simpleFoam",
        turbulence="kOmegaSST",
        wind_profile="log",
        bc_config=bc_config,
    )
    case.setup(case_path)
    case.run(nb_proc=4)
```

## 12. Phase 0 — Écosystème existant

### 12.1 Recherche par capacité

| Capacité | Candidates |
|----------|-----------|
| **Administratif / contours** | `GeoZones` (data.gouv.fr / INSEE), `osmnx` |
| **LiDAR** | `lazrs`, `pylas`, `laspy`, `PDAL` |
| **GIS** | `shapely`, `geopandas`, `pyproj`, `rasterio` |
| **CityGML** | `citygml4py`, `pycitygml` |
| **3D buildings** | `osmnx` (buildings), `citysim` |
| **Mesh** | `meshio`, `pygalmesh`, `trimesh` |
| **Gmsh** | API Python déjà utilisée |
| **OpenFOAM** | `foampilot` lui-même |

### 12.2 Sources de données retenues

| Source | Usage | Format | Décision |
|--------|-------|--------|----------|
| **GeoZones** | Contours administratifs INSEE | GeoJSON | **WRAP** |
| **IGN BD TOPO** | Bâtiments, routes, terrain | SHP/GPKG | **Phase 3** |
| **OSM via osmnx** | Bâtiments, réseau viaire | GeoJSON | **Phase 3** |
| **LiDAR HD** | Toitures, terrain | LAZ/LAS | **Phase 4** |

### 12.2 Livrable

```
docs/ecosystem/
├── benchmark.md
├── lidar.md
├── building_reconstruction.md
├── city_models.md
├── urban_cfd.md
├── meshing.md
└── decisions.md
```

Pour chaque outil :
- URL GitHub
- Licence
- Dernier commit
- Stars
- Python / Linux
- Input / Output
- Fonction
- Performance
- Qualité
- Réutilisable directement ? / Wrapper nécessaire ? / Ignore ?
- Décision : **BUILD / BUY / WRAP / IGNORE**

## 13. Phases de développement

| Phase | Objectif | Priorité | Livrable | Statut |
|-------|----------|----------|----------|--------|
| **0** | Recherche écosystème | 🔴 Critique | `docs/ecosystem/*.md` + décisions | ✅ Fait |
| **1.1** | UrbanModel + Building + WindFrame + CFDDomain | 🔴 Critique | `checkMesh` OK sur 1 bâtiment | ✅ Fait |
| **1.2** | CFDSimplifier minimal + cleanup | 🔴 Critique | `CFDGeometry` propre | ✅ Fait |
| **1.3** | GmshQuarterBuilder minimal | 🔴 Critique | Géométrie Gmsh + booléens | ✅ Fait |
| **1.4** | PatchAssigner + BC + export OpenFOAM | 🔴 Critique | Cas OpenFOAM complet | ✅ Fait |
| **2** | Quartier synthétique 10–50 bâtiments + sizing + wake | 🔴 Critique | generate_wind_cases.py refactorisé | ✅ Fait |
| **3** | **Vrai quartier** + readers + validation source | 🔴 Critique | Exemple réel fonctionnel | ✅ OSMReader + GeoZones |
| **4** | Terrain MNT | 🟠 Haute | CFDTerrain + intégration | ✅ Fait |
| **5** | LiDAR + LOD2 | 🟠 Haute | Toitures simplifiées | ✅ Fait |
| **6** | Mesh sizing avancé + boundary layers | 🟠 Haute | MeshSizing complet + cartographies | ✅ Fait |
| **7** | snappyHexMesh backend | 🟡 Moyenne | SurfaceBackend fonctionnel | ✅ Fait |
| **8** | 1000+ bâtiments / performance | 🟡 Moyenne | Tests scalability + recommandations backend | ✅ Fait |
| **9** | Végétation / LOD3+ | 🟢 Plus tard | Porosité distribuée | ⏳ À faire |

## 14. Architecture cible finale

```text
                         DATA
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
           BD TOPO      LiDAR        OSM
              │           │           │
              └───────────┼───────────┘
                          ↓
                     UrbanModel
                          │
                    CFDSimplifier
                          │
                    CFDGeometry
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
        GmshBackend             SnappyBackend
              │                       │
              └───────────┬───────────┘
                          ↓
                       Mesh
                          │
                   OpenFOAMExporter
                          │
                          ↓
                    UrbanWindCase
                          │
                 ┌────────┴────────┐
                 ↓                 ↓
             simpleFoam           LES
                 │
                 ↓
             WindAnalysis
                 │
                 ↓
               Lawson
```

## 15. Règles architecturales

1. **UrbanModel = source de vérité Python** (pas Gmsh OCC)
2. **Une seule source de vérité par attribut** (`height = roof_z - ground_z`)
3. **Géométrie ≠ physique** : PatchAssigner définit les patches, BoundaryConditionConfig définit les BC
4. **Backends interchangeables** : Gmsh vs snappyHexMesh
5. **Données réelles dès Phase 3** (pas optionnel)
6. **Benchmark écosystème avant codage** (Phase 0)
7. **Ne pas réimplémenter** : utiliser `GmshMesher`, `DirectOpenFOAMExporter`, `Solver`, `WindAnalysis`, `ValueWithUnit`
8. **Nettoyage géométrique systématique** avant Gmsh
9. **Coordonnées locales CFD** : `UrbanModel` en monde, `CFDGeometry` en local
10. **Convention vent** : direction vers laquelle souffle le vent, +X local = inlet → outlet
11. **Unités** : `ValueWithUnit` pour toutes les grandeurs dimensionnées exposées dans les API publiques. `float` en mètres pour les coordonnées/distances internes du repère CFD local.

## 16. Validation

### 16.1 Par phase

| Phase | Validation |
|-------|-----------|
| 0 | Décisions BUILD/BUY/WRAP documentées |
| 1.1 | `UrbanModel`, `WindFrame`, `CFDDomain` tests unitaires |
| 1.2 | `CFDSimplifier` tests unitaires + cleanup |
| 1.3 | `GmshQuarterBuilder` : 1 bâtiment → maillage |
| 1.4 | `checkMesh` OK + cas simpleFoam |
| 2 | `checkMesh` OK + convergence sur 10–50 bâtiments |
| 3 | `OSMReader` fonctionnel + exemple `osm_neighborhood_example.py` |
| 4 | Terrain MNT intégré |
| 5 | Toitures LiDAR validées |
| 6 | Sizing fields + wake refinement validés |
| 7 | snappyHexMesh backend fonctionnel |
| 8 | 1000+ bâtiments, temps de construction acceptable |

### 16.2 Commandes

```bash
cd /home/steven/foampilot/foampilot
PYTHONPATH=src python3 -m pytest test/ -v
PYTHONPATH=src python3 -m pytest foampilot/src/foampilot/urban/tests/ -v
PYTHONPATH=src python3 -m pytest test/test_direct_openfoam_export.py -v
checkMesh -case /tmp/building_geo_case -allGeometry -allTopology
```

## 17. Séquence de développement recommandée (PRs)

### PR 1 — Modèle urbain de base
- `Building`, `UrbanModel`, `RoofType`, `CFDLOD`
- GeoJSON export/import
- tests unitaires sans Gmsh

### PR 2 — WindFrame et CFDDomain
- `WindFrame` + transformations world/local
- `CFDDomain` + calcul de boîte
- tests de roundtrip

### PR 3 — CFDSimplifier minimal
- `CFDSimplifier`, `SimplificationOptions`
- nettoyage de base
- production de `CFDGeometry`

### PR 4 — GmshQuarterBuilder minimal
- création boîte fluide
- extrusion d'un bâtiment
- booléen
- physical groups
- export OpenFOAM

### PR 5 — PatchAssigner + BC
- `PatchAssigner`
- `PatchTypes`, `FieldBoundaryConditions`
- `BoundaryConditionConfig`

### PR 6 — UrbanWindCase minimal
- configuration simpleFoam
- génération des champs initiaux
- BC de base

### PR 7 — Refactoring example
- `examples/building_aero/minimal_example.py`
- `examples/building_aero/generate_wind_cases.py` refactorisé
- README mis à jour

## 18. API cible finale

```python
from foampilot.urban import (
    UrbanModel, Building, Terrain,
    CFDDomain, WindFrame, CFDLOD,
    CFDSimplifier, CFDGeometry,
    GmshQuarterBuilder, MeshConfig, WakeRefinement,
    PatchAssigner, BoundaryConditionConfig,
    UrbanWindCase,
)
from foampilot.utilities.manageunits import ValueWithUnit
from shapely.geometry import Polygon


def main():
    # 1. Modèle urbain
    urban = UrbanModel(crs="EPSG:2154")
    urban.add_building(Building(
        id="B001",
        footprint=Polygon([(0, 0), (42, 0), (42, 18), (0, 18)]),
        ground_z=0.0,
        roof_z=12.5,
        source="manual",
    ))
    urban.add_building(Building(
        id="B002",
        footprint=Polygon([(50, 0), (80, 0), (80, 25), (50, 25)]),
        ground_z=0.0,
        roof_z=23.0,
        source="manual",
    ))

    # 2. Domaine CFD + repère vent
    wind_frame = WindFrame(direction_deg=270.0, origin=urban.center_xy())
    domain = CFDDomain(
        upstream=ValueWithUnit(8.0, "href"),
        downstream=ValueWithUnit(15.0, "href"),
        lateral=ValueWithUnit(4.0, "href"),
        top=ValueWithUnit(2.5, "href"),
        extent_units="href",
        reference_height_method="Hmax",
    )

    # 3. Simplification CFD
    geometry = CFDSimplifier(
        urban,
        lod=CFDLOD.LOD1,
    ).simplify(wind_frame=wind_frame)

    # 4. Construction + maillage + export
    case_path = Path("cases/wind_270")
    builder = GmshQuarterBuilder(case_path, geometry)
    builder.build()
    builder.assign_patches()
    builder.build_mesh(MeshConfig(
        global_size=ValueWithUnit(15.0, "m"),
        building_size=ValueWithUnit(2.0, "m"),
        wake_size=ValueWithUnit(4.0, "m"),
        ground_size=ValueWithUnit(2.0, "m"),
        wake_refinement=WakeRefinement(
            length=10.0,
            width=4.0,
            height=2.0,
            target_size=ValueWithUnit(2.0, "m"),
        ),
    ))
    builder.export_openfoam()

    # 5. Cas CFD
    bc_config = BoundaryConditionConfig(
        top="symmetryPlane",
        sides="symmetryPlane",
        ground="wall",
        buildings="wall",
    )
    case = UrbanWindCase(
        geometry=geometry,
        solver="simpleFoam",
        turbulence="kOmegaSST",
        wind_profile="log",
        bc_config=bc_config,
    )
    case.setup(case_path)
    case.run(nb_proc=4)
```
