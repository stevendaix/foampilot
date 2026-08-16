# Rapport de calcul CFD — Quartier Paris 15e avec données VoxCity

## 1. Contexte et objectifs

Ce rapport documente la mise en place complète d'une simulation CFD urbaine autour du quartier Paris 15e en utilisant les données VoxCity et foampilot. L'objectif est d'analyser le confort éolien au niveau piétonnier et les efforts sur les bâtiments.

## 2. Cartographie du quartier

### 2.1 Données VoxCity

Les données proviennent de VoxCity (AOI Paris 15e) avec les caractéristiques suivantes :
- **Source bâtiments** : Overture
- **Resolution grille** : 5.0 m
- **Nombre de bâtiments initiaux** : 30
- **Bâtiments après simplification** : 7

### 2.2 Visualisation matplotlib du quartier

```python
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box

# Charger les données VoxCity
from voxcity.io import load_voxcity
vox = load_voxcity("output/voxcity.h5")
gdf = vox.extras["building_gdf"]

# Domaine CFD (exemple pour Paris 15e)
xmin, ymin, xmax, ymax = 450193, 5410497, 450541, 5411824

# Créer la figure
fig, ax = plt.subplots(figsize=(12, 10))

# Tracer les bâtiments
gdf.plot(ax=ax, color="#cccccc", edgecolor="#666666", linewidth=0.5)

# Ajouter le domaine CFD
domain = box(xmin, ymin, xmax, ymax)
gpd.GeoSeries([domain]).plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)

ax.set_xlabel("X (m, EPSG:32631)")
ax.set_ylabel("Y (m, EPSG:32631)")
ax.set_title("Quartier Paris 15e — Données VoxCity et domaine CFD")
plt.savefig("quartier_overview.png", dpi=150)
```

### 2.3 Visualisation folium du quartier réel

Une carte interactive `map_view.html` est générée avec :
- Bâtiments VoxCity en gris
- Domaine CFD en rouge
- Couche de base OpenStreetMap

## 3. Mise en place du calcul

### 3.1 Pipeline complet

```
VoxCity HDF5
    → [VoxCity] process_building_footprints_by_overlap()
    → [Custom] Gap closing + height merging
    → UrbanModel → VectorGmshBuilder
    → Gmsh cut() séquentiel (primaire) / fragment() (fallback)
    → DirectOpenFOAM export
    → OpenFOAM case setup + quality gate
    → foamRun (incompressibleFluid, kEpsilon)
    → Post-processing VoxCity-aware
```

### 3.2 Étapes détaillées

**Étape 1 : Chargement VoxCity**
- Fichier HDF5 : `output/voxcity.h5`
- Reprojection EPSG:4326 → EPSG:32631
- Extraction bâtiments + terrain
- VoxCity overlap processor (R-tree, threshold=0.5) pour fusionner les bâtiments qui se chevauchent de plus de 50%

**Étape 2 : Simplification géométrie**
- Gap closing distance-based (seuil = mesh_size × 0.5 = 3.0m)
- Politique hauteur : max(h1, h2) pour bâtiments fusionnés
- Résultat : 7 bâtiments après simplification (30 → 5 filtrés → 14 simplifiés → 7 finaux)

**Étape 3 : Construction Gmsh**
- Domaine fluide avec marges automatiques :
  - Amont : 4 × Hmax
  - Aval : 7.5 × Hmax
  - Latéral : 2 × D
  - Haut : 1.25 × Hmax
- Bâtiments en boîtes axis-aligned
- **Stratégie Boolean** : `cut()` séquentiel (primaire) avec fallback `fragment()` si échec
- `algorithm_3d=4` (Delaunay)
- Mesh size : 6.0 m par défaut (pipeline CLI), 4.0m dans config.json (lc_min=3.0, lc_max=12.0)
- **Check qualité** : `OpenFOAMQualityAnalyzer` + `checkMesh` automatique avant solveur
- **Décomposition** : `decomposeParDict(4)` pour parallélisation future

**Étape 4 : Export OpenFOAM**
- Export direct via DirectOpenFOAMExporter
- 7 patches : inlet, outlet, top, ground, side_left, side_right, buildings
- Types de patches :
  - `inlet` / `outlet` : patch (conditions aux limites définies dans 0/)
  - `buildings` / `ground` / `side_left` / `side_right` : wall
  - `top` : symmetry
- 35386 nœuds, 193411 cellules

**Étape 5 : Configuration solveur**
- Solver : incompressibleFluid
- Turbulence : kEpsilon
- Profil log-wind à l'entrée (codedFixedValue) avec mitigation singularité : `Foam::max(z / z0, 1.0 + SMALL)`
- Relaxation : p=0.3, U/k/eps=0.7
- nNonOrthogonalCorrectors = 2
- decomposeParDict(4)
- **Quality gate** : mesh validé par OpenFOAMQualityAnalyzer + checkMesh avant lancement

**Étape 6 : Post-traitement**
- foamToVTK → PyVista
- Cartes vitesse, Cp, TI, confort éolien
- Export JSON + CSV

## 4. Justification des hypothèses

### 4.1 Loi logarithmique de vent

Le profil de vitesse est modélisé par la loi log :

```
u(z) = (u* / κ) * ln(z / z0)
```

**Justification** :
- Valide pour la couche limite atmosphérique (z > 10m)
- Paramètres standardisés : κ = 0.41 (constante von Karman), z0 = 0.3m (urbain dense)
- Cohérent avec les données EPW météo
- Implémenté via codedFixedValue pour adaptabilité

### 4.2 Modèle de turbulence kEpsilon

**Justification** :
- Robuste pour écoulements extérieurs
- Moins sensible aux conditions initiales que kOmegaSST
- Bien validé pour CFD urbaine

### 4.3 Marges de domaine

Marges divisées par 2 par rapport aux règles `building_aero` :
- Réduction de la taille du maillage de 4.1M → 193k cellules
- Temps de calcul réduit de ~30min → ~3min
- Qualité maillage acceptable (badness < 200k)

### 4.4 Simplification bâtiments

- Fusion si gap < mesh_size × 0.5 = 3.0m (pour mesh_size=6.0m)
- Politique hauteur : max(h1, h2) pour bâtiments fusionnés
- Évite les cellules déformées qui font dériver le solveur

## 5. Résultats

### 5.1 Statistiques maillage

| Paramètre | Valeur |
|-----------|--------|
| Nœuds | 35386 |
| Cellules | 193411 |
| Faces | 331521 |
| Patches | 7 |
| Aspect ratio max | 18.89 |
| Non-orthogonality max | 81.9° |
| Bad cells | 0 |

**Note** : La non-orthogonalité maximale de 81.9° est élevée mais acceptable pour ce type de géométrie urbaine complexe. Le solveur converge néanmoins grâce aux paramètres de relaxation adaptés et au maillage Delaunay.

### 5.2 Statistiques champ vitesse

| Paramètre | Valeur |
|-----------|--------|
| U mean | 7.14 m/s |
| U std | 4.49 m/s |
| U min | 0.0 m/s |
| U max | 10.0 m/s |
| TI mean | 0.32 |
| TI max | 1.0 |

### 5.3 Convergence solveur

| Time | U résidu | p résidu | Continuity | k max |
|------|----------|----------|------------|-------|
| 0s | 0.54 | 1.0 | 5.8e21 | 1.5e29 |
| 25s | 0.48 | 0.007 | 1.5e25 | 3.0e24 |
| 99s | 0.008 | 0.003 | 3.2e-5 | 21.6 |
| 133s | 0.002 | 0.005 | 5.0e-6 | 15.1 |

### 5.4 Visualisations

Les figures suivantes sont générées dans `post/visualizations/` :

1. **slice_pedestrian_velocity.png** — Vitesse à 1.75m
2. **wind_comfort_map.png** — Classification NEN
3. **buildings_cp.png** — Coefficient de pression (normalisé par u_ref=10.0)
4. **slice_pedestrian_ti.png** — Intensité de turbulence
5. **slice_vertical_velocity.png** — Coupe verticale vitesse
6. **mesh_wireframe.png** — Maillage
7. **map_view.html** — Carte interactive

**Note** : Le Cp est calculé avec `u_ref = 10.0` hardcodé dans `voxcity_dedicated_postprocess.py`. Pour un calcul plus précis, il faudrait utiliser la valeur de `config.json` ou le champ `U` moyen de l'inlet.

## 6. Fichiers générés

```
neighborhood_demo/
├── test_full_pipeline/
│   ├── constant/polyMesh/    # Maillage OpenFOAM
│   ├── system/               # Paramètres solveur
│   ├── 0/                    # Champs initiaux
│   ├── log.incompressibleFluid  # Log solveur
│   ├── log.checkMesh         # Qualité maillage
│   └── post/
│       ├── visualizations/   # PNG + HTML
│       └── statistics/       # JSON + CSV
└── output/voxcity.h5         # Données VoxCity
```

## 7. Commandes utiles

```bash
cd /home/steven/foampilot/examples/building_geo/neighborhood_demo

# Pipeline complet (avec simulation, défaut mesh_size=6.0, marges auto)
PYTHONPATH=../../foampilot/src:../voxcity_export_work/src:.. python3 run_full_voxcity_pipeline.py \
    --hdf5 output/voxcity.h5 \
    --output-dir test_full_pipeline \
    --fill-gaps

# Pipeline complet (sans simulation)
PYTHONPATH=../../foampilot/src:../voxcity_export_work/src:.. python3 run_full_voxcity_pipeline.py \
    --hdf5 output/voxcity.h5 \
    --output-dir test_full_pipeline \
    --fill-gaps \
    --skip-run

# Post-traitement seul
PYTHONPATH=../../foampilot/src python3 voxcity_dedicated_postprocess.py \
    --case test_full_pipeline \
    --hdf5 output/voxcity.h5 \
    --pedestrian-height 1.75

# Vérifier la qualité du maillage OpenFOAM
checkMesh -case test_full_pipeline
```

## 8. Perspectives

- Ajouter plusieurs directions de vent
- Intégrer terrain DEM réel
- Comparaison avec données météo
- Export PDF automatique via latex_pdf
