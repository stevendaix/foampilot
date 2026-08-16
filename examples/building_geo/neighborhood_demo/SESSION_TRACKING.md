# Suivi de session — Pipeline VoxCity → Gmsh → OpenFOAM

## 1. Objectif initial
Lancer la chaîne de calcul complète pour vérifier que le maillage fonctionne, puis corriger les blocages identifiés.

## 2. État initial du repo
- Branche : `feature/openfoam-direct-export`
- Dernier commit stable : ajout de `DirectOpenFOAMExporter` et tests pytest OK
- Dernière version connue de VoxCity : 1.6.2 (`/home/steven/venv/lib/python3.10/site-packages/voxcity`)
- Fichier de données réel : `examples/building_geo/neighborhood_demo/output/voxcity.h5`

## 3. Hypothèses de départ
- Les footprints VoxCity bruts sont valides
- `gmsh.model.occ.extrude()` suffit pour créer les bâtiments
- Les booléens `cut()` / `fragment()` sont toujours robustes
- L’export OpenFOAM fonctionne sans post-traitement physique

## 4. Blocages constatés
| Symptôme | Étape | Message d’erreur |
|----------|-------|------------------|
| Aucun élément 3D dans le volume fluide | Maillage Gmsh | `No elements in volume 25` |
| Export OpenFOAM échoue | Export | `RuntimeError: No 3-D volume elements found in the Gmsh model.` |
| Physical groups 3-D vides après `generate(3)` | Export | `dim=3 physical groups: []` |
| `Invalid boundary mesh (overlapping facets)` | Maillage 3D | Surfaces 2D superposées après booléens |

## 5. Actions correctrices appliquées

### 5.1 Export OpenFOAM
- **Fichier** : `foampilot/src/foampilot/mesh/direct_openfoam_exporter.py`
- **Ajouté** : `GMSH_PRI = 6`, `GMSH_PYR = 7`
- **Ajouté** : `_GMSH_TO_OPENFOAM_CELL` pour mapping complet
- **Modifié** : `_collect_cells()` lève une erreur explicite si type d’élément 3D inconnu
- **Validation** : `pytest test/test_direct_openfoam_export.py` → 3 passed

### 5.2 Vector Gmsh Builder
- **Fichier** : `examples/building_geo/voxcity_export_work/src/vector_builder.py`
- **`build()`** :
  - mise à jour de `self.fluid_tag` vers le plus grand volume après `fragment()`
  - suppression des volumes bâtiments identifiés par COM
  - nettoyage des debris par seuil de masse relative
- **`assign_patches()`** :
  - sélection du volume fluide par masse (seuil `1e-6 * total_mass`)
  - suppression des anciens physical groups avant recréation
  - création des physical groups 2-D + 3-D **avant** le maillage
- **`build_mesh()`** :
  - `Mesh.ElementOrder = 1`
  - `Mesh.Algorithm3D = 1`
  - `Mesh.AngleToleranceFacetOverlap = 0.5`
  - `removePhysicalGroups()` + `assign_patches()` avant `generate(3)`
- **`export_openfoam()`** :
  - recrée le physical group 3-D `fluid` si Gmsh l’a supprimé
- **`_create_building_volume()`** :
  - `eps_ground = 1.0 m` pour éviter faces coplanaires au sol
- **`_extrude_polygon()`** :
  - extrusion depuis `base_z - eps_ground` jusqu’à `height + eps_ground`

### 5.3 Nettoyage des footprints VoxCity
- **Fichier** : `examples/building_geo/neighborhood_demo/generate.py`
- **Ajouté** : `clean_footprint()`
  - `make_valid(geom)`
  - `buffer(0.0)`
  - arrondi WKT (`rounding_precision=1`)
  - `simplify(0.5, preserve_topology=True)`
  - filtrage par `area < 1.0 m²`
- **Ajouté** : `merge_nearby_buildings()` avec :
  - tri par hauteur (`height_tol = 1.0 m`)
  - fusion seulement si bâtiments de hauteur similaire
  - distance de merge = `1.0 m`
- **Ajouté** : fermeture des gaps après merge
  - `buffer(+0.25 m)` puis `buffer(-0.25 m)`
  - `make_valid()` + `buffer(0.0)` après chaque étape
- **Résultat** : 29 footprints bruts → 29 nettoyés → 7 bâtiments fusionnés

## 6. Fichiers modifiés
| Fichier | Modification |
|---------|-------------|
| `foampilot/src/foampilot/mesh/direct_openfoam_exporter.py` | `_NODES_PER_ELEM` complet + erreur explicite types inconnus |
| `examples/building_geo/voxcity_export_work/src/vector_builder.py` | fluid_tag update, eps_ground=1.0, Mesh.ElementOrder=1, Algorithm3D=1, AngleToleranceFacetOverlap=0.5, export_openfoam recovery |
| `examples/building_geo/neighborhood_demo/generate.py` | clean_footprint(), merge_nearby_buildings() avec height_tol, gap closing |
| `examples/building_geo/neighborhood_demo/config.json` | mesh_size=10.0, margin=50.0 |

## 7. Tests et résultats

### 7.1 Tests unitaires
```bash
cd /home/steven/foampilot/foampilot
PYTHONPATH=src python3 -m pytest test/test_direct_openfoam_export.py -v
# Résultat : 3 passed
```

### 7.2 Tests pipeline
| Configuration | Résultat |
|---------------|----------|
| `mesh_size=15.0`, `--fill-gaps` | ✅ Export OK vers `neighborhood_case/constant/polyMesh` |
| `mesh_size=10.0`, `--fill-gaps` | ✅ Export OK |
| `mesh_size=8.0`, `--fill-gaps`, `margin=100.0` | ❌ `Invalid boundary mesh` |
| `mesh_size=6.0`, sans `--fill-gaps` | ❌ `Invalid boundary mesh` |
| 11 bâtiments fusionnés (distance=0.5 m) | ❌ `Invalid boundary mesh` |
| 7 bâtiments fusionnés (distance=1.0 m) + gap closing | ❌ `Invalid boundary mesh` |

## 8. Images générées
- `footprint_processing_steps.png` : 3 panneaux (raw → cleaned → merged)
- Chemin : `examples/building_geo/neighborhood_demo/footprint_processing_steps.png`

## 9. Diagnostic actuel
Le blocage principal est maintenant **géométrique** :
- `Invalid boundary mesh (overlapping facets)` sur des surfaces spécifiques (ex: 61 et 62)
- Ces surfaces correspondent à des interfaces entre bâtiments fusionnés
- Même avec `eps_ground=1.0 m` et `distance=1.0 m`, des faces restent quasi-coplanaires

## 10. Pistes d’amélioration
1. **Augmenter `eps_ground`** à `2.0 m` pour découper plus proprement le sol
2. **Fermer les gaps** avec un buffer plus agressif (`gap = 1.0 m`)
3. **Utiliser `fuse()`** au lieu de `fragment()` pour les bâtiments fusionnés
4. **Tester sans building_tags** : valider que le fluide seul maillage bien
5. **Ajouter `Mesh.Optimize`** + `Mesh.OptimizeNetgen` pour lisser le maillage 2D
6. **Exporter en `.msh` v4** pour conserver les physical groups

## 11. Prochaine étape recommandée
Tester la pipeline avec **7 bâtiments fusionnés** + **`eps_ground=2.0 m`** + **`gap=1.0 m`** et mesurer l’impact sur les facettes superposées.
