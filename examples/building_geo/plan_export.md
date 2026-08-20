# Session Summary — VoxCity / OpenFOAM Neighborhood Demo

## 1. Objectif
Construire un exemple complet et fonctionnel d'urban CFD dans `examples/building_geo/neighborhood_demo/` en utilisant VoxCity, Gmsh, OpenFOAM et foampilot, sans fallback synthétique.

## 2. Fichiers créés / modifiés
- `examples/building_geo/neighborhood_demo/config.json`
- `examples/building_geo/neighborhood_demo/generate.py`
- `examples/building_geo/neighborhood_demo/postprocess.py`
- `examples/building_geo/neighborhood_demo/README.md`
- `examples/building_geo/neighborhood_demo/generate_synthetic.py` (fallback conservé)
- `examples/building_geo/plan_export.md`
- `examples/building_geo/posts/` (5 posts de post-traitement)
- `examples/building_geo/voxcity_postprocess.py`
- `foampilot/src/foampilot/urban/readers/voxcity_reader.py`
- `examples/building_geo/voxcity_export_work/src/vector_builder.py`

## 3. Corrections apportées
### 3.1 VoxCity reader (`voxcity_reader.py`)
- Ajout de `import numpy as np` manquant.
- Correction de `_extract_buildings()` pour reprojeter les footprints WGS84 vers EPSG:32631 avant de tester l'aire, sinon tous les bâtiments étaient rejetés.
- Gestion des `MultiPolygon` et garde-fous sur `ground_z` / `height`.

### 3.2 neighborhood_demo (`generate.py`)
- Suppression du fallback synthétique obligatoire ; l'exemple utilise maintenant VoxCity directement.
- Ajout de `--voxcity-h5` pour charger un HDF5 local sans re-téléchargement.
- Ajout de `--use-cache` pour réutiliser le cache VoxCity quand il existe.
- Chargement HDF5 via `voxcity.io.load_voxcity()` avec reprojection métrique et filtrage par surface projetée.
- Pipeline solver / BCs aligné sur `generate_wind_cases.py` avec profil log-wind `codedFixedValue`.

### 3.3 vector_builder.py
- Ajout de `mesh_constraint="proximity"` avec champ Gmsh `Distance` + `Threshold`.
- Ajout de `analyze_geometry()` pour détecter les bâtiments trop proches / chevauchements.
- Nettoyage automatique après Boolean : `_remove_building_volumes()`, `_remove_debris()`, `removeAllDuplicates()`.
- Meilleure gestion des retours `cut()` / `fragment()` pour retrouver le tag du volume fluide.
- Filtrage des bâtiments dégénérés (`area < mesh_size²`, `height <= 0`).
- Sauvegarde des surfaces bâtiments pour le champ de proximité.

## 4. Résultats
- Le chargement HDF5 fonctionne : 30 bâtiments chargés avec coordonnées métriques valides.
- L'analyse de géométrie détecte correctement les bâtiments très proches/chevauchants dans l'AOI Paris 15e.
- Le sizing par proximité est disponible via `--mesh-constraint proximity`.
- Le maillage 3D échoue encore sur cette géométrie spécifique à cause de chevauchements trop forts après reprojection.

## 5. État actuel
- VoxCity HDF5 chargé correctement.
- Géométrie analysée et problématique identifiée : nombreux bâtiments à distance 0 m et chevauchements XY+Z.
- Prochaine étape nécessaire : nettoyage/pré-traitement de la géométrie VoxCity avant d'envoyer à Gmsh (buffer, simplification, suppression des superpositions).

## 6. Commandes utiles
```bash
cd /home/steven/foampilot/examples/building_geo/neighborhood_demo

# Utiliser un HDF5 local (pas de téléchargement)
PYTHONPATH=../../src:../voxcity_export_work/src:. python3 generate.py --voxcity-h5 output/voxcity.h5 --skip-run --mesh-constraint proximity

# Lancer la simulation via foampilot
PYTHONPATH=../../src:../voxcity_export_work/src:. python3 generate.py --voxcity-h5 output/voxcity.h5

# Post-traiter un cas existant
PYTHONPATH=../../src python3 postprocess.py --case neighborhood_case
```
