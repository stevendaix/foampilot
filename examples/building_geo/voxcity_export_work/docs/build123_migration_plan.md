# Plan de migration vers build123d pour la construction géométrique VoxCity

## Contexte et problème

### État actuel

Le pipeline VoxCity construit la géométrie directement via l'API OCC de Gmsh (`gmsh.model.occ`). Cette approche a révélé plusieurs problèmes sur le cas VoxCity :

1. **Booléens instables** : les opérations `cut()` successives peuvent laisser le volume fluide original en place, le résultat de la coupe étant traité comme un fragment et supprimé
2. **Perte d'information** : `gmsh.write()` ne conserve pas les noms des physical groups, imposant une reclassification après maillage
3. **Topologie complexe** : les Boolean cuts créent des nœuds géométriquement confondus et des fragments difficiles à gérer

### Diagnostic

- Le maillage Gmsh lui-même est sain : 0 tétraèdre dupliqué, 0 face avec >2 occurrences
- L'export OpenFOAM casse l'orientation : `_compact_points()` scramble les coordonnées après compaction
- La soustraction des bâtiments n'est pas visible dans le maillage final : pas de surfaces `buildings` dans la bbox des bâtiments

## Solution proposée

**Remplacer la construction géométrique OCC/Gmsh par build123d**, avec transfert vers Gmsh via BREP temporaire.

### Pourquoi build123d

- Déjà dépendance du projet (`build123d 0.10.0`)
- API plus stable pour les opérations booléennes complexes
- Même moteur OpenCASCADE que Gmsh, donc compatibilité garantie
- Meilleure gestion des `cut()` successifs

### Pourquoi BREP temporaire

- Format natif OpenCASCADE, pas un format neutre d'échange
- Conserve la topologie exacte OCC
- Transfert simple et déterministe entre build123d et Gmsh
- Complètement transparent pour le reste du pipeline

## Architecture cible

```
UrbanModel + CFDTerrain
        │
        ▼
┌──────────────────────────────────────┐
│  build123_geometry.py                │
│  - build_fluid_solid()               │
│  - retourne un build123d.Solid       │
└──────────────────────────────────────┘
        │
        │ Solid
        ▼
┌──────────────────────────────────────┐
│  vector_builder.py                   │
│  - export BREP temporaire            │
│  - import dans Gmsh                  │
│  - classification des patches        │
│  - maillage                          │
│  - export OpenFOAM                   │
└──────────────────────────────────────┘
        │
        ▼
   DirectOpenFOAMExporter
```

## Points clés

### 1. build123d ne connaît pas Gmsh

Le module de construction géométrique ne dépend que de :
- `UrbanModel`
- `CFDTerrain`
- `build123d`

Il retourne un `Solid` prêt à être importé.

### 2. Le BREP est un détail d'implémentation

Le fichier BREP temporaire est créé et détruit dans `vector_builder.py`. Le reste du code ne sait pas qu'il existe.

### 3. Les physical groups sont créés dans Gmsh

Contrairement à l'approche actuelle qui essaie de préserver les noms à travers `gmsh.write()`, la nouvelle approche :
- Importe la géométrie dans Gmsh
- Génère le maillage 2D
- Classifie les surfaces par centroïde
- Crée les physical groups dans Gmsh
- Le `DirectOpenFOAMExporter` lit ces groups directement

### 4. Validation immédiate après import

Après `gmsh.model.occ.importShapes()` :
- Vérifier qu'il y a exactement 1 volume fluide
- Vérifier la bounding box
- Vérifier le volume
- Arrêter si anomalie

## Implémentation

### Étape 1 : Module de géométrie build123d

**Fichier** : `examples/building_geo/voxcity_export_work/src/build123_geometry.py`

```python
def build_fluid_solid(urban: UrbanModel, 
                     terrain: CFDTerrain,
                     margin: float,
                     bottom_offset: float) -> Solid:
    """Construit le domaine fluide avec bâtiments soustraits.
    
    Returns:
        build123d.Solid du domaine fluide
    """
```

Responsabilités :
- Créer la boîte fluide globale
- Pour chaque bâtiment, créer une boîte alignée sur le footprint
- Effectuer les `cut()` successifs
- Retourner le solide fluide final

### Étape 2 : Importeur Gmsh

**Fichier** : `examples/building_geo/voxcity_export_work/src/gmsh_geometry_importer.py`

```python
class GmshGeometryImporter:
    def import_build123d(self, solid: Solid) -> Path:
        """Importe un solide build123d dans Gmsh via BREP temporaire."""
        
    def cleanup(self):
        """Supprime le fichier BREP temporaire."""
```

Responsabilités :
- Exporter le solide vers un fichier BREP temporaire
- Importer dans Gmsh avec `gmsh.model.occ.importShapes(..., format="brep")`
- Synchroniser le modèle OCC
- Vérifier la validité du volume importé

### Étape 3 : Modification de vector_builder.py

Modifier `VectorGmshBuilder` pour :
- Utiliser `build_fluid_solid()` au lieu de `_create_building_volume()`
- Utiliser `GmshGeometryImporter` au lieu de `gmsh.model.occ.addBox()`
- Garder `assign_patches()` et `build_mesh()` existants
- Supprimer `_extrude_polygon()`, `_identify_building_volumes()`, `_remove_debris()`, `_merge_building_volumes()`

### Étape 4 : Tests de validation

1. **Test unitaire** : 1 bâtiment simple
   - Vérifier volume fluide
   - Vérifier nombre de surfaces
   - Vérifier classification des patches

2. **Test VoxCity complet** : 22 bâtiments
   - Vérifier volume fluide
   - Vérifier présence du patch `buildings`
   - Lancer `checkMesh`
   - Lancer une simulation complète

## Fichiers modifiés

| Fichier | Action |
|---------|--------|
| `voxcity_export_work/src/build123_geometry.py` | **Créer** |
| `voxcity_export_work/src/gmsh_geometry_importer.py` | **Créer** |
| `voxcity_export_work/src/vector_builder.py` | **Modifier** |
| `foampilot/src/foampilot/mesh/direct_openfoam_exporter.py` | **Aucune modification** |
| `neighborhood_demo/generate.py` | **Aucune modification** |

## Points de vigilance

### Performance

- Export/import BREP : quelques ms pour 22 bâtiments
- Pas d'impact mesurable sur le temps total du pipeline

### Précision

- BREP conserve la précision OCC
- Vérifier `gmsh.model.occ.synchronize()` sans erreur
- Comparer les bounding boxes avant/après import

### Maillage

- Gmsh maillage un volume importé depuis build123 peut avoir une topologie différente
- Valider `checkMesh` systématiquement
- Comparer le nombre de cellules avec la version actuelle

### Gestion des erreurs

- Si `importShapes` échoue, logger le fichier BREP pour diagnostic
- Si le volume importé n'est pas unique, afficher les bounding boxes de tous les volumes
- Ne pas masquer les erreurs OCC

## Validation

### Critères de succès

1. ✅ `checkMesh` passe sur le cas VoxCity complet
2. ✅ Patch `buildings` présent dans `constant/polyMesh/boundary`
3. ✅ Simulation démarre sans erreur de maillage
4. ✅ Nombre de cellules cohérent avec la géométrie

### Tests à faire

```bash
# Test unitaire
PYTHONPATH=src python3 -c "
from build123_geometry import build_fluid_solid
from urban_model import UrbanModel
from terrain import CFDTerrain
# ... 1 bâtiment simple
"

# Test complet
PYTHONPATH=src python3 generate.py --voxcity-h5 output/voxcity.h5
checkMesh -case neighborhood_case
```

## Planning

1. **Jour 1** : Créer `build123_geometry.py` avec cas minimal 1 bâtiment
2. **Jour 1** : Créer `gmsh_geometry_importer.py` et valider l'import
3. **Jour 2** : Intégrer dans `vector_builder.py`
4. **Jour 2** : Valider sur cas VoxCity complet
5. **Jour 3** : Nettoyage du code mort et documentation

## Notes techniques

### BREP vs STEP

- **BREP** : format natif OCC, conserve la topologie exacte, utilisé ici
- **STEP** : format d'échange ISO, peut lisser la topologie, évité

### build123d API

```python
from build123d import *

# Boîte fluide
with BuildPart() as builder:
    Box(dx, dy, dz)
    part = builder.part

# Bâtiment
with BuildPart() as bldg:
    Box(bdx, bdy, bh)
    bldg_part = bldg.part

# Soustraction
fluid = fluid.cut(bldg_part)
```

### Gmsh import

```python
gmsh.model.occ.importShapes(brep_path, format="brep", highestDimOnly=True)
gmsh.model.occ.synchronize()
```

### Classification des patches

Utiliser les centroïdes des éléments de maillage 2D, pas les centroïdes géométriques OCC :

```python
for surf_tag in gmsh.model.getEntities(2):
    etypes, enodes = gmsh.model.mesh.getElements(2, surf_tag)
    # Calculer COM à partir des nœuds maillés
    # Classifier par rapport aux bornes du domaine
    # Créer physical group
```
