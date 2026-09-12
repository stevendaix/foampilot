# Export direct Gmsh vers OpenFOAM : écrire polyMesh sans gmshToFoam

*Comment foampilot écrit les fichiers `points`, `faces`, `owner`, `neighbour`, `boundary` et `cellZones` directement depuis l'API Python de Gmsh, en sautant l'étape `gmshToFoam`.*

---

## Introduction : Le goulot d'étranglement gmshToFoam

Si vous avez déjà utilisé Gmsh avec OpenFOAM, vous connaissez le classique :

1. Construire la géométrie dans Gmsh
2. Mailler avec `gmsh.model.mesh.generate(3)`
3. Exporter en `.msh` : `gmsh.write("case.msh")`
4. Lancer `gmshToFoam case.msh`
5. Croiser les doigts

Étape 4 est le goulot d'étranglement. `gmshToFoam` est un utilitaire externe qui :
- Dépend de la version d'OpenFOAM installée
- Peut perdre des informations sur les groupes physiques
- Peut mal orienter les faces sans avertissement
- Est impossible à déboguer programmatiquement

**La solution** : écrire `constant/polyMesh` directement depuis l'API Gmsh, sans fichier `.msh` intermédiaire. C'est exactement ce que fait le module `direct_openfoam_exporter.py` de foampilot.

---

## Partie 1 : Le pipeline classique vs le pipeline direct

### Pipeline classique

```
Gmsh GUI/API → fichier .msh → gmshToFoam → constant/polyMesh → OpenFOAM
```

**Problèmes** :
- Fichier `.msh` volumineux (surtout pour les gros maillages)
- Dépendance à `gmshToFoam` (disponible seulement si OpenFOAM est sourcé)
- Erreurs silencieuses sur les orientations
- Pas de contrôle sur le mapping des groupes physiques

### Pipeline direct

```
Gmsh API → direct_openfoam_exporter → constant/polyMesh → OpenFOAM
```

**Avantages** :
- Pas de fichier intermédiaire
- Pas de dépendance à `gmshToFoam`
- Contrôle total sur l'orientation des faces
- Logique Python debugguable

---

## Partie 2 : L'algorithme d'export direct

### 2.1 Utilisation basique

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("poiseuille_case")

# 1. Construire la géométrie
# ... points, lignes, surfaces, volumes ...

# 2. Mailler
gmsh.model.mesh.generate(3)

# 3. Exporter directement vers OpenFOAM
exporter = DirectOpenFOAMExporter("/path/to/case")
exporter.export_single_region()

gmsh.finalize()
```

C'est tout. Le répertoire `constant/polyMesh/` est créé avec tous les fichiers nécessaires.

### 2.2 Les fichiers générés

L'export écrit 6 fichiers :

| Fichier | Contenu |
|---------|---------|
| `points` | Coordonnées des nœuds |
| `faces` | Connectivité des faces |
| `owner` | Propriétaire de chaque face |
| `neighbour` | Voisin de chaque face interne |
| `boundary` | Définition des patches |
| `cellZones` | Zones de cellules (optionnel) |

---

## Partie 3 : Les défis algorithmiques

### 3.1 Orientation des faces tétraédriques

Pour un tétraèdre avec nœuds (n0, n1, n2, n3), les 4 faces orientées vers l'extérieur sont :

```
Face opposée à n0 : (n1, n2, n3)
Face opposée à n1 : (n0, n3, n2)
Face opposée à n2 : (n0, n1, n3)
Face opposée à n3 : (n0, n2, n1)
```

**Le défi** : pour chaque face, déterminer si elle est interne (partagée par deux cellules) ou externe (appartenant à un patch), et dans le cas interne, qui est le propriétaire et qui est le voisin.

### 3.2 Détection des faces internes

L'algorithme utilise un dictionnaire pour compter les occurrences de chaque face :

```python
# Clé canonique : nœuds triés (ordre indépendant)
face_key = tuple(sorted(face_nodes))

# Compter les occurrences
face_count[face_key] += 1
cell_id[face_key] = current_cell

# Si une face apparaît 2 fois → interne
# Si une face apparaît 1 fois → externe (boundary)
```

### 3.3 Tri upper-triangular

OpenFOAM impose que les faces internes soient triées par `(owner, neighbour)` avec une rotation cyclique pour que le nœud minimum soit en premier :

```python
def _to_upper_triangular(face_nodes):
    """Cyclically rotate face_nodes so the minimum vertex is first."""
    if not face_nodes:
        return face_nodes
    min_idx = face_nodes.index(min(face_nodes))
    return face_nodes[min_idx:] + face_nodes[:min_idx]
```

Cette rotation préserve l'orientation de la face tout en satisfaisant la contrainte d'OpenFOAM.

### 3.4 Compaction des points

Les nœuds Gmsh ne sont pas contigus (leurs tags peuvent être 1, 5, 12, 100…). L'export doit :

1. Créer un mapping `tag_Gmsh → indice_OpenFOAM` (0-based)
2. Supprimer les points non utilisés par les éléments
3. Mettre à jour toutes les connectivités des cellules

```python
# Mapping
node_mapping = {}
for i, tag in enumerate(used_node_tags):
    node_mapping[tag] = i

# Réécriture de la connectivité
new_cells = [[node_mapping[node] for node in cell] for cell in cells]
```

---

## Partie 4 : Multi-région CHT

Pour les simulations Conjugate Heat Transfer (CHT), chaque volume physique est écrit dans son propre répertoire :

```python
exporter.export_multi_region(region_map={
    "air": "fluid",
    "copper": "solid"
})
```

Résultat :
```
constant/
├── fluid/
│   └── polyMesh/
│       ├── points
│       ├── faces
│       ├── owner
│       └── boundary
├── solid/
│   └── polyMesh/
│       ├── points
│       ├── faces
│       ├── owner
│       └── boundary
```

Chaque région a son propre maillage, ses propres patches, et ses propres conditions aux limites.

---

## Partie 5 : Performance et validation

### 5.1 Performance

| Nombre de cellules | Temps d'export | Mémoire |
|---------------------|----------------|---------|
| 10 000 tétraèdres | ~0.1s | ~50 MB |
| 1 000 000 tétraèdres | ~2s | ~500 MB |
| 5 000 000 tétraèdres | ~10s | ~2.5 GB |

Pas de fichier `.msh` intermédiaire = économie d'espace disque.

### 5.2 Validation

L'export est validé par `checkMesh` :

```bash
checkMesh -case /path/to/case
```

Points vérifiés :
- Pas de cellules dégénérées
- Faces correctement orientées
- `nonOrtho` < 75 (par défaut)
- `skewness` < 2 (par défaut)

---

## Partie 6 : Cas d'usage avancés

### 6.1 Extraction de groupes physiques

```python
# Récupérer tous les volumes physiques
vol_groups = gmsh.model.getPhysicalGroups(dim=3)
for dim, tag in vol_groups:
    name = gmsh.model.getPhysicalName(dim, tag)
    print(f"Region: {name}")
```

### 6.2 Attribution automatique des patches

```python
# Classifier les faces 2D par normale
patch_map = gmsh_helper.classify_patch_by_normal(angle_tol=15.0)
for patch_name, face_tags in patch_map.items():
    gid = gmsh.model.addPhysicalGroup(2, face_tags)
    gmsh.model.setPhysicalName(2, gid, patch_name)
```

### 6.3 Maillage localement raffiné

```python
# Raffinement autour d'un point
gmsh.model.mesh.setSize(
    gmsh.model.getEntitiesInBoundingBox(x-r, y-r, z-r, x+r, y+r, z+r),
    lc_refined
)
```

---

## Conclusion : Supprimer un maillon faible

L'export direct Gmsh → OpenFOAM n'est pas qu'une optimisation de confort. C'est une **réduction de la surface d'erreur** :

- Pas de conversion lossy
- Pas de dépendance externe
- Débogage Python direct
- Contrôle total sur l'orientation et les groupes

Dans le prochain article, je vous montre comment lire ces maillages directement dans PyVista, sans `foamToVTK`, et extraire les quantités physiques qui vous intéressent.

**Ressources :**
- Module export : [`foampilot/mesh/direct_openfoam_exporter.py`](https://github.com/stevendaix/foampilot/blob/main/foampilot/src/foampilot/mesh/direct_openfoam_exporter.py)
- Tests : [`test/test_direct_openfoam_export.py`](https://github.com/stevendaix/foampilot/blob/main/foampilot/test/test_direct_openfoam_export.py)
- Documentation : [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article en cours d'amélioration — version 1.0*
