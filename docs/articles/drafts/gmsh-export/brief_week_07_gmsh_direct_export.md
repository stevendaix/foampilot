# Brief — Semaine 7 : "Export direct Gmsh → OpenFOAM sans gmshToFoam"

**Objectif** : Expliquer comment foampilot écrit `constant/polyMesh` directement depuis l'API Gmsh, en français et en anglais.
**Angle** : Deep-dive technique, algorithmes de maillage, design d'export.

---

## Structure proposée

### Introduction — Le goulot d'étranglement gmshToFoam

Storytelling : Vous avez construit une géométrie complexe dans Gmsh. Vous exportez en `.msh`, puis vous lancez `gmshToFoam`. Ça marche… jusqu'à ce que ça casse. Des volumes dupliqués, des faces mal orientées, des régions qui disparaissent.

**Promesse** : Avec l'export direct de foampilot, vous sautez l'étape `gmshToFoam`. L'API Gmsh écrit directement les fichiers `points`, `faces`, `owner`, `neighbour`, `boundary` et `cellZones`.

### Section 1 — Pourquoi sauter gmshToFoam ?

Problèmes de `gmshToFoam` :
- Dépendance externe à OpenFOAM
- Conversion lossy (perte d'information sur les groupes physiques)
- Erreurs silencieuses sur les orientations de faces
- Impossible à déboguer programmatiquement

### Section 2 — L'algorithme d'export direct

Présenter les étapes :

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("case")

# ... construction géométrie, maillage ...

exporter = DirectOpenFOAMExporter("/path/to/case")
exporter.export_single_region()  # ou export_multi_region()
gmsh.finalize()
```

### Section 3 — Les défis algorithmiques

#### 3.1 Orientation des faces tétraédriques

Pour un tétraèdre (n0, n1, n2, n3), les 4 faces sont :
- Face opposée à n0 : (n1, n2, n3)
- Face opposée à n1 : (n0, n3, n2)
- Face opposée à n2 : (n0, n1, n3)
- Face opposée à n3 : (n0, n2, n1)

**Défi** : déterminer le propriétaire (owner) et le voisin (neighbour) pour chaque face interne.

#### 3.2 Tri upper-triangular

OpenFOAM impose que les faces internes soient triées par `(owner, neighbour)` avec rotation cyclique :

```python
def _to_upper_triangular(face_nodes):
    min_idx = face_nodes.index(min(face_nodes))
    return face_nodes[min_idx:] + face_nodes[:min_idx]
```

#### 3.3 Compaction des points

Les nœuds Gmsh ne sont pas contigus. L'export doit :
- Créer un mapping `tag_Gmsh → indice_OpenFOAM`
- Supprimer les points non utilisés
- Mettre à jour toutes les connectivités

### Section 4 — Multi-région CHT

```python
exporter.export_multi_region(region_map={
    "air": "fluid",
    "copper": "solid"
})
```

Chaque volume physique est écrit dans `constant/<region>/polyMesh/`.

### Section 5 — Performance et validation

- Temps d'export pour 1M de tétraèdres : ~2s
- Mémoire : pas de fichier `.msh` intermédiaire
- Validation : `checkMesh` sur le cas exporté

### Conclusion — Supprimer un maillon faible

CTA : *"Maintenant que vous avez un maillage valide, je vous montre comment le lire directement dans PyVista sans conversion."*

---

## Code à préparer

- [ ] Script Gmsh simple → export direct
- [ ] Script multi-région CHT
- [ ] Extrait de `direct_openfoam_exporter.py` (algorithmes clés)
- [ ] Benchmark performance

## Images à préparer

- [ ] Schéma du pipeline classique (Gmsh → .msh → gmshToFoam → polyMesh)
- [ ] Schéma du pipeline direct (Gmsh → API → polyMesh)
- [ ] Capture d'écran Gmsh avec groupes physiques
- [ ] Capture d'écran `checkMesh` sur cas exporté
- [ ] Graphique de performance (temps vs nombre de cellules)
