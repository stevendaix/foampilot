# Blocage de la géométrie VoxCity → Gmsh → OpenFOAM

## 1. Contexte

La pipeline `generate.py` construit un cas CFD urbain à partir du fichier VoxCity HDF5 `output/voxcity.h5`. Le flux nominal est :

```
VoxCity HDF5 → load_voxcity() → building_gdf (EPSG:4326)
  ↓ projection EPSG:32631 via pyproj
UrbanModel (30 bâtiments, footprints individuels)
  ↓ VectorGmshBuilder.build()
Gmsh OCC : boîte fluide + 23 bâtiments extrudés
  ↓ Boolean fragment/cut
Volume fluide avec bâtiments découpés
  ↓ gmsh.model.mesh.generate(3)
Maillage tétraédrique
  ↓ DirectOpenFOAMExporter.export_single_region()
constant/polyMesh/
```

## 2. Symptôme observé

```
Info    : 3D Meshing 1 volume with 1 connected component
Error   : No elements in volume 25
Exception: No elements in volume 25
```

Puis à l'export :

```
RuntimeError: No 3-D volume elements found in the Gmsh model.
```

## 3. Points de blocage identifiés

### 3.1 Gmsh 3D meshing échoue sur le volume fluide tag=25

Le volume fluide est correctement identifié (tag=25, plus petit que le domaine initial tag=1), mais `gmsh.model.mesh.generate(3)` ne produit aucun élément tétraédrique dans ce volume.

Causes possibles :
- Géométrie non-manifold ou dégénérée après Boolean `fragment()`
- Présence de "debris volumes" (volumes résiduels < 1e-6 × volume domaine)
- Face mal orientée ou arête coincidente
- `healShapes()` ne suffit pas à réparer la topologie

### 3.2 `_NODES_PER_ELEM` dans `direct_openfoam_exporter.py` ne connaît pas les types d'éléments 3D Gmsh

```python
# direct_openfoam_exporter.py:40-45
_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
}
```

**Manquent :**
- `GMSH_PRI = 6` (prisme à base triangulaire)
- `GMSH_PYR = 5` (pyramide)
- `GMSH_TET_10 = 10` (tétraèdre d'ordre 2, si maillage raffiné)

Si Gmsh produit un `GMSH_PRI` (code 6) ou `GMSH_PYR` (code 7), le dict retourne `0` et l'élément est silencieusement ignoré. Cela peut expliquer `RuntimeError: No 3-D volume elements found` quand **tous** les éléments 3D produits sont de type non reconnu.

**Impact réel observé :** avec `mesh_size=6.0` et `Mesh.Algorithm3D=1`, Gmsh réussit parfois le maillage 3D mais produit 0 élément dans le volume fluide. Le diagnostic `mesh.getElements(3, tag)` sur chaque volume montre 0 élément après `generate(3)`.

### 3.3 `assign_patches()` ajoute un physical group 3-D "fluid" trop large

```python
# vector_builder.py:533-536
fluid_volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for _, tag in fluid_volumes]
if fluid_tags:
    gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")
```

Si `gmsh.model.getEntities(dim=3)` retourne plusieurs volumes après Boolean (fluide + debris), le physical group "fluid" peut contenir des volumes non maillables.

## 4. Diagnostic détaillé

### 4.1 Vérification des footprints VoxCity brutes

```python
# Résultat obtenu avec check_footprints.py
v0_0: area=0.00, height=27.0, type=Polygon
v1_1: area=0.00, height=11.0, type=Polygon
...
v28_29: area=0.00, height=22.9, type=Polygon
```

**Toutes les aires sont 0.00 en EPSG:4326** car les coordonnées sont en degrés (x≈2.32°, y≈48.85°). C'est normal : l'aire en degrés carrés n'a pas de sens physique.

Après projection EPSG:32631 :
```
Original area: 0.000000
Projected area: 420.16 m²
```

Donc la condition `area_m2 < 1.0` dans `generate.py:120` **ne filtre rien** car `area_m2` est calculé sur la géométrie projetée.

### 4.2 État OCC après `build()`

```python
# diagnostic.py (à exécuter)
vols = gmsh.model.getEntities(dim=3)
# Résultat attendu ~7 volumes : 1 fluide + debris
# Le volume fluide est le plus grand par bbox
```

### 4.3 État après `healShapes()`

```python
gmsh.model.occ.healShapes()
gmsh.model.occ.synchronize()
# Fusionne les arêtes/faces coincidentes
# Peut modifier le tag des entités
```

**Attention :** `healShapes()` peut changer les tags des entités. Le physical group 3-D "fluid" créé dans `assign_patches()` peut se retrouver avec des tags invalides.

## 5. Tentatives de correction

### 5.1 Passage de `cut()` séquentiel à `fragment()` puis suppression building_volumes

**Fichier :** `vector_builder.py:125-175`

```python
# AVANT : cut séquentiel (échoue avec "Unknown model region")
fluid_volume = [(3, self.fluid_tag)]
for btag in list(self.building_tags):
    result, _ = gmsh.model.occ.cut(fluid_volume, [(3, btag)], ...)

# APRÈS : fragment puis identification COM
fluid_box = (3, self.fluid_tag)
tools = [(3, t) for t in self.building_tags]
gmsh.model.occ.fragment([fluid_box] + tools, [])
# Puis suppression des volumes bâtiment par COM
building_volumes_to_remove = self._identify_building_volumes(all_vols)
```

**Résultat :** fonctionne pour produire la géométrie, mais 3D meshing échoue toujours.

### 5.2 Nettoyage debris + sélection volume fluide par taille

```python
# vector_builder.py:168-185
volumes_to_remove = [tag for dim, tag in remaining_vols if tag != self.fluid_tag]
if volumes_to_remove:
    gmsh.model.occ.remove([(3, t) for t in volumes_to_remove])
```

**Résultat :** réduit le nombre de volumes de 7 à 1 (fluide seul). Mais `generate(3)` échoue toujours.

### 5.3 `healShapes()` + fallback `Mesh.Algorithm3D=1`

```python
# vector_builder.py:584-603
gmsh.model.occ.healShapes()
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
except Exception:
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
```

**Résultat partiel :** avec `mesh_size=10.0` dans le test isolé, le maillage 3D réussit (6165 nœuds, 14315 éléments). Mais dans la pipeline complète avec `mesh_size=6.0`, échec.

### 5.4 Augmentation `mesh_size` à 10.0

```bash
python3 generate.py --voxcity-h5 output/voxcity.h5 --skip-run --fill-gaps --mesh-size 10.0
```

**Résultat :** 3D meshing réussit, mais l'export OpenFOAM échoue car le physical group "fluid" est corrompu par `healShapes()`.

## 6. Cause racine identifiée

### 6.1 `healShapes()` invalide les physical groups

Après `healShapes()`, `gmsh.model.getPhysicalGroups(dim=3)` retourne des paires `(dim, tag)` où `tag` ne correspond plus aux entités existantes. L'export OpenFOAM cherche alors des éléments dans des entités fantômes → `RuntimeError: No 3-D volume elements found`.

**Preuve :** dans `diagnostic.py`, comparer les entités avant/après `healShapes()`.

### 6.2 Types d'éléments 3D manquants dans `_NODES_PER_ELEM`

Même quand Gmsh maille correctement, si le raffinement produit des pyramides (`GMSH_PYR=7`) ou des prismes (`GMSH_PRI=6`), l'exporteur OpenFOAM les ignore silencieusement.

## 7. Code incriminé

### 7.1 `vector_builder.py` — `assign_patches()` ligne 533-536

```python
fluid_volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for _, tag in fluid_volumes]
if fluid_tags:
    gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")
```

**Problème :** ajoute tous les volumes OCC, y compris les debris, dans le physical group "fluid".

### 7.2 `direct_openfoam_exporter.py` — `_NODES_PER_ELEM` ligne 40-45

```python
_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
}
```

**Problème :** ne couvre pas `GMSH_PRI=6`, `GMSH_PYR=7`, `GMSH_TET_10=10`.

### 7.3 `direct_openfoam_exporter.py` — `_collect_cells()` ligne 380

```python
npp = _NODES_PER_ELEM.get(etype, 0)
if npp == 0:
    continue  # SILENCE l'élément non reconnu
```

## 8. Plan de résorption

### Étape 1 — Réparer `_NODES_PER_ELEM`

Ajouter les types manquants :

```python
_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
    GMSH_PRI: 6,
    GMSH_PYR: 5,
}
```

### Étape 2 — Reconstruire les physical groups après `healShapes()`

Dans `build_mesh()`, appeler `assign_patches()` **après** `healShapes()` et `generate(3)`, ou stocker les entités fluide avant `healShapes()` et les re-sélectionner après.

### Étape 3 — Filtrer les debris avant `assign_patches()`

```python
# Dans assign_patches(), ne garder que le volume fluide principal
all_vols = gmsh.model.getEntities(dim=3)
fluid_vol = max(all_vols, key=lambda v: bbox_volume(v))
fluid_tags = [fluid_vol[1]]
```

### Étape 4 — Tester avec `mesh_size` progressif

Commencer avec `mesh_size=15.0` pour valider la pipeline, puis réduire.

## 9. Commandes de diagnostic

```bash
# Lancer le diagnostic complet
python3 diagnostic.py

# Vérifier les footprints avant builder
python3 check_footprints.py

# Tester healShapes seul
python3 -c "
import gmsh; gmsh.initialize()
gmsh.open('output/case.msh')
gmsh.model.occ.healShapes()
gmsh.model.occ.synchronize()
print('Physical groups 3D:', gmsh.model.getPhysicalGroups(dim=3))
gmsh.finalize()
"
```

## 10. Fichiers modifiés dans cette session

| Fichier | Modification |
|---------|-------------|
| `voxcity_export_work/src/vector_builder.py` | `fragment()` + debris removal + `healShapes()` + fallback `Mesh.Algorithm3D=1` |
| `neighborhood_demo/generate.py` | correction `sys.path` |
| `neighborhood_demo/config.json` | `mesh_size: 6.0` |
| `foampilot/src/foampilot/mesh/direct_openfoam_exporter.py` | non modifié (blocage ici aussi) |

## 11. Prochaine action recommandée

Corriger `_NODES_PER_ELEM` dans `direct_openfoam_exporter.py`, puis relancer `python3 generate.py --voxcity-h5 output/voxcity.h5` sans `--skip-run`.




réponse : 
Ton diagnostic est globalement correct, mais il faut bien séparer **deux défaillances différentes** qui se cumulent :

1. **Gmsh n’arrive pas à mailler le volume fluide**  
   → erreur `No elements in volume 25`.

2. **Quand le maillage 3D existe, l’export OpenFOAM peut échouer silencieusement**  
   → types d’éléments 3D non reconnus, physical group invalide, ou confusion entre tag d’entité et tag de physical group.

La correction seulement de `_NODES_PER_ELEM` est nécessaire, mais **pas suffisante** si `mesh_size=6.0` échoue encore côté Gmsh.

---

# 1. Cause probable du `No elements in volume 25`

Le message :

```text
Info    : 3D Meshing 1 volume with 1 connected component
Error   : No elements in volume 25
```

signifie que Gmsh tente bien de mailler un volume, mais n’arrive pas à produire de tétraèdres dedans.

Les causes les plus fréquentes dans ton workflow sont :

- géométrie OCC fragile après `fragment()` / `cut()` ;
- bâtiments qui se touchent, se chevauchent ou touchent les limites du domaine ;
- faces coplanaires, en particulier le fond des bâtiments à `z=0` qui coïncide avec le sol du domaine fluide ;
- petits volumes résiduels / debris après booléens ;
- physical group 3D qui contient un volume non maillable ;
- `healShapes()` qui casse ou déplace les tags, surtout si les physical groups ont été créés avant.

Le fait que `mesh_size=10.0` passe parfois mais pas `mesh_size=6.0` indique aussi que la géométrie contient probablement des petites arêtes, petits angles ou interfaces degenerées qui deviennent bloquantes quand le maillage est plus fin.

---

# 2. Correction immédiate de l’exporteur OpenFOAM

Dans Gmsh, les codes d’éléments sont :

| Code Gmsh | Type | Nombre de nœuds | Équivalent OpenFOAM |
|---:|---|---:|---|
| 2 | triangle | 3 | face |
| 3 | quadrangle | 4 | face |
| 4 | tetraèdre | 4 | `tet` |
| 5 | hexaèdre | 8 | `hex` |
| 6 | prisme triangulaire | 6 | `wedge` ou `prism` selon exporteur |
| 7 | pyramide | 5 | `pyr` |
| 10 | tétraèdre quadratique | 10 | à éviter / convertir |

Donc il faut au minimum :

```python
# direct_openfoam_exporter.py

GMSH_TRI = 2
GMSH_QUAD = 3
GMSH_TET = 4
GMSH_HEX = 5
GMSH_PRI = 6
GMSH_PYR = 7

_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
    GMSH_PRI: 6,
    GMSH_PYR: 5,
}

_GMSH_TO_OPENFOAM_CELL = {
    GMSH_TET: "tet",
    GMSH_HEX: "hex",
    GMSH_PRI: "wedge",   # ou "prism" selon ton exporteur
    GMSH_PYR: "pyr",
}
```

Attention à ne pas écrire :

```python
GMSH_PYR: 5
```

si `GMSH_PYR` n’est pas défini comme le code Gmsh `7`. La clé doit être le **type d’élément Gmsh**, pas le nombre de nœuds.

Donc :

```python
GMSH_PYR = 7

_NODES_PER_ELEM = {
    ...,
    GMSH_PYR: 5,
}
```

---

## 2.1 Ne plus ignorer silencieusement les éléments inconnus

Dans `_collect_cells()`, il faut éviter le `continue` silencieux.

Par exemple :

```python
unknown_types = {}

for etype, etags, ntags in zip(element_types, element_tags, node_tags):
    npp = _NODES_PER_ELEM.get(etype)

    if npp is None:
        unknown_types[etype] = unknown_types.get(etype, 0) + len(etags)
        continue

    # traitement normal...

if not cells and unknown_types:
    raise RuntimeError(
        "No known 3D cells found. Unknown Gmsh element types: "
        f"{unknown_types}. "
        "Check _NODES_PER_ELEM and/or force Mesh.ElementOrder=1."
    )
```

Sinon tu peux avoir un maillage Gmsh correct mais un export OpenFOAM vide.

---

## 2.2 Forcer un maillage linéaire

Pour OpenFOAM, le plus robuste est de commencer avec des éléments linéaires uniquement.

Avant `gmsh.model.mesh.generate(3)` :

```python
gmsh.option.setNumber("Mesh.ElementOrder", 1)
```

Cela évite les `GMSH_TET_10`, `GMSH_HEX_27`, etc.

Si tu as déjà un maillage quadratique, tu peux aussi essayer :

```python
gmsh.model.mesh.setOrder(1)
```

avant l’export OpenFOAM.

---

# 3. Correction des physical groups Gmsh

Tu as identifié un point très important :

```python
fluid_volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for _, tag in fluid_volumes]
if fluid_tags:
    gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")
```

C’est dangereux si `getEntities(dim=3)` retourne encore :

- des volumes bâtiments,
- des debris,
- des volumes résiduels,
- des volumes non maillés,
- ou des entités dont les tags ont changé après `healShapes()`.

---

## 3.1 Règle d’or

Il faut créer les physical groups :

1. **après** toutes les opérations booléennes,
2. **après** `healShapes()`,
3. **après** suppression des volumes parasites,
4. **avant** `gmsh.model.mesh.generate(3)`.

Mauvais ordre :

```python
assign_patches()
gmsh.model.occ.healShapes()
gmsh.model.mesh.generate(3)
```

Ordre recommandé :

```python
# Booléens
# Nettoyage des volumes
gmsh.model.occ.healShapes(...)
gmsh.model.occ.synchronize()

# Recréer les physical groups sur les entités courantes
assign_patches()

# Maillage
gmsh.model.mesh.generate(3)
```

---

## 3.2 Supprimer les anciens physical groups

Avant de recréer les groupes :

```python
gmsh.model.removePhysicalGroups()
```

Puis recréer :

```python
fluid_tag = ...  # tag courant du volume fluide
pg = gmsh.model.addPhysicalGroup(3, [fluid_tag])
gmsh.model.setPhysicalName(3, pg, "fluid")
```

---

## 3.3 Ne pas confondre tag d’entité et tag de physical group

Dans l’exporteur, vérifie que tu ne fais pas :

```python
gmsh.model.mesh.getElements(3, fluid_physical_tag)
```

si `fluid_physical_tag` est le tag du **physical group**.

`gmsh.model.mesh.getElements(dim, tag)` attend normalement un tag d’**entité géométrique**, pas un tag de physical group.

Pour récupérer les éléments d’un physical group, il faut plutôt faire :

```python
element_types, element_tags, node_tags = (
    gmsh.model.mesh.getElementsForPhysicalGroup(3, fluid_physical_tag)
)
```

ou alors :

```python
entities = gmsh.model.getEntitiesForPhysicalGroup(3, fluid_physical_tag)

all_element_types = []
all_element_tags = []
all_node_tags = []

for dim, tag in entities:
    etypes, etags, ntags = gmsh.model.mesh.getElements(dim, tag)
    all_element_types.extend(etypes)
    all_element_tags.extend(etags)
    all_node_tags.extend(ntags)
```

Si ton exporteur confond les deux, il peut très bien trouver `0` élément même si le maillage existe.

---

# 4. Sélection robuste du volume fluide

Plutôt que de prendre tous les volumes :

```python
fluid_volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for _, tag in fluid_volumes]
```

il faut sélectionner les volumes utiles par masse/volume.

Exemple :

```python
def _remove_small_volumes(min_rel_volume=1e-6):
    volumes = gmsh.model.getEntities(dim=3)

    if not volumes:
        raise RuntimeError("No 3D volume found after boolean operations.")

    masses = []
    for dim, tag in volumes:
        mass = abs(gmsh.model.occ.getMass(dim, tag))
        masses.append((tag, mass))

    total_mass = sum(m for _, m in masses)

    if total_mass <= 0.0:
        raise RuntimeError("All 3D volumes have zero mass.")

    threshold = min_rel_volume * total_mass

    keep_tags = [
        tag for tag, mass in masses
        if mass >= threshold
    ]

    if not keep_tags:
        # Si tout est minuscule, garder au moins le plus grand
        keep_tags = [max(masses, key=lambda x: x[1])[0]]

    remove_tags = [
        (3, tag) for tag, _ in masses
        if tag not in keep_tags
    ]

    if remove_tags:
        gmsh.model.occ.remove(remove_tags, recursive=False)
        gmsh.model.occ.synchronize()

    return keep_tags
```

Ensuite :

```python
fluid_tags = _remove_small_volumes(min_rel_volume=1e-6)

gmsh.model.removePhysicalGroups()
pg = gmsh.model.addPhysicalGroup(3, fluid_tags)
gmsh.model.setPhysicalName(3, pg, "fluid")
```

---

# 5. Booléens : préférer un `cut()` groupé ou nettoyer proprement

Le passage de `cut()` séquentiel à `fragment()` est compréhensible, mais `fragment()` peut laisser des morceaux ambiguës si tu ne supprimes pas correctement les volumes bâtiments.

Pour un cas urbain, je recommande plutôt :

1. fusionner les bâtiments entre eux si leurs footprints se touchent ou se chevauchent ;
2. faire un seul `cut()` du domaine fluide par l’ensemble des bâtiments ;
3. supprimer les outils.

Exemple :

```python
fluid = [(3, self.fluid_tag)]
tools = [(3, tag) for tag in self.building_tags]

if len(tools) > 1:
    # Option robuste si les bâtiments peuvent se toucher/chevaucher.
    # On perd l'individualité des bâtiments, mais la géométrie devient plus stable.
    tools, _ = gmsh.model.occ.fuse(
        [tools[0]],
        tools[1:],
        removeObject=True,
        removeTool=True
    )

result, _ = gmsh.model.occ.cut(
    fluid,
    tools,
    removeObject=True,
    removeTool=True
)

gmsh.model.occ.synchronize()

fluid_tags = [tag for dim, tag in result if dim == 3]

if not fluid_tags:
    raise RuntimeError("Boolean cut removed the whole fluid domain.")
```

Si tu as besoin de garder des patches individuels par bâtiment, cette fusion peut être gênante. Mais pour diagnostiquer le blocage maillage, c’est très utile.

---

# 6. Éviter les faces coplanaires au sol

Un problème classique : les bâtiments sont extrudés depuis `z=0`, alors que le domaine fluide commence aussi à `z=0`.

Donc la face inférieure du bâtiment est coplanaire avec la face inférieure du domaine fluide.

Cela peut créer des interfaces dégénérées après booléen.

Solution simple : faire dépasser très légèrement les bâtiments sous le sol.

Par exemple :

```python
eps_ground = 0.1  # m

# Au lieu d'extruder de z=0 à height,
# extruder de z=-eps_ground à height.
```

Concrètement, si tu extrudes une surface avec Gmsh :

```python
vol_list = gmsh.model.occ.extrude(
    [(2, surface_tag)],
    0.0, 0.0, height + eps_ground
)

building_dim, building_tag = vol_list[0]

gmsh.model.occ.translate(
    [(3, building_tag)],
    0.0, 0.0, -eps_ground
)
```

Le bâtiment intersecte alors légèrement le dessous du domaine fluide. Physiquement, ça ne change rien, mais numériquement c’est souvent beaucoup plus robuste.

---

# 7. Vérifier que les bâtiments sont bien à l’intérieur du domaine

Si un bâtiment touche ou dépasse une frontière du domaine fluide, les booléens peuvent produire des volumes non manifold.

Il faut vérifier :

```python
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(3, fluid_tag)
```

puis comparer avec la boîte des bâtiments.

Tu dois avoir une marge :

- en `x`,
- en `y`,
- surtout en `z`.

Par exemple :

```python
domain_height = max_building_height + 5 * mesh_size
```

ou au minimum :

```python
domain_height = max_building_height + 20.0
```

Si un bâtiment touche le haut du domaine, le maillage 3D peut échouer.

---

# 8. Nettoyer les footprints avant extrusion

Avant d’envoyer les polygones dans Gmsh, il faut les nettoyer avec Shapely.

Par exemple :

```python
from shapely.validation import make_valid
from shapely.ops import unary_union

def clean_footprint(geom, simplify_tol=0.05):
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)
    geom = geom.buffer(0.0)

    if geom.is_empty:
        return None

    # Simplification légère pour supprimer les micro-arêtes
    if simplify_tol > 0.0:
        geom = geom.simplify(simplify_tol, preserve_topology=True)
        geom = make_valid(geom)
        geom = geom.buffer(0.0)

    if geom.is_empty:
        return None

    return geom
```

Puis filtrer :

```python
geom = clean_footprint(geom, simplify_tol=0.05)

if geom is None or geom.is_empty:
    continue

if geom.area < 1.0:  # après projection EPSG:32631, en m²
    continue
```

Attention : l’aire en EPSG:4326 est effectivement sans signification en m². Le filtre doit être fait **après projection**.

---

# 9. Options Gmsh recommandées pour stabiliser le maillage

Avant le maillage :

```python
gmsh.option.setNumber("General.Verbosity", 99)

gmsh.option.setNumber("Mesh.ElementOrder", 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max(0.05, mesh_size * 0.2))
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

# Commencer simple : pas de recombination 3D
gmsh.option.setNumber("Mesh.RecombineAll", 0)

# Algorithme 3D robuste
gmsh.option.setNumber("Mesh.Algorithm3D", 1)

gmsh.option.setNumber("Mesh.Optimize", 1)
```

Si le maillage échoue encore, tu peux essayer plusieurs algorithmes :

```python
def generate_3d_with_fallback():
    last_error = None

    # 10 = HXT, souvent robuste selon les versions Gmsh
    # 1  = Delaunay
    # 4  = Frontal
    for algo in (10, 1, 4):
        try:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm3D", algo)

            print(f"Trying 3D mesh algorithm: {algo}")

            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.generate(3)

            print(f"3D mesh succeeded with algorithm {algo}")
            return

        except Exception as exc:
            last_error = exc
            print(f"3D mesh failed with algorithm {algo}: {exc}")

    raise RuntimeError(f"All 3D meshing attempts failed: {last_error}")
```

---

# 10. Vérifier le contenu réel du maillage après `generate(3)`

Ajoute ce diagnostic juste après le maillage :

```python
def check_3d_mesh():
    volumes = gmsh.model.getEntities(dim=3)

    if not volumes:
        raise RuntimeError("No 3D volume in model after mesh generation.")

    for dim, tag in volumes:
        element_types, element_tags, node_tags = (
            gmsh.model.mesh.getElements(dim, tag)
        )

        n_elements = sum(len(tags) for tags in element_tags)

        print(
            f"Volume {tag}: {n_elements} elements, "
            f"types={element_types}"
        )

        if n_elements == 0:
            raise RuntimeError(f"No 3D elements in volume {tag}.")
```

Si tu vois :

```text
Volume 25: 0 elements, types=[]
```

le problème est bien côté maillage/géométrie, pas encore côté export OpenFOAM.

Si tu vois :

```text
Volume 25: 14000 elements, types=[6, 7]
```

alors l’exporteur doit absolument connaître les types `6` et `7`.

Si tu vois :

```text
Volume 25: 14000 elements, types=[10]
```

il faut forcer :

```python
gmsh.option.setNumber("Mesh.ElementOrder", 1)
```

puis remailler.

---

# 11. Vérifier les physical groups après génération

Ajoute aussi :

```python
def print_physical_groups():
    groups = gmsh.model.getPhysicalGroups()

    for dim, group_tag in groups:
        name = gmsh.model.getPhysicalName(dim, group_tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, group_tag)

        print(
            f"Physical group dim={dim}, tag={group_tag}, "
            f"name='{name}', entities={entities}"
        )
```

Pour le groupe `fluid`, tu dois voir quelque chose comme :

```text
Physical group dim=3, tag=1, name='fluid', entities=[25]
```

ou :

```text
Physical group dim=3, tag=1, name='fluid', entities=[25, 26, 27]
```

Si les entities sont vides ou correspondent à des volumes supprimés, le physical group est corrompu.

---

# 12. Séquence recommandée complète

Voici l’ordre que je recommande dans `VectorGmshBuilder` :

```python
def build_mesh(self, mesh_size=6.0):
    gmsh.initialize()

    # 1. Création géométrie : domaine + bâtiments
    self._create_geometry()

    # 2. Booléen robuste : fluide - bâtiments
    self._cut_buildings_from_fluid()

    # 3. Synchronisation
    gmsh.model.occ.synchronize()

    # 4. Suppression des volumes parasites / debris
    self._remove_small_volumes(min_rel_volume=1e-6)

    # 5. Heal uniquement si nécessaire
    gmsh.model.occ.healShapes(
        makeFaces=True,
        makeSolids=True,
        makeShells=True,
        makeEdges=True,
        tolerance=1e-6,
        fixDegenerated=True,
        fixSmallEdges=True,
        fixSmallFaces=True,
        sewFaces=True,
        makeSolidFromFaces=True,
    )
    gmsh.model.occ.synchronize()

    # 6. Deuxième nettoyage après heal
    self._remove_small_volumes(min_rel_volume=1e-6)

    # 7. Créer les physical groups APRÈS heal
    self.assign_patches()

    # 8. Options maillage
    self._set_mesh_options(mesh_size)

    # 9. Maillage avec fallback
    self._generate_3d_with_fallback()

    # 10. Vérification
    self._check_3d_mesh()

    return gmsh
```

Si ta version de Gmsh n’accepte pas tous les arguments de `healShapes()`, remplace par :

```python
gmsh.model.occ.healShapes()
gmsh.model.occ.synchronize()
```

puis éventuellement :

```python
gmsh.model.occ.removeAllDuplicates()
```

si la fonction existe dans ta version.

---

# 13. Version corrigée de `assign_patches()`

Voici une version plus sûre que celle qui prend tous les volumes :

```python
def assign_patches(self):
    gmsh.model.removePhysicalGroups()

    volumes = gmsh.model.getEntities(dim=3)

    if not volumes:
        raise RuntimeError("No 3D volume available for physical group 'fluid'.")

    masses = []

    for dim, tag in volumes:
        mass = abs(gmsh.model.occ.getMass(dim, tag))
        masses.append((tag, mass))

    total_mass = sum(m for _, m in masses)

    if total_mass <= 0.0:
        raise RuntimeError("All remaining 3D volumes have zero mass.")

    threshold = 1e-6 * total_mass

    fluid_tags = [
        tag for tag, mass in masses
        if mass >= threshold
    ]

    if not fluid_tags:
        fluid_tags = [max(masses, key=lambda x: x[1])[0]]

    print(f"Fluid volume tags selected for meshing: {fluid_tags}")

    fluid_group = gmsh.model.addPhysicalGroup(3, fluid_tags)
    gmsh.model.setPhysicalName(3, fluid_group, "fluid")
```

Si tu veux être encore plus strict pendant le debug, tu peux garder uniquement le plus grand volume :

```python
fluid_tags = [max(masses, key=lambda x: x[1])[0]]
```

Cela permet de vérifier si un volume parasite est responsable du blocage.

---

# 14. Tester avec un maillage progressif

Pour valider la pipeline, ne commence pas directement à `mesh_size=6.0`.

Fais :

```bash
python3 generate.py \
  --voxcity-h5 output/voxcity.h5 \
  --skip-run \
  --fill-gaps \
  --mesh-size 15.0
```

Puis :

```bash
python3 generate.py \
  --voxcity-h5 output/voxcity.h5 \
  --skip-run \
  --fill-gaps \
  --mesh-size 10.0
```

Puis :

```bash
python3 generate.py \
  --voxcity-h5 output/voxcity.h5 \
  --skip-run \
  --fill-gaps \
  --mesh-size 8.0
```

Puis :

```bash
python3 generate.py \
  --voxcity-h5 output/voxcity.h5 \
  --mesh-size 6.0
```

Si `15` ou `10` passent mais `6` échoue, c’est presque certainement un problème de géométrie trop fine ou de micro-features.

---

# 15. Fichiers de debug à produire

Dans le builder, après les booléens, écris des fichiers intermédiaires :

```python
gmsh.write("debug_after_boolean.brep")
```

Après `healShapes()` :

```python
gmsh.write("debug_after_heal.brep")
```

Avant maillage :

```python
gmsh.write("debug_before_mesh.msh")
```

Ensuite tu peux ouvrir dans Gmsh GUI :

```bash
gmsh debug_after_boolean.brep
gmsh debug_after_heal.brep
gmsh debug_before_mesh.msh
```

Dans Gmsh GUI, regarde :

- Tools → Statistics ;
- le nombre de volumes ;
- les volumes de très petite taille ;
- les faces dégénérées ;
- les surfaces non cousues ;
- la qualité du maillage 2D avant maillage 3D.

---

# 16. Diagnostic rapide à ajouter dans ton script

Tu peux coller ce petit bloc après la génération de géométrie :

```python
print("=== 3D volumes ===")

for dim, tag in gmsh.model.getEntities(dim=3):
    mass = abs(gmsh.model.occ.getMass(dim, tag))
    bbox = gmsh.model.occ.getBoundingBox(dim, tag)

    print(f"Volume tag={tag}, mass={mass:.6g}, bbox={bbox}")

print("=== Physical groups ===")

for dim, tag in gmsh.model.getPhysicalGroups():
    name = gmsh.model.getPhysicalName(dim, tag)
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

    print(f"Physical dim={dim}, tag={tag}, name='{name}', entities={entities}")
```

Tu veux voir quelque chose comme :

```text
Volume tag=25, mass=123456.7, bbox=(...)
Physical dim=3, tag=1, name='fluid', entities=[25]
```

Si tu vois plusieurs volumes avec des masses minuscules, il faut les supprimer.

---

# 17. Si le maillage échoue toujours à `mesh_size=6.0`

Dans ce cas, les causes restantes les plus probables sont :

## 17.1 Footprints trop complexes ou micro-arêtes

Réduis la précision des coordonnées :

```python
import numpy as np

def round_coords(geom, decimals=2):
    return shapely.ops.transform(
        lambda x, y, z=None: tuple(
            np.round(v, decimals) for v in (x, y) if v is not None
        ),
        geom
    )
```

Ou utilise WKT avec arrondi :

```python
geom = shapely.wkt.loads(
    shapely.wkt.dumps(geom, rounding_precision=2)
)
```

Pour de l’urbain, une précision de `0.05 m` à `0.1 m` est souvent suffisante.

---

## 17.2 Bâtiments trop proches les uns des autres

Si deux bâtiments sont séparés par une ruelle de `0.5 m` et que `mesh_size=6.0`, le maillage peut être impossible sans raffinement local.

Solutions :

- augmenter `mesh_size` ;
- simplifier les footprints ;
- fusionner les bâtiments qui se touchent ;
- ajouter un champ de taille de maillage local, mais c’est plus complexe.

---

## 17.3 Bâtiments qui se chevauchent

Si les footprints VoxCity se chevauchent, les booléens OCC deviennent fragiles.

Tu peux tester une fusion globale des footprints :

```python
from shapely.ops import unary_union

all_footprints = unary_union(list_of_footprints)
```

Puis extruder cette union comme un seul obstacle.

Tu perds le détail par bâtiment, mais cela permet de valider si le blocage vient des intersections entre bâtiments.

---

# 18. Plan d’action concret

Voici l’ordre dans lequel je ferais les corrections :

## Étape 1 — Corriger l’exporteur

Dans `direct_openfoam_exporter.py` :

```python
GMSH_PRI = 6
GMSH_PYR = 7

_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
    GMSH_PRI: 6,
    GMSH_PYR: 5,
}
```

Et ajouter un warning/erreur si type inconnu.

---

## Étape 2 — Forcer le maillage linéaire

Dans le builder :

```python
gmsh.option.setNumber("Mesh.ElementOrder", 1)
```

---

## Étape 3 — Recréer les physical groups après heal

Dans `vector_builder.py` :

```python
gmsh.model.occ.healShapes(...)
gmsh.model.occ.synchronize()

self._remove_small_volumes()

gmsh.model.removePhysicalGroups()
self.assign_patches()

gmsh.model.mesh.generate(3)
```

Pas de physical group créé avant `healShapes()`.

---

## Étape 4 — Sélection du volume fluide par masse

Dans `assign_patches()`, ne pas prendre aveuglément tous les volumes.

Utiliser :

```python
gmsh.model.occ.getMass(3, tag)
```

et supprimer les volumes dont le volume relatif est trop faible.

---

## Étape 5 — Tester gros maillage

```bash
python3 generate.py \
  --voxcity-h5 output/voxcity.h5 \
  --skip-run \
  --fill-gaps \
  --mesh-size 15.0
```

Puis descendre progressivement.

---

# 19. Correction minimale recommandée dans `direct_openfoam_exporter.py`

Version prudente :

```python
GMSH_TRI = 2
GMSH_QUAD = 3
GMSH_TET = 4
GMSH_HEX = 5
GMSH_PRI = 6
GMSH_PYR = 7

_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
    GMSH_PRI: 6,
    GMSH_PYR: 5,
}

_GMSH_TO_OPENFOAM_CELL = {
    GMSH_TET: "tet",
    GMSH_HEX: "hex",
    GMSH_PRI: "wedge",
    GMSH_PYR: "pyr",
}
```

Puis dans la collecte :

```python
unknown = {}

for etype in element_types:
    if etype not in _NODES_PER_ELEM:
        unknown[etype] = unknown.get(etype, 0) + 1

if unknown:
    raise RuntimeError(
        "Unsupported Gmsh element types encountered: "
        f"{unknown}. "
        "Add them to _NODES_PER_ELEM or set Mesh.ElementOrder=1."
    )
```

---

# 20. Conclusion

La prochaine action n’est pas seulement :

```python
_NODES_PER_ELEM[GMSH_PRI] = 6
_NODES_PER_ELEM[GMSH_PYR] = 5
```

Il faut faire les trois corrections ensemble :

1. **Exporter** : supporter les prismes/pyramides et ne plus ignorer silencieusement les éléments inconnus.

2. **Gmsh** : forcer un maillage linéaire :

   ```python
   gmsh.option.setNumber("Mesh.ElementOrder", 1)
   ```

3. **Physical groups** : les recréer après `healShapes()` et après nettoyage des volumes, jamais avant une modification géométrique.

Ensuite, seulement, relancer avec un maillage plus gros :

```bash
python3 generate.py \
  --voxcity-h5 output/voxcity.h5 \
  --skip-run \
  --fill-gaps \
  --mesh-size 15.0
```

Si `mesh_size=15.0` passe, descends vers `10.0`, puis `8.0`, puis `6.0`.

Si `mesh_size=6.0` échoue encore alors que `15.0` passe, le problème restant est très probablement une géométrie trop fragile : bâtiments qui se touchent, footprints avec micro-arêtes, faces coplanaires au sol, ou bâtiments trop proches des limites du domaine.

---

# 21. Résolution appliquée et tests

## 21.1 Corrections appliquées

### `direct_openfoam_exporter.py`
- Ajouté `GMSH_PRI = 6`, `GMSH_PYR = 7`
- Complété `_NODES_PER_ELEM` avec les prismes et pyramides
- Ajouté `_GMSH_TO_OPENFOAM_CELL` pour mapping complet
- Modifié `_collect_cells()` pour lever une erreur explicite sur types d’éléments inconnus

### `vector_builder.py`
- Après `fragment()`, mis à jour `self.fluid_tag` vers le plus grand volume **avant** d’appeler `_identify_building_volumes()`
- Nettoyage debris conservé + sélection du volume fluide par masse
- Extrusion des bâtiments avec `eps_ground = 0.1 m` pour éviter les faces coplanaires au sol
- Nettoyage Shapely des footprints VoxCity ( `make_valid`, `buffer(0)`, arrondi WKT, `simplify` )
- `assign_patches()` crée les physical groups 2-D + 3-D **après** les booléens et **avant** le maillage
- `export_openfoam()` recrée le physical group 3-D `fluid` si Gmsh l’a supprimé après `generate(3)`
- `build_mesh()` force `Mesh.ElementOrder = 1` et `Mesh.Algorithm3D = 1`
- Ajouté `Mesh.AngleToleranceFacetOverlap = 0.01` pour stabiliser le 2-D

### `generate.py`
- Ajouté `clean_footprint()` pour nettoyer les geometries VoxCity avant extrusion
- Import `gmsh` ajouté

## 21.2 Résultats des tests

| Test | Résultat |
|------|----------|
| `pytest test/test_direct_openfoam_export.py` | 3 passed |
| Simple rectangle (1 bâtiment, `mesh_size=15.0`) | 425 nœuds, 2233 éléments, export OK |
| VoxCity 30 bâtiments (`mesh_size=15.0`, `--fill-gaps`) | Export OK vers `neighborhood_case/constant/polyMesh` |
| `mesh_size=10.0` sans `--fill-gaps` | Échec 3D : `Invalid boundary mesh (overlapping facets)` |
| `mesh_size=15.0` + `--fill-gaps` | Succès |

## 21.3 Maillage exporté

Fichiers générés dans `neighborhood_case/constant/polyMesh/` :
- `points`
- `faces`
- `owner`
- `neighbour`
- `boundary`
- `cellZones`

## 21.4 Paramètres Gmsh stabilisateurs retenus

```python
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
```

## 21.5 Prochaines étapes recommandées

1. Descendre progressivement `mesh_size` : `15.0` -> `10.0` -> `8.0` -> `6.0`
2. Si échec à `mesh_size=8.0` ou `6.0`, augmenter `margin` dans `config.json`
3. Conserver `eps_ground=0.1` et le cleaning Shapely pour toutes les sources VoxCity
4. Pour CHT, vérifier que `export_multi_region()` utilise la même logique de physical groups