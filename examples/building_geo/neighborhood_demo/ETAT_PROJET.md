# État du projet : Maillage OpenFOAM urbain VoxCity

## Date : 15 août 2026

## Objectif
Générer un maillage 3D OpenFOAM valide pour un quartier urbain avec des patches physiques nommés (buildings, ground, inlet, side_left, side_right, outlet, top) à partir d'empreintes de bâtiments fusionnées.

## État actuel : ⚠️ PARTIEL

### ✅ Réalisé
1. **Géométrie fluide valide**
   - 1 volume fluide unique, masse préservée (64.5M m³)
   - 117 surfaces (6 domaine + 111 interfaces bâtiments)
   - Bounding box correct : x=[450142.6, 450591.1], y=[5410894.8, 5411427.1], z=[-5.0, 127.0]

2. **Maillage Gmsh réussi**
   - 77 114 nœuds
   - 131 999 tétraèdres
   - Maillage valide dans Gmsh

3. **Export OpenFOAM fonctionnel**
   - Fichiers polyMesh générés
   - 6 patches nommés exportés : ground, inlet, outlet, side_left, side_right, top
   - Patch "buildings" manquant dans le fichier boundary

4. **Configuration cas OpenFOAM**
   - controlDict, fvSchemes, fvSolution créés
   - Conditions aux limites définies pour U et p

### ❌ Problème bloquant

**checkMesh échoue avec 105 916 cellules à volume négatif**

```
***Zero or negative cell volume detected. Minimum negative volume: -5.67416e+06
***Error in face pyramids: 103673 faces are incorrectly oriented.
```

**Diagnostic :**
- Le maillage Gmsh est valide
- L'export OpenFOAM corrompt l'orientation des faces internes
- Pour certaines faces internes, le centroïde du propriétaire = centroïde du voisin (to_neighbour = [0,0,0])
- Le produit scalaire face_normal · to_neighbour = 0, donc l'orientation ne peut pas être déterminée
- Cela conduit à des faces mal orientées et des cellules de volume négatif dans OpenFOAM

**Fichiers modifiés :**
- `/home/steven/foampilot/examples/building_geo/voxcity_export_work/src/vector_builder.py`
- `/home/steven/foampilot/foampilot/src/foampilot/mesh/direct_openfoam_exporter.py`

### 🔄 En cours de résolution

**Tentatives effectuées :**
1. ✅ `removeObject=False` dans cut() pour préserver la surface top
2. ✅ Suppression des fragments de bâtiments après cut
3. ✅ Algorithme MeshAdapt (1) au lieu de Delaunay (4)
4. ✅ `mesh.generate(2)` puis `generate(3)`
5. ✅ `gmsh.model.mesh.setSize()` sur toutes les surfaces
6. ✅ Orientation des faces internes recalculée par centroïdes
7. ❌ Compaction des points (n'améliore pas le problème d'orientation)
8. ❌ Classification directe par centroïdes (ne résout pas l'orientation)
9. ✅ **Bug identifié et corrigé** : `to_neighbour = [0,0,0]` venait du fait que `owner_face` et `neighbour_face` partagent les mêmes nœuds → centroïdes identiques → orientation systématiquement fausse
10. ✅ Correction : utiliser directement `owner_face` (déjà orientée par `_orient_face_outward`)
11. ✅ `patch_map` maintenant utilisé pour la classification des boundary faces
12. ✅ Suppression du `offset` fragile dans l'itération des éléments

**Problème résolu :**
- Les centroïdes des faces internes étaient calculés sur les sommets de la face, pas sur les cellules
- `owner_face` et `neighbour_face` ont les mêmes sommets → centroïdes identiques
- Le test `np.dot(face_normal, to_neighbour) > 0` retournait toujours False
- L'orientation choisie était systématiquement celle du neighbour au lieu de l'owner
- Cela provoquait 105 916 cellules négatives et 103 673 faces mal orientées

### 📋 Prochaines étapes

**À tester :**
1. Relancer le pipeline complet VoxCity pour vérifier que `checkMesh` passe maintenant
2. Vérifier que le patch "buildings" apparaît correctement dans le fichier boundary
3. Si OK, lancer une simulation de validation

**Si problème persiste :**
- Investiguer la topologie du maillage VoxCity (faces dupliquées, tétraèdres dupliqués)
- Comparer avec `gmshToFoam` officiel

### 📁 Fichiers clés

- `vector_builder.py` : Construction géométrie, cut séquentiel, export
- `direct_openfoam_exporter.py` : Export polyMesh, orientation faces
- `neighborhood_demo/generate.py` : Pipeline complet
- `neighborhood_demo/config.json` : Configuration domaine
- `test_case/` : Cas OpenFOAM exporté (invalide)

### 📊 Statistiques

- Bâtiments : 22 (18 fusionnés)
- Surfaces fluide : 117
- Nœuds maillage : 77 114
- Tétraèdres : 131 999
- Patches exportés : 6/7 (manque "buildings")
- Cellules négatives : 105 916

### 🎯 Décisions à prendre

1. Continuer avec l'export direct OpenFOAM et corriger l'orientation des faces ?
2. Basculer vers `gmshToFoam` officiel avec perte de noms personnalisés ?
3. Investiguer pourquoi les centroïdes sont identiques (problème de tolérance Gmsh) ?
