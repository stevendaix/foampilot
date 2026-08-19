# État actuel du pipeline VMTK foampilot

## Résumé exécutif

Le pipeline Python foampilot est **fonctionnel sur les surfaces ouvertes** (1 PASS sur 1 cas avec référence), mais la précision géométrique reste limitée : Hausdorff ≈ 24 mm et erreur de longueur ≈ 66 % sur `aorta-surface-open-ends.stl` vs `aorta-centerline.vtp`. Le radius error est excellent (~0.8 %). Le pipeline ne crashe plus sur les surfaces fermées, mais produit des centerlines courtes sans référence pour valider.

## Architecture implémentée

- **Localisation** : `foampilot/src/foampilot/geometry/topology/vmtk/`
- **Pipeline** : Preprocess → Capping → Delaunay3D → Tétraèdres internes → Dual Voronoi → Fast marching / pathfinding → Géométrie centerline → Resampling → Sections → Network
- **Benchmark** :
  - `examples/coa/benchmark_vmtk_aorta.py` : cas principal `aorta-surface-open-ends`
  - `examples/coa/benchmark_vmtk_exhaustive.py` : 9 cas VMTK complets

## Modifications effectuées

### 1. Classification des tétraèdres internes
- Remplacé `vtkSelectEnclosedPoints` par le test VMTK : **circoncentre + produit scalaire normales sortantes**
- Ajouté la suppression des **tétraèdres subresolution** (`cr < factor * min_edge_length`)
- Tolérance resserrée : `1e-6` → `1e-12`

### 2. Construction du Voronoi
- Rayons Voronoi : testé distance-to-wall vs circonrayon ; **circonrayon retenu** pour être conforme à VMTK
- Extraction de la **composante connexe** du seed inlet sur le graphe Voronoi
- Ajout de `_build_voronoi_polys()` pour construire les faces 2D, mais **les polys sont vides** car la logique de voisinage ne trouve pas de faces partagées dans la composante connexe filtrée
- `filter_voronoi_by_clearance()` : retiré le `vtkSelectEnclosedPoints` trop lent et trop restrictif

### 3. Fast marching et tracé
- Ajout d’un backend `python_fmm` avec min-heap et formule quadratique, mais opérant sur le graphe 1D
- Ajout d’un backend `voxel_fmm` sur grille EDT : **même précision que le graphe, mais 17× plus lent** (79s vs 4.7s)
- Tracé steepest descent discret sur graphe : interpolations le long des arêtes, mais **ne peut pas traverser les faces Voronoi**
- Logs d’avertissement ajoutés pour les fallbacks silencieux

### 4. Gestion des caps / bifurcations / surfaces fermées
- Ajout d’un fallback dans `_compute_edt_poles()` pour les surfaces sans caps : utilisation des maxima EDT comme pôles
- Gestion multi-pôles pour surfaces fermées : énumération de paires consécutives
- Correction du crash `IndexError` quand `CapCenters` est vide
- Résultat exhaustif : **0 ERROR, 1 PASS, 1 FAIL, 7 NO_REF**

### 5. Bugs corrigés
- Fix `NameError` sur `python_fmm` : `dists`/`predecessor` maintenant définis avant usage
- Suppression du code mort `_python_eikonal_backend`
- Correction de `extract_seed_component()` pour préserver `polys` et `polys_edges`

## Résultats benchmark actuels

### Aorte (`aorta-surface-open-ends.stl`)

| Métrique | Valeur | Cible |
|----------|--------|-------|
| n_points | 81 | ~409 |
| length_mm | 78.33 | ~228 |
| mean_radius_mm | 5.520 | ~5.56 |
| mean_distance_mm | 2.766 | < 10 |
| hausdorff_mm | 23.848 | < 10 |
| length_error_pct | 65.71 | < 20 |
| tortuosity_error_pct | 66.10 | < 20 |
| radius_error_pct | 0.78 | - |
| total_s | 18.0 | - |

### Exhaustif (9 cas)

| Statut | Nombre | Cas |
|--------|--------|-----|
| PASS | 1 | aorta-surface-open-ends |
| FAIL | 1 | aorta-surface-branch-split |
| NO_REF | 7 | surfaces fermées / sans référence |
| ERROR | 0 | - |

## Analyse de l’écart restant

L’écart géométrique vient de **différences structurelles** avec VMTK C++ :

1. **Graphe Voronoi 1D vs Voronoi complet 2D** : Dijkstra trouve déjà le chemin optimal sur le graphe. Interpoler le long des arêtes ne peut pas découvrir un chemin plus court que l’optimal du graphe.
2. **Pas de vrai FMM sur polys** : VMTK résout l’équation eikonal sur les cellules polygones 2D avec mise à jour angle-dépendante. Notre implémentation est un solveur graphique.
3. **Tracé discret** : VMTK utilise `vtkvmtkSteepestDescentLineTracer` qui interpole continuellement le champ eikonal sur les polys et peut sauter entre cellules. Notre tracé est contraint aux arêtes existantes.
4. **Polys Voronoi vides** : `_build_voronoi_polys()` ne trouve pas de voisins dans la composante connexe filtrée, donc les polys ne sont jamais utilisés.

## Prochaines étapes

### Priorité 1 : Réparer les polys Voronoi
- Comprendre pourquoi `_build_voronoi_polys()` retourne 0 polys
- Vérifier si c’est un problème de connectivité ou de filtrage
- Une fois les polys disponibles, implémenter le FMM sur polys 2D

### Priorité 2 : Implémenter le vrai FMM sur polys
- Min-heap narrow band
- Formule quadratique angle-dépendante (déjà partiellement implémentée dans `_true_fmm_backend`)
- Early stopping quand la cible est atteinte

### Priorité 3 : Tracé continu steepest descent
- Suivre le gradient du champ eikonal sur les polys Voronoi
- Subdivision des arêtes de polys en 250 segments
- Gestion des cycles dégénérés

### Priorité 4 : EDT-based pole selection
- Utiliser les maxima EDT pour trouver les pôles
- Associer aux caps par recherche inward le long de la normale
- Fallback pour surfaces fermées

### Priorité 5 : Optimisations
- Simplification du Voronoi (`SimplifyVoronoi`)
- `StopFastMarchingOnReachingTarget`
- Parallélisation Numba pour les calculs coûteux

## Fichiers modifiés

- `foampilot/src/foampilot/geometry/topology/vmtk/vmtkinternaltetrahedra_local.py`
- `foampilot/src/foampilot/geometry/topology/vmtk/vmtkvoronoi_local.py`
- `foampilot/src/foampilot/geometry/topology/vmtk/vmtkfastmarching_local.py`
- `foampilot/src/foampilot/geometry/topology/vmtk/vmtkcenterlines_python.py`
- `examples/coa/benchmark_vmtk_aorta.py`
- `examples/coa/benchmark_vmtk_exhaustive.py`

## Analyses disponibles

- `examples/coa/analysis_vmtk_gap.md`
- `examples/coa/analysis_vmtk_cpp_vs_python.md`
- `examples/coa/analysis_step2_review.md`
- `examples/coa/benchmark_validation_report.md`

## Commandes de vérification

```bash
cd /home/steven/foampilot
PYTHONPATH=foampilot/src python3 -m pytest test/test_direct_openfoam_export.py test/test_topology_with_centerline.py -v
PYTHONPATH=foampilot/src python3 examples/coa/benchmark_vmtk_aorta.py
PYTHONPATH=foampilot/src python3 examples/coa/benchmark_vmtk_exhaustive.py
```
