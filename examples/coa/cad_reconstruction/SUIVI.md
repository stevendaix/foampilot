# Suivi CAD Reconstruction TBAD

## Objectif
Reproduire en Python/VTK local (`vmtk_local`) le pipeline VMTK complet pour TBAD :
1. Extraction centerlines (Voronoi/Dijkstra)
2. Extraction sections
3. Fitting B-spline
4. Construction CAD (loft OCC)
5. Maillage surface adaptatif
6. Maillage volume avec boundary layer
7. Export OpenFOAM direct

## Stratégie
- **Centerlines** : Voronoi/Dijkstra inspiré de `vtkvmtkPolyDataCenterlines.cxx`
- **Sections** : coupes perpendiculaires le long de la centerline via `trimesh.section`
- **B-spline** : fitting via `geomdl` + option least-squares `scipy.interpolate.splprep`
- **CAD** : lofting via `gmsh.model.occ.addThruSections(makeSolid=True)`
- **Maillage** : `vmtkMeshGenerator` local avec sizing adaptatif + boundary layer
- **Export** : `DirectOpenFOAMExporter` existant dans foampilot

## Fichiers créés

| Fichier | Rôle | Statut |
|---------|------|--------|
| `vmtk_local/__init__.py` | Module package | ✅ |
| `vmtk_local/pypes.py` | Framework de base VMTK | ✅ |
| `vmtk_local/vmtkcenterlines.py` | Extraction centerlines | ✅ |
| `vmtk_local/vmtkcenterlinesections.py` | Sections centerline | ✅ |
| `vmtk_local/vmtkbranchsections.py` | Sections branches | ✅ |
| `vmtk_local/vmtksurfacereader.py` | Lecteurs/écrivains surface | ✅ |
| `vmtk_local/vmtkdistancetocenterlines.py` | Distance aux centerlines | ✅ |
| `vmtk_local/vmtkmeshgenerator.py` | Maillage volume Gmsh | ✅ |
| `vmtk_local/vmtkmeshwriter.py` | Export mesh | ✅ |
| `vmtk_local/vmtksurfaceremesher.py` | Remaillage surface | ✅ |
| `vmtk_local/vmtkmeshquality.py` | Qualité maillage | ✅ |
| `centerline_extractor.py` | Wrapper TBAD | ✅ |
| `section_extractor.py` | Extraction sections 2D/3D | ✅ |
| `bspline_fitter.py` | Fitting B-spline geomdl + scipy | ✅ |
| `occ_builder.py` | Construction B-rep + maillage + export | ✅ |
| `cad_reconstruction.py` | Pipeline complet VMTK-like | ✅ |
| `run_patient.py` | CLI patient | ✅ |
| `test_validation.py` | Tests validation | ✅ |

## Tests de validation

### Inspirés de VMTK
- `test_vmtkcenterlines.py` : centerlines sur surface d'aorte
- `test_vmtkcenterlinesections.py` : sections le long des centerlines
- `test_vmtkbranchsections.py` : sections de branches avec assertions numériques
- `test_vmtkdistancetocenterlines.py` : array de distance
- `test_vmtkmeshgenerator.py` : maillage volume tétraédrique
- `test_vmtksurfaceremesher.py` : remaillage surface
- `test_vmtkmeshquality.py` : qualité maillage

### Implémentés
- `test_centerline_points_count` : >= 10 points, 3D
- `test_centerline_continuous` : pas de sauts
- `test_sections_count` : >= 1 section
- `test_section_points_count` : >= 3 points par section
- `test_section_center_consistency` : cohérence centre/points
- `test_section_direction_unit` : direction normalisée
- `test_section_local_frame_orthogonal` : frame orthogonale
- `test_section_2d_projection` : projection 2D correcte
- `test_adaptive_sizing` : tableau distance centerlines
- `test_mesh_generation` : maillage volume tétraédrique
- `test_surface_remeshing` : remaillage surface
- `test_mesh_quality` : qualité maillage

## Résultats d'exécution

### Pipeline patient58 (2026-08-12)
- **Centerlines** : 65 points extraits
- **Sections** : 64 sections extraites
- **Loft OCC** : 1 volume généré (32 courbes, indices pairs)
- **Maillage** : Gmsh tétraèdres + optimisation Netgen
- **Export OpenFOAM** : `constant/polyMesh/` écrit
- **Tests pytest** : 8/8 passés

## Corrections apportées

1. Import `_trimesh_to_vtk_polydata` dans `centerline_extractor.py`
2. Fonctions helper dans `vmtkcenterlinesections.py`
3. Compatibilité `geomdl 5.4.0` : `generate_knot_vector` → `generate`
4. `curve.evaluatepts()` → `curve.evalpts`
5. `addThruSections` : `ruled=False` → `makeRuled=False`
6. `addBSpline` nécessite des `pointTags`, pas des coordonnées
7. `addThruSections` nécessite des `wireTags`, pas des `curveTags`
8. Filtrage des sections pires pour éviter l'échec OCC
9. Centre de section calculé comme `points.mean(axis=0)`
10. Orientation cohérente des sections via `signed_area()`
11. `makeSolid=True` pour export OpenFOAM volume
12. Ajout `_mesh_and_export()` dans `OCCBuilder`
13. `DirectOpenFOAMExporter` intégré au pipeline
14. `synchronize()` avant physical group
15. `gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", ...)`
16. Ajout `vmtkDistanceToCenterlines`
17. Ajout `vmtkMeshGenerator`
18. Ajout `vmtkMeshWriter`
19. Ajout `vmtkSurfaceRemesher`
20. Ajout `vmtkMeshQuality`
21. Ajout `README.md`

## Prochaines étapes
1. Tester sur plusieurs patients
2. Ajouter boundary layer Gmsh
3. Télécharger données référence VMTK
4. Validation Hausdorff / qualité maillage
5. Support CHT multi-région
