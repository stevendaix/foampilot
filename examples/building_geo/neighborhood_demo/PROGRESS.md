# Suivi du projet : Maillage OpenFOAM urbain VoxCity

## Objectif
Générer un maillage 3D OpenFOAM valide pour un quartier urbain avec des patches physiques nommés (buildings, ground, inlet, side_left, side_right, outlet, top, fluid) à partir d'empreintes de bâtiments fusionnées dans `/home/steven/foampilot/examples/building_geo/neighborhood_demo/`.

## Contraintes et préférences
- Utiliser `gmsh.model.occ.cut()` séquentiel avec `removeObject=True, removeTool=True` pour découper les bâtiments du volume fluide
- Les bâtiments doivent dépasser sous le domaine fluide (`base_z - eps_ground - 0.1`) pour éviter les échecs booléens coplanaires
- Les marges du domaine suivent les règles `building_aero` quand `margin=None` (amont=`4*Hmax`, aval=`7.5*Hmax`, latéral=`2*D`, haut=`1.25*Hmax`)
- Les noms de groupes physiques doivent survivre à la génération du maillage pour l'export OpenFOAM

## Progression

### ✅ Réalisé
- Corriger le bug de comparaison de hauteur dans `merge_nearby_buildings()` (`group[0]` → `group[0][1]`)
- Ajouter le filtrage `post_merge_cleanup()` des fragments avec `min_area=2.0 m²`
- Ajouter la génération d'images matplotlib étape par étape (`footprint_processing_steps.png`) : 129 brutes → 88 nettoyées → 8 bâtiments fusionnés
- Corriger le flag `_patches_assigned` empêchant la recréation des groupes physiques après `removePhysicalGroups()`
- Corriger la base des bâtiments coplanaire (`zmin = -5.0000001`) en l'abaissant à `base_z - eps_ground - 0.1`
- Implémenter la boucle séquentielle `cut()` qui découpe avec succès les 16 bâtiments du volume fluide
- Vérifier que le volume fluide conserve le tag 1 après les découpes, masse préservée (~31.5M m³)
- Revenir sur le `_cleanup_geometry()` post-extrusion qui invalidait les tags de volume
- Corriger la casse de l'import `DirectOpenFOAMExporter`
- Retirer les logs de debug de `export_openfoam()` et `assign_patches()`
- Restaurer la logique complète de `build_mesh()` avec suppression des volumes bâtiments et nettoyage des surfaces dupliquées
- Corriger le bug `_saved_phys_groups` manquant dans `build_mesh()`
- Utiliser `removeObject=False` dans `cut()` pour préserver la surface du dessus du domaine
- Éviter `removeAllDuplicates()` après chaque `cut` car il supprime la surface top
- Classifier les volumes 3D restants par masse comme "fluid" (seuil `1e-6 * total_mass`) car les volumes bâtiments sont supprimés lors des découpes
- Corriger `assign_patches()` pour utiliser `_domain_bbox` quand `margin` est fourni
- Corriger `build_mesh()` pour appeler `assign_patches()` **avant** `mesh.generate(3)`
- Forcer `mesh.generate(2)` puis `generate(3)` pour que les surfaces soient maillées avant le volume
- Corriger `_get_surface_patch_map()` dans `DirectOpenFOAMExporter` pour construire `patch_map` directement depuis les faces des cellules 3D (car les éléments 2D ne correspondent pas aux clés de faces 3D)
- Corriger la tolérance de classification des patches de `1e-4` à `1.0` dans `_get_surface_patch_map()`
- Corriger le bug `all_coords` dans `_get_surface_patch_map()` qui utilisait `for i in tag_to_index` au lieu de `for i in range(len(tag_to_index))`
- Remplacer la classification par `patch_map` dans `_build_mesh_data()` par une classification directe par centroïdes de faces avec tolérance `1.0`
- Supprimer le debug `_build_face_orientations` et restaurer le code complet de `_build_mesh_data()`

### 🔄 En cours
- Tester l'export OpenFOAM complet avec tous les patches nommés

### ⛔ Bloqué
- `DirectOpenFOAMExporter` n'exporte que 2 patches (`ground` et `buildings`) au lieu de 6 patches nommés
- Les patches `inlet`, `outlet`, `side_left`, `side_right`, `top` sont manquants dans le fichier `boundary`
- Dernière erreur : `AttributeError: 'DirectOpenFOAMExporter' object has no attribute '_write_all'`

## Décisions clés
- Utiliser `cut` séquentiel plutôt que `fragment` pour éviter les facettes superposées aux interfaces de bâtiments
- Après chaque `cut`, ne PAS appeler `removeAllDuplicates()` car ça supprime la surface top du domaine
- Utiliser `removeObject=False` dans `cut()` pour préserver la géométrie du fluide
- Supprimer les volumes bâtiments avec `gmsh.model.occ.remove()` avant le maillage
- Assigner les patches **avant** `mesh.generate(3)` pour que les tags de surfaces soient valides
- Classifier les faces de bord directement par centroïde dans `_build_mesh_data()` au lieu de dépendre de `patch_map`

## Prochaines étapes
- Corriger l'erreur `_write_all` manquante dans `DirectOpenFOAMExporter`
- Vérifier que `_build_mesh_data()` retourne bien toutes les données
- Tester l'export OpenFOAM complet et valider le fichier `boundary`
- Si les patches sont corrects, augmenter la résolution du maillage (`mesh_size` plus petit)
- Lancer `checkMesh` pour valider la qualité du maillage

## Contexte critique
- `Invalid boundary mesh (overlapping facets)` initialement causé par `fragment` créant des faces dupliquées ; `cut` séquentiel a résolu ce problème
- Les empreintes de bâtiments avaient `zmin = -5.0000001` exactement coplanaire avec le domaine fluide `zmin = -5.0000001`, causant des échecs booléens et une sélection de volume incorrecte après `cut`
- `fragment` + `healShapes` a corrompu le volume fluide (masse passée de 31.5M m³ à ~10K m³) — confirmé non viable
- `cut` séquentiel avec `removeObject=False, removeTool=True` préserve correctement le volume fluide
- Le `patch_map` de l'exporteur ne correspond pas aux clés de faces 3D car les éléments 2D sont maillés différemment
- Dernière erreur bloquante : `_write_all` manquante dans `DirectOpenFOAMExporter`

## Fichiers pertinents
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/generate.py` : génération des empreintes, logique de fusion, image matplotlib étape par étape
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/config.json` : configuration du domaine
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/output/voxcity.h5` : cache HDF5 pour les données voxcity
- `/home/steven/foampilot/examples/building_geo/voxcity_export_work/src/vector_builder.py` : construction géométrie Gmsh, logique cut séquentiel, assignation patches, export OpenFOAM
- `/home/steven/foampilot/foampilot/src/foampilot/mesh/direct_openfoam_exporter.py` : exporteur OpenFOAM direct, classification patches
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/output/footprint_processing_steps.png` : visualisation des étapes de traitement des empreintes

## Statistiques du maillage (dernière exécution)
- Nœuds : 9 902
- Tétraèdres 3D : 1 245
- Type d'éléments 3D : [4] (tetraèdres linéaires)
- Volumes fluides : 19
- Volumes bâtiments : 0 (supprimés pour maillage)

## Configuration du domaine
- x=[450142.6003483982, 450591.09971917124]
- y=[5410894.800473494, 5411427.099638114]
- z=[-5.0, 127.0]
