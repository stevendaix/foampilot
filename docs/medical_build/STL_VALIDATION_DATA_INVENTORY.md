# Données disponibles pour valider le STL

## Données présentes

| Donnée | Emplacement | Validation possible |
|---|---|---|
| STL global voxelisé | `examples/medical_build/outputs/aorta_six_branch_union_voxel_0p5mm.stl` | Fermeture, composantes, volume, normales, arêtes frontière et non-manifold |
| Rapport du STL global | `examples/medical_build/outputs/aorta_six_branch_union_voxel_0p5mm.validation.json` | Résultats topologiques reproductibles |
| Six STL de branches | `/tmp/vmtk_six_branch_stl/branch_00.stl` à `branch_05.stl` | Validité de chaque branche et comparaison des raccordements |
| Sections réelles | `/tmp/vmtk_six_branch_sections/vmtk_real_sections.json` | Position, tangent, contour, aire, longueur et volume local |
| Centerlines complexes | `examples/medical_build/case_complex/analysis/centerlines.vtp` | Distribution spatiale des centerlines et comparaison avec la reconstruction complexe |
| Diagnostics de centerline | `examples/medical_build/case_complex/analysis/*.json` | Nombre de branches, connectivité, métriques et temps |
| Graphe NetworkX | Généré par `vascular_graph.py` | Connectivité, bifurcations, branches isolées et cycles |
| Sorties OpenFOAM | `examples/medical_build/case_complex/openfoam/constant/triSurface/` | Patches et préparation CFD, lorsque le cas est disponible |

## Données de référence manquantes ou incomplètes

La surface VMTK fermée originale de l’aorte six branches n’est pas encore disponible dans un fichier directement comparable au STL global actuel. Les fichiers du dépôt `vmtk-test-data` identifiés dans l’environnement sont principalement des sorties de tests génériques VMTK et ne constituent pas automatiquement la surface de référence de cette aorte.

Il manque donc encore une référence anatomique surfacique fermée permettant de calculer directement l’écart de volume et les distances de Hausdorff ou Chamfer. Tant que cette surface n’est pas retrouvée, le volume STL doit être comparé aux sections et centerlines, mais cette comparaison reste une validation indirecte.

## Validations déjà possibles

Le STL global actuel est validé comme maillage CFD de base : il est fermé, constitué d’une seule composante, possède zéro arête frontière, zéro arête non-manifold et des normales cohérentes. Les six branches individuelles disposent également de rapports de composants.

Les sections permettent de vérifier les positions et les aires localement. Cependant, leur intégration directe donne un volume de 20246,64 unités³, alors que le STL global donne 11238,10 unités³. La différence de 80,16 % s’explique principalement par le double comptage des portions communes des branches et ne doit pas être interprétée seule comme une erreur du STL.

## Validation à ajouter lorsque la surface VMTK est retrouvée

La surface de référence devra être fermée et orientée. La pipeline devra alors calculer son volume, son aire, sa boîte englobante, ses composantes et ses arêtes non-manifold, puis comparer ces métriques au STL reconstruit. Une distance symétrique de surface, une distance maximale et une comparaison par sections perpendiculaires aux centerlines devront compléter la comparaison volumique.
