# Reconstruction STL globale améliorée

## Objectif

Les six branches étaient auparavant exportées comme surfaces fermées séparées, mais leur concaténation produisait des recouvrements au niveau des bifurcations, plusieurs composantes et des arêtes non-manifold. La reconstruction améliorée effectue une union volumique sur une grille régulière, puis extrait l’isosurface par Marching Cubes.

## Méthode

Chaque STL de branche est voxelisé et rempli. Les grilles sont placées dans un repère global commun, puis réunies par union booléenne discrète. Une fermeture morphologique minimale d’une itération est appliquée uniquement pour supprimer les discontinuités d’un voxel au niveau des raccordements. L’isosurface est ensuite extraite avec un pas de 0,5 mm, fusionnée et nettoyée.

Cette méthode ne prétend pas reproduire exactement la surface VMTK originale : elle produit un volume CFD robuste à partir des branches reconstruites. Le paramètre de fermeture morphologique doit donc rester explicite et contrôlé.

## Campagne de résolution

| Pas voxel | Fermeture | Composantes | Watertight | Arêtes frontière | Arêtes non-manifold | Volume |
|---:|---:|---:|---:|---:|---:|---:|
| 1,00 mm | 0 | 1 | Oui | 0 | 0 | 12918,25 |
| 0,75 mm | 0 | 2 | Oui | 0 | 0 | 12047,40 |
| 0,50 mm | 0 | 4 | Oui | 0 | 0 | 11117,44 |
| 0,50 mm | 1 | 1 | Oui | 0 | 0 | 11238,10 |
| 0,50 mm | 2 | 1 | Oui | 0 | 0 | 11407,29 |
| 0,50 mm | 3 | 1 | Oui | 0 | 0 | 11579,89 |

## Candidat recommandé

Le fichier recommandé est `examples/medical_build/outputs/aorta_six_branch_union_voxel_0p5mm.stl`. Il utilise un pas de 0,5 mm et une fermeture morphologique d’une itération. Les vérifications indépendantes donnent une composante, zéro arête frontière, zéro arête non-manifold, des normales cohérentes et un maillage watertight.

Le choix d’une seule itération est volontaire : les fermetures 2 et 3 produisent également des volumes valides, mais modifient davantage les bifurcations et augmentent le volume. Elles doivent être utilisées seulement si la géométrie présente encore des discontinuités visibles ou si la génération du maillage CFD échoue.

## Limites

La qualité finale dépend de la qualité des STL de branches en entrée. La fermeture morphologique peut combler une discontinuité numérique, mais elle ne corrige pas une centerline incorrecte, une section mal orientée ou une branche absente. Une comparaison de distance à la surface VMTK originale doit encore être ajoutée lorsque la surface six branches de référence sera disponible dans un format directement comparable.

Le validateur indépendant est `examples/medical_build/validate_stl.py`. Le générateur est `examples/medical_build/voxel_union_stl.py`.
