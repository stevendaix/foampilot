# Amélioration du STL par lissage VTK

Le STL dense partitionné avait déjà un volume proche de VMTK, mais une aire supérieure de 13,52 %. Un lissage `vtkWindowedSincPolyDataFilter` contrôlé a été testé après l’union globale, sans modifier le nombre de points ou de cellules et sans activer le lissage des arêtes frontières.

## Variante retenue

La variante retenue utilise 15 itérations et un `passband=0,05`.

| Métrique | VMTK officiel | STL dense lissé | Écart |
|---|---:|---:|---:|
| Volume | 13184,2667 | 13142,7135 | **−0,32 %** |
| Aire | 4517,7631 | 4770,5324 | **+5,60 %** |
| Composantes | 1 | 1 | Conforme |
| Arêtes frontière | 0 | 0 | Conforme |
| Arêtes non-manifold | 0 | 0 | Conforme |
| Normales cohérentes | Oui | Oui | Conforme |

Le lissage réduit l’erreur d’aire de 13,52 % à 5,60 %, tout en conservant une erreur de volume inférieure à 0,4 %. Les variantes avec des passbands de 0,1 et 0,2 ont été écartées car elles augmentent l’erreur d’aire et déplacent davantage la surface.

## Fichier recommandé

`examples/medical_build/outputs/aorta_six_branch_union_dense_smoothed_0p75mm.stl`

Ce fichier est propre pour la CFD selon les contrôles effectués : une composante, watertight, zéro arête frontière, zéro arête non-manifold et normales cohérentes.

La géométrie n’est pas encore identique point par point à VMTK. Les bornes et les distances locales restent différentes, mais ce candidat est actuellement le meilleur compromis mesuré entre volume, aire, topologie et régularité de surface.
