# Fonctionnement du STL VMTK et contrôle de propreté

## Ce que fait réellement VMTK

Dans VMTK, l’écriture STL est réalisée par la classe `vmtkSurfaceWriter`. Lorsque le format est `stl`, elle instancie directement `vtk.vtkSTLWriter`, lui transmet le `vtkPolyData` et appelle `Write()`.

La conséquence importante est que le writer **n’effectue pas à lui seul** une reconstruction anatomique, une union booléenne, un remplissage de trous ou une correction des bifurcations. Il écrit la surface qui lui est fournie. La propreté du STL dépend donc des étapes précédentes de la pipeline.

La chaîne conceptuelle VMTK est généralement :

```text
Surface ou image
  → extraction de surface / Marching Cubes
  → triangulation
  → nettoyage et connectivité
  → caps des ouvertures si nécessaire
  → orientation des normales
  → validation
  → vmtkSurfaceWriter / vtkSTLWriter
```

Les caps sont réalisés en amont par `vmtkSurfaceCapper`, les normales par `vmtkSurfaceNormals`, la connectivité par `vmtkSurfaceConnectivity` et les propriétés géométriques par `vmtkSurfaceMassProperties`. Le writer ne fait que sérialiser le `vtkPolyData` final.

## Test de conservation du writer VMTK

La surface officielle VMTK six branches a été lue puis réécrite avec les classes natives Python VMTK :

`/tmp/vmtk-test-data/input/aorta-surface.stl`

vers

`/tmp/vmtk_native_export/aorta_surface_native_writer.stl`

Les métriques sont strictement identiques avant et après écriture :

| Métrique | Original | Export VMTK | Écart |
|---|---:|---:|---:|
| Points | 6468 | 6468 | 0 |
| Cellules | 12932 | 12932 | 0 |
| Aire | 4517,7631 | 4517,7631 | 0 % |
| Volume | 13184,2667 | 13184,2667 | 0 % |
| Composantes | 1 | 1 | 0 |
| Arêtes frontière | 0 | 0 | 0 |
| Arêtes non-manifold | 0 | 0 | 0 |

Le STL officiel VMTK est donc propre pour les critères topologiques usuels : fermé, connecté, sans trous ouverts, sans arêtes non-manifold et avec un volume cohérent.

## Comparaison avec foampilot

Le STL global foampilot est également propre topologiquement : une composante, zéro arête frontière, zéro arête non-manifold et normales cohérentes. En revanche, il n’est pas encore aussi fidèle géométriquement à la surface VMTK : son volume est inférieur de 14,76 % avec la variante 0,5 mm et fermeture minimale, et son aire est supérieure de 24,92 %.

Cette différence ne provient pas du format STL ni du writer. Elle provient de la reconstruction des branches, des sections et du raccordement aux bifurcations. VMTK écrit une surface déjà construite ; il ne transforme pas automatiquement six volumes de branches qui se recouvrent en une surface anatomique globale correcte.

## Conclusion

Le STL VMTK officiel est propre et le writer conserve exactement les métriques du `vtkPolyData`. La bonne stratégie pour foampilot est donc de reproduire les étapes qui précèdent le writer : triangulation, nettoyage, caps, normales, connectivité et union géométrique. Il ne faut pas chercher une amélioration dans l’écriture STL elle-même.
