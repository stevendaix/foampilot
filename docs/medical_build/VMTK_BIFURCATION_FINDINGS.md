# Vérification du workflow VMTK pour les sections et bifurcations

## Sources consultées

- `vtkvmtkPolyDataCenterlineSections.cxx`
- `vtkvmtkPolyDataCenterlineSections.h`
- `vtkvmtkPolyDataCenterlines.cxx`

## Constats vérifiés

`vmtkcenterlinesections.py` instancie `vtkvmtkPolyDataCenterlineSections`, lui fournit séparément la surface et les centerlines, puis produit une section pour chaque point de chaque cellule de centerline. Le filtre C++ coupe la surface avec un plan défini par le point de centerline et une tangente locale. La tangente est calculée comme la moyenne normalisée des directions vers le point précédent et le point suivant, et non comme une interpolation arbitraire entre deux sections.

Le filtre C++ appelle `vtkvmtkPolyDataBranchSections::ExtractCylinderSection(input, point, tangent, section, closed)`. Il ne reconstruit pas le contour par un cercle de rayon moyen. Il conserve les points de l’intersection plan-surface, calcule l’aire, les tailles minimale et maximale, l’indice de forme et l’indicateur `closed`. Une section peut donc être ouverte ou de forme non circulaire, notamment près d’une bifurcation.

Le filtre de centerlines VMTK ne se limite pas à un graphe des sommets de surface. Il recalcule les normales orientées, construit une Delaunay 3D, extrait les tétraèdres internes avec `vtkvmtkInternalTetrahedraExtractor`, construit le diagramme de Voronoï volumique et conserve les pôles. Le coût par défaut est `1/R`. Le fast marching non-manifold est exécuté sur le Voronoï, puis le backtracking utilise `vtkvmtkSteepestDescentLineTracer` avec la solution eikonale, le rayon, `EdgeArray` et `EdgePCoordArray`.

## Conséquence pour le cylindre observé

VMTK ne transforme pas directement une section contaminée en un cylindre propre. La section est d’abord une intersection géométrique avec la surface. Si l’intersection n’est pas fermée, l’algorithme le signale via `CenterlineSectionClosed`; il ne faut pas la loft-er comme une boucle valide.

Notre reconstruction actuelle reçoit déjà les points des sections, mais elle les traite comme des boucles toujours valides et les relie uniformément. Les diagnostics indiquent que les sections 90–94 de la branche 2 contiennent déjà des rayons médians de 28 à 49 et des rayons maximaux de 56 à 98, alors que les sections normales sont autour de 10–15. Les arrays `points` et `phase_locked_points` présentent les mêmes anomalies. Le cylindre est donc créé par le raccordement de contours anormaux; il n’est pas une preuve que les sections sont ignorées.

## Écarts principaux à corriger

1. Conserver et exploiter un indicateur `closed` pour interdire le loft d’une section ouverte.
2. Calculer et stocker aire, diamètre minimal, diamètre maximal et shape index pour chaque section.
3. Détecter les sections de bifurcation avant le loft et ne pas les traiter comme une série tubulaire simple.
4. Comparer la section à ses voisines avec un contrôle de variation du rayon, de l’aire et de la forme.
5. Raccorder chaque branche sur une zone de jonction tronquée ou utiliser une reconstruction de jonction dédiée.
6. Vérifier l’ordre et la phase des points du contour avant triangulation; le resampling seul ne corrige pas une boucle contaminée.

## Références

[1]: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataCenterlineSections.cxx "VMTK vtkvmtkPolyDataCenterlineSections.cxx"

[2]: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataCenterlineSections.h "VMTK vtkvmtkPolyDataCenterlineSections.h"

[3]: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataCenterlines.cxx "VMTK vtkvmtkPolyDataCenterlines.cxx"

[4]: https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkcenterlinesections.py "VMTK vmtkcenterlinesections.py"

[5]: https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkcenterlines.py "VMTK vmtkcenterlines.py"

## Reconstruction volumique PolyBall-like ajoutée

Un second test a été ajouté dans `examples/medical_build/reconstruct_vmtk_like_polyball.py`. Il utilise uniquement Python, NumPy et VTK : les points de centerline et `MaximumInscribedSphereRadius` définissent un champ implicite de type union de boules, puis `vtkMarchingCubes` extrait l’isosurface à zéro. Sur le centerline officiel à six branches, avec un pas de 0,75, le test produit 51 673 points et 101 620 triangles.

Cette étape reproduit l’idée de `vmtkcenterlinemodeller.py`/`vtkvmtkPolyBallModeller`, mais elle ne doit pas être confondue avec une reconstruction de la surface originale : elle reconstruit une géométrie à partir du centerline et des rayons. La surface VMTK d’origine reste la référence anatomique. Pour reproduire cette dernière, il faut conserver l’export surface original ou reconstruire les contours plan-surface avec gestion des coupes ouvertes et des bifurcations.
