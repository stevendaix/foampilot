# Audit détaillé de la reconstruction VMTK

## Distinction fondamentale

VMTK ne reconstruit pas directement une surface CAD à partir des polylines dans `vmtkcenterlinesections.py`. Ce script appelle `vtkvmtkPolyDataCenterlineSections`, qui produit des sections d’analyse par coupe de la surface d’entrée avec un plan local perpendiculaire à la tangente.

La section est calculée pour chaque point du centerline. La tangente C++ est la moyenne normalisée des directions des segments précédent et suivant. À chaque point, VMTK appelle `vtkvmtkPolyDataBranchSections::ExtractCylinderSection(input, point, tangent, section, closed)`. Il récupère ensuite la cellule 0 du résultat, calcule l’aire, les tailles minimale/maximale, le shape index et l’état `closed`, et écrit ces valeurs à la fois dans les cellules de sortie et dans les arrays point-data du centerline.

## Gestion VMTK des branches

`vmtkbranchsections.py` n’utilise pas une section à chaque point. Il attend une surface et des centerlines déjà split en branches, avec les arrays `GroupIds`, `CenterlineIds`, `TractIds` et `Blanking`. Il exclut les groupes blanked via `GetNonBlankedGroupsIdList`.

Pour une branche, la coupe est située à une distance exprimée en nombre de sphères inscrites tangentes (`NumberOfDistanceSpheres`) depuis le début ou la fin. Cette distance n’est donc pas un index arbitraire ni une distance euclidienne fixe.

`vmtkbifurcationsections.py` traite séparément les groupes blanked, c’est-à-dire les tracts redondants de bifurcation. Il cherche les groupes amont et aval via les utilitaires centerline, calcule les vecteurs de bifurcation et extrait une section de bifurcation dédiée. Les tracts blanked ne sont donc pas loftés comme des branches ordinaires.

## Gestion PolyBall

`vmtkcenterlinemodeller.py` appelle `vtkvmtkPolyBallModeller`, active `UsePolyBallLineOn()` et passe le rayon `MaximumInscribedSphereRadius`. Il s’agit d’une fonction tube continue sur les segments, et non d’une union discrète de sphères centrées sur les points.

## Écarts identifiés dans les prototypes locaux

1. Le premier export choisissait `max(contours, key=len)` alors que les sections doivent rester associées à la coupe de la branche et à son état de fermeture.
2. Les contours JSON perdaient le point final répété ; l’état `closed` doit donc être conservé explicitement.
3. Les profils ouverts ou de bifurcation étaient envoyés au loft comme des boucles fermées.
4. Les STL de branches contiennent le tronc commun complet pour plusieurs trajectoires ; leur append compte donc plusieurs fois le même volume.
5. L’union implicite directe des STL produit des nappes parasites et ne reproduit pas PolyBall.
6. Le test VMTK six-cellules n’est pas le cas complexe : le cas complexe courant possède 8 cellules et 8 terminaux.
7. Une validation globale de volume/topologie ne valide pas la sélection locale des intersections.

## Méthode de correction retenue

La reproduction Python doit suivre cette chaîne :

```text
surface cappée propre
→ centerlines split avec arrays VMTK
→ exclusion des tracts blanked
→ pour chaque branche valide : sections plan-surface
→ pour chaque jonction : coupe dédiée et zone exclue des lofts
→ reconstruction des tronçons non-junction
→ assemblage d’une jonction unique
→ caps entrée/sorties
→ validation locale puis globale
```

La prochaine implémentation doit donc partir des données originales `branch_00.npz` à `branch_07.npz` et de `aorta_surface_patches.vtp`, en conservant `Blanking`, `GroupIds`, `CenterlineIds` et `TractIds`. Les anciens STL manuels ne doivent pas être utilisés comme entrée de reconstruction.

## Références officielles

- [1] `vmtkcenterlinesections.py`: https://raw.githubusercontent.com/vmtk/vmtk/master/vmtkScripts/vmtkcenterlinesections.py
- [2] `vmtkcenterlinemodeller.py`: https://raw.githubusercontent.com/vmtk/vmtk/master/vmtkScripts/vmtkcenterlinemodeller.py
- [3] `vtkvmtkPolyDataCenterlineSections.cxx`: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataCenterlineSections.cxx
- [4] `vtkvmtkPolyDataBranchSections.cxx`: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataBranchSections.cxx
- [5] `vtkvmtkPolyDataBranchSections.h`: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataBranchSections.h
- [6] `vtkvmtkPolyDataBifurcationSections.cxx`: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataBifurcationSections.cxx
- [7] `vmtkbranchsections.py`: https://raw.githubusercontent.com/vmtk/vmtk/master/vmtkScripts/vmtkbranchsections.py
