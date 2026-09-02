# Déformation locale optionnelle

## Objectif

`medical_build` fournit maintenant une déformation géométrique locale inspirée du workflow paramétrique d’AneuPy. Elle agit sur les contours des sections déjà extraites par notre pipeline Python. Elle ne modifie ni la centerline ni les données d’analyse originales.

La fonctionnalité est désactivée par défaut. Lorsque `spec=None` ou `amplitude=0`, `apply_local_deformation` retourne une copie équivalente des données de référence.

## Modèle radial

Pour une section de centre `c` et un point de contour `x`, le déplacement radial est :

\[
 x' = c + \lambda(s)(x-c)
\]

avec :

\[
 \lambda(s)=1+a\exp\left[-\frac{1}{2}\left(\frac{s-s_0}{\sigma}\right)^2\right]b(s)
\]

Ici `a` est l’amplitude relative, `s0` la position du maximum, `sigma` la largeur et `b(s)` un facteur de protection des jonctions. Le facteur `b` annule progressivement la déformation près des extrémités d’une branche lorsque `junction_protection` est utilisé.

## Utilisation

```python
from foampilot.geometry.medical_build import (
    LocalDeformationSpec,
    apply_local_deformation,
    deformation_report,
)

spec = LocalDeformationSpec(
    branch_ids=(2,),
    center_abscissa=12.0,
    sigma=3.0,
    amplitude=0.20,
    junction_protection=2.0,
)

deformed = apply_local_deformation(analysis_data, spec)
report = deformation_report(deformed)
```

L’objet retourné peut ensuite être transmis aux reconstructeurs existants pour produire un STL, une géométrie Build123d ou les patches OpenFOAM. La déformation est donc située entre l’analyse et la reconstruction :

```text
GeometryAnalysisData de référence
        ↓
apply_local_deformation(...)
        ↓
GeometryAnalysisData déformée
        ↓
STL / Build123d / snappyHexMesh
```

## Garanties de non-régression

L’objet source est copié en profondeur et n’est jamais modifié. Les valeurs de section, aire, périmètre et rayon équivalent sont recalculées uniquement dans la copie déformée. Les métadonnées enregistrent les paramètres et le facteur d’échelle de chaque section touchée.

Les tests vérifient que `spec=None` est un no-op exact, que l’objet source reste inchangé, que la déformation est plus forte au centre de la fenêtre et que les extrémités sont protégées. Les sept tests `medical_build` passent.

## Limites actuelles

La première version est radiale et conserve la forme locale du contour. Elle ne réalise pas encore de déformation anisotrope en fonction d’un angle, ni de déplacement de centerline, ni de recalcul mécanique de paroi. Ces extensions pourront être ajoutées séparément sans modifier l’API de référence.
