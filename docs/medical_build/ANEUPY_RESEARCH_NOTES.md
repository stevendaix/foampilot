# Notes d’analyse AneuPy

## Sources consultées

- Article : https://arxiv.org/html/2504.15285v1
- Dépôt : https://github.com/mdeluci/AneuPy

## Principes relevés

AneuPy suit un workflow centré sur une centerline : import de points XYZ, lissage par spline cubique, extraction de points d’intérêt, interpolation d’une courbe, calcul des tangentes, génération de sections circulaires orientées dans un repère local, interpolation NURBS des contours, création d’une enveloppe fermée puis création d’un solide.

La déformation anatomique est introduite par les données de rayon/aire interpolées le long de la centerline et par les paramètres de section. Le script `Patient_specific.py` convertit l’aire en rayon avec `R = sqrt(A / pi)`, interpole le rayon le long de la longueur normalisée, place une section à chaque point centerline et oriente la section selon la tangente si `use_tangent_normal` est activé.

Le code AneuPy utilise SALOME/GEOM pour les opérations CAD : `Section`, `Shell`, `Solid`, `MakeInterpol`, `MakeFilling`, `MakeSewing`, `MakeSolid` et `MakeCut`. Cette dépendance SALOME ne doit pas être introduite dans la production foampilot si l’objectif reste une pipeline Python/VTK/Build123d indépendante.

## Adaptation proposée à medical_build

Introduire un objet optionnel `LocalDeformationSpec` agissant sur les sections existantes, sans modifier les centerlines ni les données de référence par défaut. La déformation serait un champ scalaire local `g(s)` le long de chaque branche, par exemple une gaussienne ou une bosse compacte, appliqué au rayon ou au contour dans le repère local de la section. Pour une section circulaire : `r'(s) = r(s) * (1 + amplitude * g(s))`. Pour une déformation non axisymétrique, utiliser `r'(s,theta) = r(s) * (1 + amplitude * g(s) * h(theta))`, avec amplitude bornée et conservation optionnelle de l’aire.

La méthode doit être désactivée par défaut, créer de nouvelles données de sections, écrire un rapport JSON des paramètres et métriques, puis passer par les reconstructeurs existants STL/Build123d/snappyHexMesh. Les tests doivent vérifier que `spec=None` reproduit exactement les sorties du cas de référence et que la déformation locale reste limitée à la fenêtre spatiale demandée.

## Limites AneuPy

AneuPy vise surtout les AAA et génère des sections circulaires à partir de rayons/aires. Il ne fournit pas directement une solution pour préserver nos contours VMTK non circulaires, les bifurcations multi-branches et les patches CFD. Il faut donc réutiliser son idée de champ de rayon local, mais l’appliquer dans notre contrat `GeometryAnalysisData` et notre graphe NetworkX, branche par branche.
