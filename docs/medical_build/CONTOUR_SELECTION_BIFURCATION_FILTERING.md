# Sélection des contours et filtrage des bifurcations

## 1. Principe général

Une coupe de surface par un plan de centerline ne doit jamais être réduite directement à `max(contours, key=len)`. Une coupe peut produire plusieurs composantes : une boucle fermée valide, plusieurs boucles, une ligne ouverte ou un contour traversant une bifurcation.

La pipeline doit conserver toutes les composantes, calculer leurs métriques, les associer à la branche par géométrie, puis décider si la station est valide pour un loft.

```text
surface + centerline
        ↓
vtkCutter
        ↓
vtkStripper sans perdre les cellules
        ↓
composantes de contour
        ↓
fermeture, aire, diamètre, forme, orientation, distance au centre
        ↓
association à la branche
        ↓
station VALID / JUNCTION / INVALID
        ↓
loft uniquement des stations VALID
```

## 2. Ne pas utiliser seulement `vtkStripper`

`vtkStripper` peut relier plusieurs segments contigus. Pour le diagnostic, il faut d’abord extraire les cellules de la sortie de `vtkCutter` et construire un graphe des segments. La fermeture doit être mesurée avant toute fermeture artificielle.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ContourCandidate:
    points: np.ndarray
    closed: bool
    length: float
    area: float
    radius_min: float
    radius_median: float
    radius_max: float
    shape: float
    centroid: np.ndarray
    score: float = -np.inf
    reason: str = ""
```

Après la coupe, chaque polyline est convertie dans le repère du plan. Le repère est défini par le centre `c`, la tangente `t` et deux vecteurs orthonormés `u`, `v`.

```python
def plane_basis(tangent):
    t = np.asarray(tangent, dtype=float)
    t /= max(np.linalg.norm(t), 1e-12)
    ref = np.array([1., 0., 0.]) if abs(t[0]) < 0.8 else np.array([0., 1., 0.])
    u = ref - np.dot(ref, t) * t
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(t, u)
    v /= max(np.linalg.norm(v), 1e-12)
    return u, v, t


def project_to_plane(points, center, tangent):
    u, v, t = plane_basis(tangent)
    q = np.asarray(points) - np.asarray(center)
    q = q - (q @ t)[:, None] * t[None, :]
    return np.column_stack((q @ u, q @ v)), u, v, t
```

## 3. Métriques d’une composante

La fermeture est vraie uniquement si le premier et le dernier point sont proches ou si le graphe de segments possède une boucle. Il ne faut jamais fermer une ligne ouverte en ajoutant arbitrairement une arête.

```python
def polygon_area_2d(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def contour_metrics(points, center, tangent):
    xy, _, _, _ = project_to_plane(points, center, tangent)
    radial = np.linalg.norm(xy, axis=1)
    closed = len(points) >= 3 and np.linalg.norm(points[0] - points[-1]) <= 1e-5
    if closed:
        xy_area = polygon_area_2d(xy[:-1]) if len(xy) > 3 else 0.0
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    else:
        xy_area = 0.0
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return {
        "closed": bool(closed),
        "area": float(xy_area),
        "length": length,
        "radius_min": float(radial.min()),
        "radius_median": float(np.median(radial)),
        "radius_max": float(radial.max()),
        "shape": float(radial.min() / max(radial.max(), 1e-12)),
        "centroid": np.mean(points, axis=0),
    }
```

La métrique de forme doit être calculée à partir des diamètres du contour, comme dans VMTK, et non uniquement à partir d’un rayon équivalent. Pour une première version robuste, utiliser aussi les quantiles afin d’éviter qu’un seul point contaminé domine la décision :

```python
q05, q50, q95 = np.quantile(radial, [0.05, 0.50, 0.95])
outlier_ratio = q95 / max(q50, 1e-12)
```

## 4. Association d’un contour à la branche

À une station d’une branche `b`, une composante est candidate si son centroïde est proche du point de centerline et si sa normale de plan est cohérente avec la tangente. La distance au centre seule ne suffit pas, car plusieurs branches peuvent traverser la même zone.

On calcule un score :

```python
def branch_score(candidate, center, tangent, expected_radius, previous_centroid=None):
    d = np.linalg.norm(candidate.centroid - center)
    radius_error = abs(candidate.radius_median - expected_radius) / max(expected_radius, 1e-12)
    continuity = 0.0
    if previous_centroid is not None:
        continuity = np.linalg.norm(candidate.centroid - previous_centroid)
    return 3.0 * d + 8.0 * radius_error + 0.5 * continuity
```

La sélection se fait uniquement parmi les contours fermés. Si plusieurs composantes fermées ont des scores proches, la station est marquée `JUNCTION` et n’est pas envoyée au loft de la branche.

```python
def choose_component(candidates, center, tangent, expected_radius, previous_centroid=None):
    valid = [c for c in candidates if c.closed]
    if not valid:
        return None, "OPEN_OR_EMPTY"
    for c in valid:
        c.score = branch_score(c, center, tangent, expected_radius, previous_centroid)
    valid.sort(key=lambda c: c.score)
    best = valid[0]
    if len(valid) > 1:
        second = valid[1]
        if second.score <= best.score * 1.15 + 1e-6:
            return None, "AMBIGUOUS_MULTIPLE_LOOPS"
    return best, "SELECTED"
```

## 5. Détection robuste d’une bifurcation

Une bifurcation doit être détectée à partir de la topologie des centerlines et non uniquement à partir d’un rayon anormal. Une station est une zone de jonction si plusieurs centerline cells sont proches du même point, si les tangentes divergent, ou si plusieurs composantes de coupe sont présentes.

```python
def detect_junction(center, tangent, all_branch_points,
                    radius, candidates, distance_factor=2.5,
                    angle_threshold_deg=25.0):
    nearby = []
    for branch_id, p, t in all_branch_points:
        d = np.linalg.norm(np.asarray(p) - center)
        if d <= distance_factor * max(radius, 1e-12):
            cosang = abs(np.dot(tangent, t) /
                         max(np.linalg.norm(tangent) * np.linalg.norm(t), 1e-12))
            angle = np.degrees(np.arccos(np.clip(cosang, -1., 1.)))
            if angle >= angle_threshold_deg:
                nearby.append((branch_id, d, angle))
    return len(nearby) > 0 or len(candidates) != 1
```

La zone de jonction doit être étendue sur une distance physique, par exemple `2 à 3 fois le rayon local` ou `0,5 à 1 diamètre` de chaque côté du nœud. Toutes les stations de cette zone doivent être conservées dans le rapport, mais exclues du loft branché ordinaire.

## 6. Filtrage par continuité longitudinale

Un contour valide ne doit pas présenter une variation brutale par rapport aux stations voisines. Les seuils doivent être relatifs à la médiane locale et non codés en unités absolues.

```python
def reject_by_continuity(current, previous, next_,
                         max_radius_ratio=1.8,
                         max_area_ratio=2.5,
                         min_shape=0.35):
    if not current.closed:
        return True, "OPEN"
    if current.shape < min_shape:
        return True, "BAD_SHAPE"
    neighbors = [x for x in (previous, next_) if x is not None and x.closed]
    if not neighbors:
        return False, "NO_NEIGHBOR_REFERENCE"
    ref_radius = np.median([x.radius_median for x in neighbors])
    ref_area = np.median([x.area for x in neighbors if x.area > 0])
    if current.radius_median / max(ref_radius, 1e-12) > max_radius_ratio:
        return True, "RADIUS_SPIKE"
    if ref_area > 0 and current.area / ref_area > max_area_ratio:
        return True, "AREA_SPIKE"
    return False, "OK"
```

Pour le cas observé, les sections 90–94 de la branche 2 seraient rejetées car leur rayon médian est compris entre environ 28 et 49 alors que les voisins normaux sont autour de 10–15. Elles doivent être marquées dans le JSON, pas supprimées silencieusement.

## 7. Règle de reconstruction

Le reconstructeur doit recevoir uniquement les stations valides :

```python
valid_sections = [
    s for s in sections
    if s.status == "VALID" and s.closed
]
```

Il faut conserver au moins deux sections valides de chaque côté de la zone de jonction. Le début du loft de chaque branche doit donc être déplacé après la zone de bifurcation. La jonction centrale ne doit pas être fabriquée par le loft indépendant des branches.

```python
branch_sections = [s for s in sections if s.status == "VALID"]
if len(branch_sections) < 2:
    raise ValueError("No valid branch profiles after bifurcation filtering")
```

La jonction peut ensuite être traitée selon deux niveaux :

1. Une première version robuste utilise un raccord voxel/implicit global autour du nœud, puis fusionne ce raccord avec les tronçons valides.
2. Une version CAD avancée utilise une surface de transition multi-branches ou un patch paramétrique, mais ne doit pas fermer les profils ouverts par une simple arête.

## 8. Champs à ajouter dans `SectionLoftInput`

```python
@dataclass
class SectionLoftInput:
    center: np.ndarray
    points: np.ndarray
    tangent: np.ndarray
    radius: float
    metadata: dict
    closed: bool = True
    area: float = 0.0
    min_size: float = 0.0
    max_size: float = 0.0
    shape: float = 1.0
    status: str = "VALID"
    rejection_reason: str = ""
```

`normalize_sections()` doit lire ces valeurs depuis le JSON et conserver les valeurs originales. Le backend Build123d ne doit pas recalculer silencieusement ces propriétés.

## 9. Validation obligatoire

La validation doit comporter deux niveaux. Le niveau global vérifie le volume, l’aire, la fermeture et le nombre de composantes. Le niveau local vérifie toutes les stations : nombre de composantes, fermeture, aire, diamètre, forme et distance du contour au centerline.

| Test | Critère recommandé |
|---|---|
| Contour ouvert utilisé par un loft | Interdit |
| Deux boucles concurrentes | Station `JUNCTION`, loft interdit |
| Rayon médian / voisin | inférieur à 1,8 |
| Aire / voisin | inférieur à 2,5 |
| Shape index | supérieur à 0,35 |
| Variation de tangente | signaler au-delà de 25° |
| Centre du contour / centerline | inférieur à 0,5 rayon local |
| Sections rejetées | Exportées dans le rapport JSON |

Ces seuils sont des valeurs initiales. Ils doivent être calibrés sur les données VMTK six branches et non imposés comme une vérité anatomique universelle.

## 10. Fichiers à modifier

La logique d’extraction doit être placée dans un module séparé, par exemple `medical_build/section_filtering.py`, afin de ne pas modifier l’analyse de référence. `export_vmtk_like_sections.py` doit produire tous les candidats et leurs diagnostics. `analysis_data.py` doit stocker le statut et les métriques. `reconstruction.py` doit ignorer les sections non `VALID`. La visualisation doit afficher les sections valides en vert, les sections rejetées en rouge et les jonctions en orange.

## Références

[1]: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataCenterlineSections.cxx "VMTK PolyDataCenterlineSections C++"

[2]: https://raw.githubusercontent.com/vmtk/vmtk/master/vtkVmtk/ComputationalGeometry/vtkvmtkPolyDataBranchSections.cxx "VMTK PolyDataBranchSections C++"

[3]: https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkcenterlinesections.py "VMTK centerline sections script"
