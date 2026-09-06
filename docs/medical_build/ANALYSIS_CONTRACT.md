# Contrat de données de `MedicalBuildAnalysis`

## Objectif

La phase d’analyse extrait et vérifie toutes les données nécessaires à une reconstruction ultérieure sans relancer les calculs VMTK-like. Elle ne crée aucun loft Build123d. Son résultat est un paquet géométrique sérialisable, exportable et réutilisable par plusieurs stratégies de reconstruction.

## Découpage architectural

```text
Surface d’entrée
      |
      v
MedicalBuildAnalysis
  - preprocessing
  - capage et topologie
  - Delaunay / Voronoi
  - seeds et pôles
  - fast marching / backtracking
  - branches et bifurcations
  - sections et repères
      |
      v
GeometryAnalysisData
      |
      +--> ReconstructionStrategy: Build123d polygonal
      +--> ReconstructionStrategy: Build123d circulaire
      +--> ReconstructionStrategy: Build123d elliptique
      +--> ReconstructionStrategy: export CFD / maillage
```

## Contenu minimal obligatoire

| Groupe | Données | Usage |
|---|---|---|
| Surface | surface originale, surface prétraitée, surface cappée | contrôle et reconstruction |
| Caps | identifiant, centre, normale, aire, boucle, rôle inlet/outlet | position des entrées/sorties |
| Delaunay | points, tétraèdres, cellules internes, volumes | audit topologique |
| Voronoi | sommets, arêtes, rayon inscrit, adjacency | centerline et diagnostics |
| Centerlines | points par branche, ordre, longueur, abscisses, tangentes | reconstruction |
| Branches | id, cap source, cap cible, parent, enfants, bifurcations | loft par branche |
| Sections | branche, station, centre, tangent, normale, binormale, points, aire, rayon, périmètre | profils CAD et CFD |
| Qualité | validité, pcoords, reachability, fallbacks, warnings | vérification |
| Performance | temps par phase, accélération, version backend | benchmark |

## Contrat d’une section

Chaque section doit contenir :

```python
{
    "branch_id": int,
    "station_id": int,
    "abscissa": float,
    "center": [x, y, z],
    "tangent": [tx, ty, tz],
    "normal": [nx, ny, nz],
    "binormal": [bx, by, bz],
    "points": [[x, y, z], ...],
    "phase_locked_points": [[x, y, z], ...],
    "area": float,
    "perimeter": float,
    "equivalent_radius": float,
    "valid": bool,
    "metadata": dict,
}
```

Les points doivent être ordonnés, fermés implicitement ou explicitement selon le backend, sans doublons consécutifs. Le repère doit être orthonormé à une tolérance documentée. La section doit être marquée invalide plutôt que silencieusement supprimée si son aire, son orientation ou sa fermeture sont incorrectes.

## Contrat d’une branche

```python
{
    "branch_id": int,
    "source_cap_id": int,
    "target_cap_id": int,
    "parent_branch_id": int | None,
    "children_branch_ids": [int],
    "points": [[x, y, z], ...],
    "abscissas": [float, ...],
    "tangents": [[tx, ty, tz], ...],
    "length": float,
    "sections": [section_records],
    "diagnostics": dict,
}
```

Une branche ne doit jamais être reconstruite à partir d’une polyline réseau aplatie. Chaque branche conserve sa propre progression, ses sections et son association source-cible.

## Sérialisation

Le paquet doit être exportable en deux formes complémentaires :

1. un fichier JSON pour les métadonnées, diagnostics, caps, branches et statistiques ;
2. un fichier NPZ ou VTP pour les tableaux numériques et les géométries lourdes.

La reconstruction doit pouvoir être relancée depuis ces fichiers sans surface originale, lorsque les contours de sections et les repères sont présents.

## Critères de validation

La phase d’analyse est valide si :

- toutes les branches attendues possèdent une paire source-cible ;
- les points, tangentes, pcoords et rayons sont finis ;
- les pcoords sont dans `[0, 1]` avec la tolérance définie ;
- les sections valides sont ordonnées par abscisse croissante ;
- les contours ne contiennent pas de segment nul ;
- chaque section possède au moins trois points ;
- les warnings topologiques sont conservés dans le rapport ;
- le paquet peut être sérialisé puis rechargé sans perte numérique significative.

## Principe d’optimisation

Les essais Build123d doivent consommer exactement le même `GeometryAnalysisData`. Une modification de stratégie CAD ne doit donc jamais modifier les centerlines, les sections VMTK, les seeds ou les diagnostics. Toute amélioration de reconstruction sera comparée à un même paquet d’analyse figé.


## Implémentation validée — paquet `schema v1`

La première exportation réelle du contrat est disponible dans :

`/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package/analysis_sections.json`

Pour chaque branche, le tableau numérique correspondant est enregistré dans `sections_branch_XX.npz`. Le NPZ contient les tableaux `raw_points`, `phase_locked_points`, `offsets`, `station_ids`, `abscissas`, `centers`, `tangents`, `normals`, `binormals`, `area`, `perimeter` et `equivalent_radius`. Les contours sont stockés sans répétition du premier point final ; la fermeture est implicite et doit être appliquée par le consommateur CAD ou CFD.

La campagne réelle sur la surface cappée a produit 100 sections par branche pour les 8 branches, soit 800 sections. Les fichiers lourds associés sont `capped_surface.vtp`, `delaunay.vtu` et `voronoi.vtp`.

Le rapport `sections_validation.json` confirme :

| Contrôle | Résultat |
|---|---:|
| Branches exportées | 8 |
| Sections exportées | 800 |
| Sections par branche | 100 |
| Stations monotones et uniques | Oui |
| Repères orthonormés | Oui, erreur maximale < 1e-6 |
| Aires positives | Oui |
| Profils contenant au moins 3 points | Oui |
| Distance maximale centre-section / centerline discrète | 0,500 environ |
| Statut global | `all_ok=true` |

La distance centre-section indiquée est une borne conservative calculée vers les points centerline discrets ; elle ne constitue pas une erreur de section, car les stations sont interpolées sur la polyline et peuvent se situer entre deux points de centerline. La validation fine devra utiliser l’abscisse curviligne interpolée, et non le seul voisin discret, lors de la prochaine phase de non-régression.

Une correction importante a également été appliquée à l’extracteur local : le champ `LocalSection.tangent` contient désormais la tangente de la section, et non plus le binormal du repère de transport parallèle. Cette correction est nécessaire pour rendre le contrat `SectionRecord.tangent` cohérent avec le centerline.
