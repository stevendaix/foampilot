# Intégration NetworkX dans `medical_build`

## Rôle

NetworkX est utilisé comme couche de **topologie vasculaire** entre l’analyse géométrique et la reconstruction. Le graphe ne remplace ni VTK, ni le calcul des centerlines, ni l’union volumique STL. Il décrit les relations entre les caps, les bifurcations et les branches afin que les opérations géométriques disposent d’une structure anatomique explicite.

> Une arête du graphe représente une branche centerline ; un nœud représente un cap ou une jonction anatomique.

## Construction

Le module `vascular_graph.py` accepte un objet `GeometryAnalysisData`. Chaque `BranchRecord` fournit les identifiants `source_cap_id` et `target_cap_id`, les points de centerline, la longueur, les sections et les relations parent-enfants. Ces informations sont converties en un graphe NetworkX non orienté, avec les métriques géométriques conservées dans les attributs des arêtes.

Lorsque les identifiants de caps ne sont pas fiables ou sont locaux à chaque branche, `endpoint_tolerance` permet de fusionner les extrémités spatialement proches. Cette opération est utile avec des résultats d’extraction où deux branches aboutissent au même point anatomique mais possèdent des identifiants distincts.

```python
from foampilot.geometry.medical_build import build_vascular_graph

graph = build_vascular_graph(analysis_data, endpoint_tolerance=0.5)
status = graph.validate()
graph.save_json("analysis/vascular_graph.json")
```

## Contrôles effectués

| Contrôle | Utilité |
|---|---|
| Connectivité | Détecte une branche isolée ou une bifurcation non raccordée |
| Composantes connexes | Vérifie que le réseau anatomique est globalement unique |
| Acyclicité | Détecte les cycles topologiques anormaux dans un arbre artériel |
| Degré des nœuds | Identifie les extrémités et bifurcations |
| Branchements parent-enfant | Permet d’ordonner les reconstructions et les patches CFD |
| Positions des extrémités | Permet de corriger les identifiants de caps incohérents |

## Utilisation pour le STL et le volume

Le graphe doit être utilisé avant l’union STL pour déterminer quelles branches partagent une bifurcation. La somme des volumes intégrés par section ne doit pas être considérée comme le volume de l’aorte entière tant que les zones communes ne sont pas partitionnées. La procédure correcte est la suivante : les sections fournissent les volumes locaux ; le graphe identifie les relations de partage ; une règle de partition retire les segments communs ; l’union volumique produit enfin le volume global fermé.

NetworkX contrôle donc la **cohérence topologique**, tandis que `trimesh` ou VTK contrôlent la **fermeture géométrique**, les composantes, les arêtes frontière, les arêtes non-manifold et le volume final.

## Installation

NetworkX est maintenant déclaré comme dépendance directe de foampilot dans `pyproject.toml` avec `networkx>=3.0`. Cette déclaration est volontaire : même si d’autres bibliothèques peuvent l’installer indirectement, la pipeline medical_build l’utilise explicitement et doit rester reproductible dans un environnement propre.
