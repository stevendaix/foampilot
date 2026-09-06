# Validation du reconstructeur STL manuel

Le script `section_stl_reconstruction.py` relie directement les points ordonnés des sections. Il nettoie les doublons, resample chaque contour avec un nombre commun de points, aligne les phases entre stations, triangule les bandes entre sections et ferme les deux extrémités par des éventails.

## Résultat sur la fixture reproductible

La fixture `minimal_analysis_contract.json` a été reconstruite avec 16 points par contour.

| Contrôle | Résultat |
|---|---:|
| Triangles | 96 |
| Sommets après fusion | 50 |
| Arêtes frontières internes | 0 |
| Arêtes non-manifold | 0 |
| Trimesh `watertight` après fusion des sommets | `true` |
| Orientation cohérente | `true` |
| Nombre d’Euler | 2 |
| Volume signé | 4.0 |
| VTK boundary edges | 0 |
| VTK non-manifold edges | 0 |

La lecture STL brute de Trimesh duplique naturellement les sommets de chaque facette et peut alors afficher `watertight=false`. Le contrôle correct utilise une fusion des sommets (`process=True`) puis confirme la fermeture. Le vérificateur fourni compare volontairement les deux lectures afin d’éviter ce faux diagnostic.

## Limite avant intégration complexe

Cette validation ne constitue pas encore une validation de l’aorte complexe. Le fichier complet des sections ordonnées n’est pas présent dans l’environnement courant ; seuls les centerlines et les rapports de campagne sont disponibles. Le script ne doit être déclaré validé sur l’aorte complexe qu’après exécution avec le JSON complet des sections et contrôle de chaque branche ainsi que du STL combiné.

La commande attendue est :

```bash
python examples/medical_build/section_stl_reconstruction.py \
  path/to/analysis_sections.json \
  --output output/manual_stl \
  --points 32

python examples/medical_build/verify_manual_stl.py \
  output/manual_stl/aorta_manual_sections.stl
```

L’intégration dans le cas complexe doit être acceptée seulement si `boundary_edges=0`, `nonmanifold_edges=0`, `watertight=true`, `winding_consistent=true`, un volume positif cohérent et une lecture VTK sans arêtes frontières sont obtenus.
