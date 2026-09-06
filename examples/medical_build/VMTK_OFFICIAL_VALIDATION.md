# Validation officielle VMTK

Cette procédure utilise les fixtures du dépôt officiel [vmtk-test-data](https://github.com/vmtk/vmtk-test-data) afin de vérifier les entrées nécessaires à une reconstruction anatomique par intersections surface-plan.

## Données validées

La fixture contient une surface aortique triangulée de 6 468 points et 12 932 cellules, une centerline de 409 points et 2 lignes, ainsi qu’une centerline branchée de 417 points et 6 lignes. Les références de surface STL ouverte, surface séparée par branches, connectivité, segment local, système de référence centerline et volume tétraédrique sont également présentes.

## Commande

Depuis la racine du dépôt :

```bash
python3 examples/medical_build/validate_vmtk_official_fixture.py \
    foampilot/test/vmtk_test_data \
    --output /tmp/vmtk_official_validation.json
```

Le script accepte également le checkout direct du dépôt officiel, dont les fichiers se trouvent sous `input/`.

## Contrôles réalisés

Le script lit la surface, les centerlines et les branches avec VTK, vérifie les dimensions et les bornes, confirme la présence des références, puis exécute des intersections de plans d’échantillonnage avec la surface. Sur la campagne exécutée, les 13 intersections échantillonnées ont produit des cellules et des points non nuls, avec un temps total de 0,296 seconde.

| Objet | Résultat |
|---|---:|
| Surface | 6 468 points, 12 932 cellules |
| Centerline principale | 409 points, 2 lignes |
| Centerlines branchées | 417 points, 6 lignes |
| Références attendues | 8/8 présentes |
| Intersections de plans échantillonnées | 13/13 non vides |
| Temps de validation | 0,296 s |

## Utilisation pour le STL global

Cette fixture constitue le cas de validation reproductible pour l’étape suivante : extraire les boucles fermées par plans perpendiculaires aux tangentes, les exprimer dans les repères transport-parallèle, reconstruire les surfaces de branches, appliquer une union volumique ou voxelisée au niveau des bifurcations, extraire une isosurface unique et comparer le résultat à `aorta-surface-connectivity-reference.stl`.

Elle ne doit pas être confondue avec l’aorte complexe de la campagne foampilot. Cette dernière contient 1 694 points centerline, 8 branches et des arrays avancés ; sa surface source correspondante n’a pas été retrouvée dans le dépôt officiel VMTK.
