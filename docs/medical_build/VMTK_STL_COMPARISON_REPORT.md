# Comparaison directe VMTK / STL foampilot

## Installation de référence

VMTK `1.5.0` a été installé avec VTK `9.6.2` dans l’environnement Python `3.12`. Les filtres C++ suivants sont disponibles : `vtkvmtkPolyDataCenterlines`, `vtkvmtkInternalTetrahedraExtractor`, `vtkvmtkSimpleCapPolyData` et `vtkvmtkPolyDataCenterlineSections`.

Le filtre nommé `vtkvmtkDelaunayVoronoi` n’est pas exposé sous ce nom dans le module Python installé ; la fonctionnalité correspondante est distribuée sous d’autres classes du paquet. Cette différence de nom ne bloque pas la validation surfacique.

## Surface de référence

La surface officielle a été retrouvée dans les fixtures VMTK :

`/tmp/vmtk-test-data/input/aorta-surface.stl`

Elle contient 6468 points et 12932 cellules, une composante, zéro arête frontière, une aire de 4517,7631 unités² et un volume de 13184,2667 unités³.

## Comparaison des variantes

| STL | Pas | Fermeture | Composantes | Volume | Erreur volume | Erreur aire |
|---|---:|---:|---:|---:|---:|---:|
| Union brute | 1,00 mm | 0 | 1 | 12918,25 | −2,02 % | +31,38 % |
| Union brute | 0,75 mm | 0 | 2 | 12047,40 | −8,62 % | +30,56 % |
| Union brute | 0,50 mm | 0 | 4 | 11117,44 | −15,68 % | +32,22 % |
| Union corrigée | 0,50 mm | 1 | 1 | 11238,10 | −14,76 % | +24,92 % |
| Union corrigée | 0,50 mm | 2 | 1 | 11407,29 | −13,48 % | +20,72 % |
| Union corrigée | 0,50 mm | 3 | 1 | 11579,89 | −12,17 % | +17,14 % |

Toutes les variantes fermées possèdent zéro arête frontière et zéro arête non-manifold. La surface de référence VMTK possède les mêmes propriétés topologiques.

## Interprétation

L’installation de VMTK permet maintenant une comparaison indépendante et montre que le STL global est topologiquement valide, mais encore trop petit en volume par rapport à la surface de référence. La variante initiale recommandée à 0,5 mm avec une fermeture minimale est robuste pour la topologie, mais son volume est inférieur de 14,76 % à la référence VMTK.

La variante 1,0 mm est la meilleure uniquement selon le volume, avec une erreur de −2,02 %, mais elle est plus grossière et possède une erreur d’aire de +31,38 %. La variante 0,5 mm avec trois itérations est la meilleure parmi les variantes testées selon l’aire, avec une erreur de +17,14 %, mais elle reste à −12,17 % sur le volume.

La conclusion est que la perte de volume provient principalement des STL de branches et de leur raccordement, et pas seulement de la résolution Marching Cubes. Il faut donc comparer les branches individuelles à la surface VMTK, puis améliorer les sections ou les extrémités avant de choisir une résolution globale.

Le script reproductible est `examples/medical_build/compare_vmtk_stl.py`. Le résultat machine-readable est `examples/medical_build/outputs/aorta_vmtk_reference_vs_foampilot.json`.
