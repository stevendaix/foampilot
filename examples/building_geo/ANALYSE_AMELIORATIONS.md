# Analyse et améliorations de `examples/building_geo`

## Résumé

L’analyse du dossier [`examples/building_geo`](https://github.com/stevendaix/foampilot/tree/main/examples%2Fbuilding_geo) a mis en évidence trois problèmes principaux : des chemins d’import et de commande incohérents avec l’arborescence réelle du dépôt, une géométrie VoxCity remplacée par des boîtes englobantes rectangulaires, et une chaîne Gmsh qui recréait des groupes physiques sans noms après le maillage.

Les corrections ont été appliquées sur la branche `improve-building-geo`.

## Problèmes identifiés

| Gravité | Problème | Conséquence |
| --- | --- | --- |
| Élevée | Plusieurs scripts utilisaient des chemins relatifs incorrects vers `foampilot/src`, notamment depuis `neighborhood_demo` et `voxcity_export_work`. | Les exemples échouaient avec `ModuleNotFoundError: foampilot` lorsqu’ils étaient lancés depuis le dépôt cloné. |
| Élevée | Les commandes documentées utilisaient `PYTHONPATH=../../foampilot/src` ou un nombre de niveaux incorrect dans les sous-dossiers. | Les commandes copiées depuis la documentation ne pouvaient pas résoudre les modules internes. |
| Élevée | La fusion des empreintes dans `neighborhood_demo/generate.py` ignorait effectivement la distance de rapprochement et attribuait ensuite la hauteur maximale de tout le jeu de données à chaque bâtiment fusionné. | La géométrie pouvait être artificiellement fusionnée et les hauteurs de bâtiments pouvaient être fortement surestimées, ce qui faussait le volume bâti et la simulation aérodynamique. |
| Critique | Le builder `vector_builder_build123.py` construisait chaque bâtiment avec `gmsh.model.occ.addBox(...)` à partir de `footprint.bounds`, au lieu d’extruder le contour polygonal VoxCity. | Les bâtiments en L, T, avec cour intérieure ou contour irrégulier devenaient des pavés rectangulaires, ce qui expliquait directement la mauvaise géométrie et les booléens Gmsh fragiles. |
| Élevée | Les physical groups étaient supprimés puis recréés après le maillage dans certains chemins, ce qui produisait des groupes sans noms dans le fichier Gmsh. | L’export OpenFOAM pouvait perdre les patches `inlet`, `outlet`, `ground`, `top` et `buildings`. |
| Moyenne | La présence de Gmsh n’était pas vérifiable dans l’environnement d’exécution utilisé pour les tests. | Les imports complets de la chaîne de génération ne peuvent pas être validés sans installer les dépendances scientifiques optionnelles. |

## Améliorations appliquées

Les chemins Python des scripts situés directement dans `examples/building_geo` pointent désormais vers `foampilot/src/`. Les scripts de `neighborhood_demo` utilisent le niveau de parent approprié, et les utilitaires de `voxcity_export_work/src` utilisent également la racine correcte du dépôt. Les exemples et les notes d’exécution ont été synchronisés avec ces chemins corrigés.

La fonction de fusion des bâtiments a été refactorisée pour regrouper les empreintes par hauteur comparable, appliquer réellement une fusion par enveloppes tamponnées selon la distance demandée, puis conserver une hauteur représentative calculée par moyenne pondérée par la surface. Les étapes de nettoyage, de visualisation et de création des objets `Building` utilisent maintenant les couples `(empreinte, hauteur)` jusqu’à la fin du pipeline.

Le builder Gmsh actif utilise maintenant les contours polygonaux Shapely pour créer les volumes OCC par extrusion. Les `MultiPolygon` sont séparés en bâtiments individuels, la simplification est moins destructive, et l’enfoncement artificiel sous le terrain a été supprimé. Le domaine fluide est ensuite obtenu par une soustraction booléenne des volumes polygonaux, et non par des boîtes englobantes.

Les physical groups sont créés avant le maillage et conservés jusqu’à l’export. Les noms sont explicitement définis avec `setPhysicalName`, et la recréation tardive qui produisait des groupes anonymes a été supprimée. L’algorithme 3D HXT est utilisé en première intention, avec repli sur Delaunay en cas d’échec.

La propriété de viscosité cinématique reste explicitement affectée à `solver.constant.transportProperties.nu` dans les générateurs de cas, conformément à la structure attendue pour produire un fichier `transportProperties` complet.

## Vérifications effectuées

| Vérification | Résultat |
| --- | --- |
| Compilation syntaxique de tous les fichiers Python du dossier | Réussie : 35 fichiers analysés par `compileall`. |
| Import de `wind_profile` et compilation des scripts avec les chemins corrigés | Réussi. |
| Vérification des chemins de fichiers principaux | Réussie. |
| Vérification des espaces et erreurs de patch Git | Réussie avec `git diff --check`. |
| Test de fumée Gmsh sur deux empreintes non rectangulaires en L et en T | Réussi : 1 volume fluide, 1 589 nœuds, 22 004 éléments volumiques et huit physical groups nommés. |
| Test du builder Gmsh actif | Validé avec un harnais isolé reproduisant les interfaces `Building`, `UrbanModel` et `CFDTerrain`, afin d’éviter les imports optionnels non nécessaires au test géométrique. |

## Limites et recommandations

La génération complète d’un cas VoxCity/Gmsh/OpenFOAM réel n’a pas été lancée, car elle dépend des données HDF5 VoxCity et, selon le mode choisi, de l’authentification Google Earth Engine. En revanche, la géométrie et le maillage ont été reproduits sur des empreintes polygonales non rectangulaires dans Gmsh, avec un volume fluide unique et des patches nommés. Une validation finale sur les données VoxCity réelles reste nécessaire.

Il est recommandé d’ajouter ultérieurement un test unitaire indépendant pour la fusion d’empreintes, couvrant au minimum deux bâtiments éloignés, deux bâtiments séparés par moins que la distance de fusion et deux hauteurs distinctes. Il serait également utile d’ajouter un test de fumée qui vérifie que `transportProperties` contient bien `nu` après génération d’un cas minimal.

## Références

[1]: https://github.com/stevendaix/foampilot/tree/main/examples%2Fbuilding_geo "Dossier building_geo du dépôt foampilot"
[2]: https://github.com/stevendaix/foampilot "Dépôt GitHub foampilot"
