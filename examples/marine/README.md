# Études marines OpenFOAM reproductibles avec FoamPilot

Ce répertoire ajoute une orchestration Python explicite pour trois familles de cas de référence : la manœuvre libre d’un navire, le propulseur en référentiel tournant MRF et la carène mobile avec maillage overset. Les données de géométrie et les dictionnaires originaux **ne sont pas copiés** ici ; les dépôts de référence doivent être obtenus séparément, conformément à leurs conditions respectives.

| Étude | Référence | Approche numérique | Exécutable principal | Version déclarée dans les entrées |
| --- | --- | --- | --- | --- |
| Manœuvre libre | [`balabibo/maneuveringLib`][1] | Coques/rudder overset, corps rigide, contrôleurs de propulsion et de gouvernail | `overInterDyMFoam` | v2206 dans le tutoriel ; bibliothèque annonçant v2506 |
| Propulseur | [`skfelix/propeller-OpenFOAM`][2] | Maillage cfMesh, interfaces AMI et référentiel tournant MRF | `rhoSimpleFoam` | v2012 |
| DTC mobile | [`myozinaung/DTCMoving_Overset`][3] | Maillage overset, corps rigide à deux degrés de liberté | `overInterDyMFoam` | v2206 |

> **Compatibilité.** Ces études ciblent des distributions OpenCFD historiques. Avant toute exécution, sourcez la version OpenFOAM associée au dépôt choisi et vérifiez la présence des exécutables demandés avec `command -v overInterDyMFoam`, `rhoSimpleFoam`, `mergeMeshes` et `snappyHexMesh`.

## Apport à FoamPilot

Les nouveaux composants se trouvent dans `foampilot.workflows`.

| Composant | Rôle |
| --- | --- |
| `OpenFOAMWorkflow` | Décrit et exécute de manière ordonnée des commandes, copies et nettoyages, sans dépendre de `RunFunctions`. |
| `ConstantDirectory.add_dict_file()` | Génère des dictionnaires supplémentaires dans `constant`, par exemple `dynamicMeshDict` et `MRFProperties`. |
| `SystemDirectory.add_dict_file()` | Génère les dictionnaires supplémentaires de `system`, par exemple `fvOptions` ou `topoSetDict`. |
| `write_mrf_properties()` | Écrit une définition de zone de référence tournante pour le propulseur. |
| `write_overset_dynamic_mesh()` | Écrit le `dynamicMeshDict` overset de compatibilité pour les distributions OpenCFD historiques. |
| `write_openfoam13_rigid_body_mover()` | Écrit le bloc `mover` de corps rigide attendu par OpenFOAM Foundation 13. |
| `maneuvering_turning_workflow()`, `propeller_mrf_workflow()`, `dtc_overset_workflow()` | Conservent les séquences des trois références historiques. |
| `dtc_openfoam13_workflow()` | Exécute la voie Foundation 13 de coque DTC mobile, fondée sur `incompressibleVoF` et `foamRun`. |

## Installation de l’environnement

Installez FoamPilot depuis le dépôt modifié, puis sourcez la distribution OpenFOAM voulue. La bibliothèque `maneuveringLib` doit en outre être compilée et chargée avant l’étude de manœuvre :

```bash
cd /chemin/vers/maneuveringLib
./Allwmake
# Sourcez ensuite votre environnement OpenFOAM et, si nécessaire,
# ajoutez le répertoire de bibliothèques utilisateur à FOAM_USER_LIBBIN.
```

L’étude de propulseur nécessite que le maillage soit créé avant la simulation. Le dépôt de référence indique de construire le maillage à partir de `mesh/`, puis de copier `constant/polyMesh` dans `simulationTemplate/`. Le workflow FoamPilot couvre la partie simulation et post-traitement après cette préparation.[2]

## Prévisualiser avant d’exécuter

La prévisualisation ne modifie aucun fichier et ne nécessite pas OpenFOAM. Elle doit être utilisée d’abord pour vérifier les chemins, les prérequis et l’ordre des étapes.

```bash
cd examples/marine
python reproduce_reference_cases.py maneuvering /chemin/vers/maneuveringLib/tutorial/Turning35 --processors 8
python reproduce_reference_cases.py propeller /chemin/vers/propeller-OpenFOAM/simulationTemplate --processors 4
python reproduce_reference_cases.py dtc /chemin/vers/DTCMoving_Overset --processors 4
. /opt/openfoam13/etc/bashrc
python reproduce_reference_cases.py dtc13 "$FOAM_TUTORIALS/incompressibleVoF/DTCHullMoving" --mesh-source ../DTCHull --processors 1
```

## Exécuter un cas

Ajoutez `--execute` uniquement après avoir vérifié la prévisualisation et chargé la version OpenFOAM compatible. Chaque commande écrit un journal individuel dans `<racine-du-cas>/logs/`.

```bash
python reproduce_reference_cases.py dtc /chemin/vers/DTCMoving_Overset --processors 4 --execute

# Voie native OpenFOAM Foundation 13, après préparation du maillage DTCHull
python reproduce_reference_cases.py dtc13 "$FOAM_TUTORIALS/incompressibleVoF/DTCHullMoving" --mesh-source ../DTCHull --processors 1 --execute
```

Le workflow de manœuvre traite d’abord les sous-cas `hull` et `rudder`, prépare le maillage de fond, puis enchaîne une phase d’auto-propulsion et une phase de virage. La transition est explicite : les dictionnaires de contrôle et de mouvement sont remplacés entre les deux phases, et l’état `maneuvers` est transféré dans le répertoire du processeur.

## Validation avec OpenFOAM 13

La validation a été exécutée sous **OpenFOAM 13** installé depuis le paquet officiel `openfoam13`. Le test d’intégration `test_openfoam13_integration.py` utilise le tutoriel natif `pitzDailySteady` pour exécuter réellement `blockMesh` puis `foamRun` via `OpenFOAMWorkflow`. Il fait ensuite analyser par `foamDictionary` les fichiers `MRFProperties` et `dynamicMeshDict` produits par les générateurs marins.

```bash
. /opt/openfoam13/etc/bashrc
cd foampilot
PYTHONPATH=src pytest -q test/test_openfoam_workflows.py
cd ..
PYTHONPATH=foampilot/src python examples/marine/test_openfoam13_integration.py
```

La validation confirme la génération, la copie, le nettoyage, les journaux par étape et l’exécution d’un calcul OpenFOAM 13 réel. La version 13 fournit `foamRun`, `rhoSimpleFoam`, `blockMesh`, `mergeMeshes`, `snappyHexMesh`, `topoSet`, `refineMesh`, `setFields`, `decomposePar` et `reconstructPar`.[4]

| Workflow | État sous OpenFOAM 13 | Interprétation |
| --- | --- | --- |
| Propulseur MRF | Dictionnaire et orchestration validés ; `rhoSimpleFoam` est disponible | Le maillage cfMesh/AMI et les données v2012 restent à fournir pour le calcul de référence complet. |
| DTC historique overset | `dynamicMeshDict` legacy analysé par `foamDictionary` | `overInterDyMFoam` n’est pas fourni par OpenFOAM Foundation 13 ; cette variante historique reste liée à OpenCFD. |
| DTC native v13 | Maillage DTC officiel, restauration de `0/*.orig`, `setFields` et `foamRun` exécutés avec succès | Cette voie est une migration vers le tutoriel Foundation `DTCHullMoving`, sans maillage overset. |
| Manœuvre libre | Séquence déclarative et bascule de dictionnaires validées | La bibliothèque de manœuvre et `overInterDyMFoam` ciblent des distributions OpenCFD historiques ; une migration spécifique est nécessaire. |

> **Résultat important.** La correction de la liste `joints` dans `dynamicMeshDict` a été déclenchée par ce test : OpenFOAM 13 rejetait la forme initiale, et le dictionnaire généré passe désormais son propre parseur. La migration DTC Foundation 13 utilise désormais un bloc `mover` et a réalisé un calcul mobile réel via `foamRun`.

## Paramétrer de nouveaux cas

L’exemple suivant génère les deux dictionnaires déterminants sans éditer manuellement les entrées OpenFOAM :

```python
from foampilot.workflows.marine import write_mrf_properties, write_overset_dynamic_mesh

write_mrf_properties(
    "cases/propeller",
    cell_zone="rotor",
    axis=(0.0, 1.0, 0.0),
    omega=314.16,
)

write_overset_dynamic_mesh(
    "cases/dtc/background",
    cell_set="c1",
    body_name="hull",
    patch_name="hull",
    joints=("Pz", "Ry"),
)
```

> **Validation disponible dans cette proposition.** La génération des dictionnaires, la validation de séquence et l’exécution de commandes contrôlées sont testées automatiquement. Une validation CFD numérique complète reste conditionnée à la présence locale d’OpenFOAM/cfMesh dans la version attendue ainsi qu’aux maillages et bibliothèques externes des dépôts de référence.

## Références

[1]: https://github.com/balabibo/maneuveringLib "maneuveringLib — GitHub"
[2]: https://github.com/skfelix/propeller-OpenFOAM "propeller-OpenFOAM — GitHub"
[3]: https://github.com/myozinaung/DTCMoving_Overset "DTCMoving_Overset — GitHub"
[4]: https://openfoam.org/version/13/ "OpenFOAM 13 — Foundation"
