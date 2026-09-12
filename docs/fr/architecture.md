# Architecture et flux de travail

FoamPilot est une couche d’orchestration Python autour d’OpenFOAM. Il ne remplace pas les solveurs, les utilitaires de maillage ni les formats de fichiers d’OpenFOAM. Il fournit plutôt des objets Python qui créent, inspectent, exécutent et post-traitent un cas OpenFOAM.

> Un cas FoamPilot doit être considéré comme un artefact de construction reproductible : le script Python est la source de vérité, tandis que `0/`, `constant/` et `system/` sont des entrées générées et des sorties de simulation.

## Flux de travail de bout en bout

Un flux de travail typique comporte six étapes :

| Étape | Responsabilité de FoamPilot | Principales sorties |
| --- | --- | --- |
| Définir | Créer un `Solver`, des propriétés physiques et des objets de conditions aux limites. | Configuration Python |
| Mailler | Générer ou importer un maillage à l’aide de `blockMesh`, Gmsh, snappyHexMesh, ou d’un maillage OpenFOAM direct. | `constant/polyMesh` et dictionnaires de maillage |
| Configurer | Écrire `controlDict`, les schémas de discrétisation, les solveurs linéaires, les propriétés de transport, la turbulence, la gravité et, le cas échéant, des function objects. | `system/` et `constant/` |
| Exécuter | Lancer un solveur OpenFOAM sériel ou parallèle et conserver le journal dans le répertoire du cas. | Répertoires temporels et fichiers journal |
| Inspecter | Lire directement les résultats OpenFOAM natifs ou les convertir en VTK pour PyVista. | Maillages PyVista et champs dérivés |
| Rapporter | Générer des graphiques, des tableaux de bord, des résumés CSV, des PDF LaTeX ou des documents Typst. | Figures, tableaux et rapports |

Les étapes sont délibérément explicites. Un script peut s’arrêter après la génération du maillage, modifier un dictionnaire généré, ou relancer uniquement le post-traitement sans reconstruire le cas.

## Carte des packages

Le package public est organisé par responsabilité plutôt que par exécutable OpenFOAM :

| Package | Objectif |
| --- | --- |
| `foampilot.base` | Chemins du cas, abstractions de fichiers et orchestration du maillage. |
| `foampilot.solver` | Sélection du solveur, préparation du cas, exécution, décomposition et reconstruction. |
| `foampilot.boundaries` | Affectation des patches, conditions aux limites standard, dictionnaires bruts et conditions pilotées par CSV. |
| `foampilot.constant` | Dictionnaires de fluide, turbulence, gravité, phase, rayonnement et matériaux. |
| `foampilot.system` | `controlDict`, `fvSchemes`, `fvSolution`, function objects, contraintes, modèles et décomposition. |
| `foampilot.cht` | Transfert de chaleur conjugué (CHT) multi-région avec régions fluide/solide et conditions d’interface. |
| `foampilot.mesh` and `foampilot.openfoam` | Génération de maillage, export direct de maillage, assistants Gmsh et snappyHexMesh. |
| `foampilot.postprocess` | Post-traitement PyVista, lecteurs OpenFOAM natifs, analyse du vent et présentations web. |
| `foampilot.report` | Rapports de maillage, rapports de convergence, études de parallélisation, rendu LaTeX et Typst. |
| `foampilot.urban` | Modèles de données CFD urbaines expérimentaux, simplification, géométrie, maillage, patches, validation et lecteurs OSM. |
| `foampilot.utilities` | Unités, propriétés de fluide, résidus, fichiers météo, conversion de géométrie et utilitaires de couplage. |

## Fichiers générés et validation

FoamPilot écrit les dictionnaires OpenFOAM via des objets fichiers Python. Après chaque étape de génération, inspectez les fichiers produits plutôt que de vous fier uniquement aux attributs en mémoire. En particulier, vérifiez que `system/controlDict`, tous les champs initiaux dans `0/`, le maillage sous `constant/polyMesh`, et les dictionnaires de matériaux pertinents ont été écrits.

Pour les cas incompressibles, `constant/transportProperties` doit contenir les valeurs utilisées par le solveur, y compris la viscosité cinématique `nu`. Si une valeur est affectée dynamiquement mais n’apparaît pas dans le dictionnaire généré, considérez le cas comme invalide et vérifiez la routine d’écriture correspondante du répertoire constant avant de lancer OpenFOAM.

Une séquence de validation minimale est :

```bash
checkMesh -case path/to/case
foamDictionary path/to/case/constant/transportProperties -entry nu
foamDictionary path/to/case/system/controlDict -entry application
```

Les commandes exactes de validation dépendent de la distribution OpenFOAM. FoamPilot peut générer des fichiers, mais OpenFOAM reste l’autorité pour la syntaxe des dictionnaires, la validité du maillage et la compatibilité des solveurs.

## Dépendances optionnelles

L’installation de base et les extras optionnels sont séparés dans `pyproject.toml`. L’extra `docs` installe MkDocs, l’extra `dev` installe les outils de test et de linting, `gnn` contient les dépendances d’apprentissage sur graphes, et `urban` contient des lecteurs géospatiaux tels que OSMnx, GeoPandas, Rasterio, ainsi que la prise en charge LAS/LAZ.

```bash
pip install -e ".[dev,docs]"
# Optional urban workflows
pip install -e ".[urban]"
```

Certaines utilitaires spécialisés nécessitent également des applications systèmes ou des jeux de données externes. Consultez l’exemple pertinent avant d’exécuter un flux de travail dans un environnement vierge.
