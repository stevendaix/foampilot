# Exemples UrbGEN et chaîne aérodynamique

Ce dossier contient l’intégration UrbGEN dans `foampilot`, depuis la population des régions jusqu’à l’export d’un domaine Gmsh/OpenFOAM. Les scripts produisent leurs sorties dans des répertoires de validation ignorés par Git ; aucun maillage ou cas généré n’est requis pour installer le code.

## Installation

Depuis la racine du dépôt :

```bash
pip install -e '.[urban]'
```

L’option `urban` installe les dépendances de lecture et de traitement géospatial. Les fonctions UrbGEN utilisent Shapely ; Gmsh et Meshio sont nécessaires pour les validations de maillage et l’export CFD. OpenFOAM doit être installé séparément si un solveur doit être exécuté.

## API de population

```python
from shapely.geometry import Polygon
from foampilot.urban.generation import (
    PopulateRegionConfig,
    UrbGENConfig,
    generate_urbgen,
    populate_region,
)

site = Polygon([(0, 0), (120, 0), (120, 80), (0, 80)])
population = populate_region(
    site,
    PopulateRegionConfig(
        count=24,
        mode=2,       # 0 Random, 1 Regular, 2 Jittered, 3 Staggered
        jitter=0.20,
        angle=0.0,    # radians
        seed=42,
        min_dist=None,
    ),
)
result = generate_urbgen(
    site,
    UrbGENConfig(bcr=0.30, far=2.5, setback=4.0, seed=42),
    centroids=population.points,
)
```

`PopulateRegion` produit uniquement des points candidats. Le générateur peut encore rejeter un point si la typologie demandée ne tient pas dans le setback ou viole la distance minimale entre empreintes. Pour reproduire un quartier, il faut peupler chaque îlot séparément et conserver les points par branche ou identifiant de parcelle.

## Paramètres et parité

La table complète des entrées, valeurs par défaut, unités, bornes et interactions se trouve dans [`URBGEN_PARAMETERS_DETAILED.md`](URBGEN_PARAMETERS_DETAILED.md). La matrice de correspondance est dans [`URBGEN_PARITY_MATRIX.md`](URBGEN_PARITY_MATRIX.md). Les limites connues de la comparaison avec Rhino/Grasshopper sont dans [`URBGEN_REAL_CASE_COMPARISON.md`](URBGEN_REAL_CASE_COMPARISON.md).

## Reproduire les exemples

Les tests ciblés :

```bash
PYTHONPATH=foampilot/src pytest -q \
  foampilot/src/foampilot/urban/tests/test_urbgen.py \
  foampilot/src/foampilot/urban/tests/test_urbgen_features.py
```

La scène synthétique multi-îlots et sa vue 3D :

```bash
PYTHONPATH=foampilot/src python3 \
  foampilot/examples/building_geo/plot_urbgen_realistic_district.py
```

Le rapport de cardinalité :

```bash
PYTHONPATH=foampilot/src python3 \
  foampilot/examples/building_geo/urbgen_population_report.py
```

La validation du maillage et de l’export de cas :

```bash
PYTHONPATH=foampilot/src python3 \
  foampilot/examples/building_geo/validate_urbgen_cfd_chain.py
```

Les commandes `Allrun` et `Allclean` fournissent un raccourci reproductible pour lancer ou nettoyer les validations locales :

```bash
./foampilot/examples/building_geo/Allrun
./foampilot/examples/building_geo/Allclean
```

`validate_urbgen_cfd_chain.py` est actuellement un **smoke test géométrie/maillage/export**. Il vérifie Gmsh, la structure OpenFOAM et les patches, mais ne prétend pas exécuter un solveur ni démontrer une convergence physique. Une validation CFD complète nécessite un solveur OpenFOAM disponible, des conditions aux limites validées et des critères de résidus documentés.

## Artefacts

Les fichiers PNG, JSON de sortie, MSH, VTK, logs, caches Python et cas OpenFOAM générés sont exclus du versionnement. Ils doivent être produits localement par les scripts ci-dessus. Le manifeste [`configs/reference_case.json`](configs/reference_case.json) conserve les paramètres d’un cas de référence ; il ne remplace pas encore un export Rhino de référence.

## Organisation

| Élément | Rôle |
|---|---|
| `src/foampilot/urban/generation/population.py` | API `PopulateRegion` |
| `src/foampilot/urban/generation/urbgen.py` | Génération des typologies et solveurs BCR/FAR |
| `src/foampilot/mesh/direct_openfoam_exporter.py` | Export direct vers OpenFOAM |
| `src/foampilot/urban/tests/` | Tests unitaires UrbGEN et population |
| `configs/reference_case.json` | Manifeste reproductible du cas de référence |
| `Allrun` / `Allclean` | Lancement et nettoyage des validations |
| `URBGEN_PARAMETERS_DETAILED.md` | Documentation détaillée des paramètres |
