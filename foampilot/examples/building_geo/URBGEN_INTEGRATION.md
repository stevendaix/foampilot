# Intégration UrbGEN dans foampilot

Cette branche ajoute `foampilot.urban.generation`, une implémentation Python compatible avec le contrat public du composant **UrbGEN generator**. L’objectif est de générer des quartiers aléatoires déterministes à partir d’une emprise de site et de contraintes de planification, puis de transmettre les bâtiments à `UrbanModel` pour la construction Gmsh ou build123d.

## Correspondance des paramètres

| UrbGEN / Grasshopper | foampilot | Rôle |
|---|---|---|
| `BCR`, `upperBCR`, `FAR` | `bcr`, `upper_bcr`, `far` | Cibles de couverture et de surface de plancher |
| `setback` | `setback` | Zone constructible par offset intérieur |
| `minWidth` | `min_width` | Largeur minimale des bâtiments |
| `towerSizeMode` | `tower_size_mode` | Compact, medium, maximized ou random |
| `minFootprintPerTower`, `maxFootprintPerTower` | mêmes noms en snake case | Bornes de croissance |
| `minTowerDistance` | `min_tower_distance` | Distance libre minimale |
| `towerBCRPriority` | `tower_bcr_priority` | Part de BCR portée par les tours avant podium |
| `towerGrowStep`, `towerGrowIterations` | mêmes noms en snake case | Croissance itérative vers la cible |
| `seed` | `seed` | Reproductibilité des quartiers |
| `towerTypologyMode` | `tower_typology_mode` | I, L, T, H, C/U, Plus, random, courtyard |
| `armLengthRatio` | `arm_length_ratio` | Dimension des bras des typologies composées |
| paramètres `podium*` | paramètres `podium_*` | Expansion et hauteur du podium |
| paramètres `globalRotation*` | paramètres `global_rotation_*` | Orientation uniforme ou par bâtiment |
| paramètres `courtyard*` | paramètres `courtyard_*` | Découpage et ruptures des anneaux |
| paramètres de hauteur | paramètres `height_*` | Variation et bornes réglementaires |
| paramètres de positionnement | paramètres `move_*`, `align_*` | Déplacement vers limite et alignement |

## Utilisation

```python
from shapely.geometry import Polygon
from foampilot.urban.generation import UrbGENConfig, generate_urbgen

site = Polygon([(0, 0), (240, 0), (240, 160), (0, 160)])
config = UrbGENConfig(
    bcr=0.35,
    far=2.5,
    setback=8.0,
    tower_typology_mode=6,
    tower_size_mode=3,
    seed=42,
    podium_floors=2,
    global_rotation_mode=0,
)
result = generate_urbgen(site, config)
urban_model = result.model
```

Le résultat conserve les empreintes et les métadonnées de chaque masse dans `UrbanModel`. Les attributs `typology`, `typology_name`, `angle_deg`, `floors` et `podium_offset` sont disponibles pour la classification des surfaces et le post-traitement aérodynamique. Le même résultat peut donc alimenter le backend Gmsh vectoriel ou un futur exporteur build123d/BREP.

## Fidélité et limites explicites

La documentation publique d’UrbGEN expose le contrat, les paramètres et le workflow, tandis que le dépôt original distribue la logique Grasshopper sous forme de composant compilé `.gha`; les fichiers C# versionnés sont principalement des wrappers RhinoCode. Cette PR reproduit donc le **contrat fonctionnel et les règles documentées** dans une API Python native, sans dépendance à Rhino ou Grasshopper. Une parité bit-à-bit avec le composant compilé n’est pas revendiquée.

La prochaine étape est d’implémenter le mode `Courtyard` comme générateur de polygones d’anneau dédié et de compléter les déplacements `moveToBoundary`, `moveAllToSetback`, `alignTowersToEdge` et `moveTowerToPodiumEdge`. Ces règles sont déjà réservées dans l’API afin de préserver la correspondance des paramètres.
