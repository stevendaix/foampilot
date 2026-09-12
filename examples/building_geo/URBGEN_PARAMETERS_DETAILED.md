# UrbGEN dans foampilot — documentation détaillée des paramètres

## 1. Objet et périmètre

Cette documentation décrit séparément les deux étapes du workflow UrbGEN : **la population de la région** (`UrbGEN PopulateRegion`) et **la génération des masses bâties** (`UrbGEN generator`). Cette séparation est essentielle pour expliquer la différence entre une simple génération de bâtiments sur une emprise rectangulaire et le résultat visible dans les exemples Grasshopper : la première étape fournit les points candidats et la seconde transforme ces points en empreintes, podiums et volumes.

Le port Python expose désormais les deux étapes dans `foampilot.urban.generation`. La fonction de population est `populate_region(region, PopulateRegionConfig(...), holes=...)`, et la fonction principale de génération est `generate_urbgen(site, config, centroids=None)`. Le paramètre `centroids` permet de transmettre explicitement les points produits par `populate_region` ou par une autre source. Lorsque `centroids` est omis, le générateur conserve son repli interne de type lattice pour les typologies classiques ; le mode Courtyard dérive sa disposition de l’emprise elle-même.

> **Important.** La documentation originale permet de reproduire le contrat fonctionnel et les règles d’utilisation, mais elle ne suffit pas à garantir une parité bit-à-bit avec Rhino/Grasshopper. Une parité numérique complète nécessite une sortie de référence exportée depuis le composant original, avec les mêmes courbes, points, unités, graine et paramètres.

## 2. Pipeline UrbGEN complet

Le workflow de référence doit être compris comme la chaîne suivante :

```text
Courbe de site + trous
        │
        ▼
UrbGEN PopulateRegion
  points candidats / branches
        │
        ▼
UrbGEN generator
  setback → filtrage → typologie → placement
  → croissance BCR → podium → hauteurs FAR
        │
        ▼
TowerFootprints + PodiumFootprints + masses 3D
        │
        ▼
UrbanModel foampilot → Gmsh → OpenFOAM
```

La population ne construit aucun bâtiment. Elle contrôle principalement **le nombre, la position, l’ordre et l’organisation spatiale des points candidats**. Le générateur peut ensuite rejeter certains points si une empreinte ne tient pas dans la zone constructible, si elle est trop proche d’une autre ou si sa surface sort des limites demandées. Il est donc normal que `Count` soit une cible et non nécessairement le nombre final de bâtiments dans les modes qui appliquent des contraintes géométriques strictes.

## 3. `UrbGEN PopulateRegion`

### 3.1 Entrées

| Paramètre | Type original | Défaut documenté | Unité | Effet précis |
|---|---|---:|---|---|
| `Crv` | `Curve` item | vide | modèle | Courbe fermée et plane définissant la région à peupler. Une courbe ouverte ou non plane produit une sortie vide. |
| `Count` | `Integer` item | vide | points | Nombre cible de points. Dans les modes grille, l’espacement est résolu automatiquement ; le nombre obtenu est annoncé comme approximatif, avec une tolérance documentée d’environ ±2 %. |
| `Mode` | `Number` item | `0` | code | `0` Random, `1` Regular grid, `2` Jittered grid, `3` Staggered grid triangulaire. |
| `Jitter` | `Number` item | vide | fraction ou valeur normalisée | Amplitude de perturbation appliquée à la grille pour le mode jittered ; la documentation indique une conversion en collection de flottants. La valeur exacte de normalisation doit être confirmée sur une sortie originale. |
| `Angle` | `Generic Data` item | vide | radians | Rotation de la grille autour de l’axe Z du plan de la courbe. Actif dans les modes 1 à 3. En Grasshopper, il faut convertir les degrés en radians avant connexion. |
| `Seed` | `Integer` item | `0` | entier | Graine déterministe. Le même seed et les mêmes entrées doivent produire le même résultat. Elle affecte explicitement les modes Random et Jittered. |
| `MinDist` | `Number` item | vide | unités modèle | Distance minimale entre les points, active dans le mode Random. La documentation recommande environ `0,55 × √(A/Count)` ; au-delà d’environ `0,70 × √(A/Count)`, le nombre de points peut devenir inférieur à la cible. |
| `Holes` | `Curve` list | vide | modèle | Courbes fermées exclues de la région : cours, blocs existants, servitudes ou emprises interdites. Leur surface est retirée lors de la résolution de l’espacement de grille. |

### 3.2 Sortie et ordre des points

La sortie `Pts` est une collection de points situés dans la frontière et hors des trous. Dans les modes `Regular`, `Jittered` et `Staggered`, les points sont ordonnés par rangée de grille. Dans le mode `Random`, ils sont ordonnés dans leur ordre de génération. Cet ordre est important car le générateur original utilise ensuite une logique de score et de mélange déterministe avant le placement.

### 3.3 Modes de population

| Mode | Code | Géométrie | Déterminisme | Densité spatiale | Cas d’emploi |
|---|---:|---|---|---|---|
| Random | `0` | Tirages dans la région, avec rejet des points invalides et possibilité de contrainte `MinDist` | Seed requis pour reproduire | Variable, souvent grumeleuse | Quartiers organiques et exploration stochastique |
| Regular grid | `1` | Grille cartésienne régulière tournée par `Angle` | Déterministe | Très uniforme | Étude paramétrique, comparaison contrôlée |
| Jittered grid | `2` | Grille régulière dont les points sont perturbés | Seed requis | Uniforme mais moins artificielle | Cas urbain réaliste sans perdre la maîtrise du nombre |
| Staggered grid | `3` | Grille triangulaire, rangées décalées | Déterministe sauf perturbation éventuelle | Bonne couverture isotrope | Implantations serrées avec distances homogènes |

### 3.4 Résolution de l’espacement

Pour les modes de grille, l’algorithme doit tenir compte de l’aire utile `A`, c’est-à-dire l’aire de la région après soustraction des trous. Une estimation initiale de l’espacement est proportionnelle à `√(A/Count)`, puis l’espacement est ajusté afin d’obtenir une cardinalité proche de `Count`. La grille est ensuite découpée par la frontière et les trous ; les points hors région sont supprimés.

Pour le mode Random, `MinDist` introduit un rejet spatial. Une valeur trop grande ne diminue pas seulement la densité locale : elle peut empêcher d’atteindre `Count`. La relation recommandée par la documentation est donc un repère, pas une garantie mathématique pour des régions concaves ou trouées.

### 3.5 Interaction avec le générateur

Le composant générateur ne doit pas recevoir la frontière brute comme une liste de points non contrôlée. Il doit recevoir les points par branche correspondant aux sites ou parcelles. Pour chaque point, le générateur applique d’abord le setback, puis vérifie la présence du point dans la zone constructible. Les points hors setback sont éliminés. Si aucun point valide n’est fourni, le code original conserve un repli sur le centroïde du site ; ce repli explique pourquoi un résultat peut contenir un seul bâtiment même lorsque `Count` était supérieur.

## 4. Paramètres du générateur foampilot

Les valeurs ci-dessous correspondent à `UrbGENConfig` dans `foampilot/src/foampilot/urban/generation/urbgen.py`. Les noms Python utilisent `snake_case`, tandis que les noms Grasshopper sont indiqués lorsque la correspondance est connue.

### 4.1 Site et objectifs de planification

| Python | GHA | Défaut | Bornes / normalisation | Unité | Fonction |
|---|---|---:|---|---|---|
| `bcr` | `BCR` | `0,50` | `0 < BCR ≤ 1` | ratio | Couverture au sol cible : aire de l’union des empreintes divisée par l’aire du site. |
| `upper_bcr` | `upperBCR` | `max(1,05×BCR, 1,10×BCR)` | doit être ≥ `bcr` et ≤ 1 | ratio | Limite haute utilisée pendant la convergence avant éventuelle coupe stricte au BCR demandé. |
| `far` | `FAR` | `3,0` | strictement positif | ratio | Surface de plancher totale cible divisée par l’aire du site. Contrôle principalement le nombre de niveaux. |
| `setback` | `setback` | `5,0` | ≥ 0 dans le port | m | Offset intérieur de la frontière. Toutes les empreintes doivent être couvertes par la zone constructible. |
| `floor_height` | `floorHeight` | `3,5` | positive en pratique | m | Hauteur d’un niveau, utilisée pour convertir les niveaux en hauteur métrique. |
| `floor_height_override` | — | `None` | si définie, remplace `floor_height` pour le calcul final | m | Override foampilot destiné aux chaînes de calcul spécifiques. |

La cible GFA est `site.area × far`. La cible de couverture est `site.area × bcr`. Ces deux objectifs ne sont pas équivalents : le BCR contrôle l’emprise au sol tandis que le FAR contrôle les surfaces superposées par niveau. À emprise constante, augmenter le FAR augmente la hauteur, pas nécessairement le nombre de bâtiments.

### 4.2 Taille, surface et espacement des tours

| Python | GHA | Défaut | Bornes / règle | Unité | Effet |
|---|---|---:|---|---|---|
| `min_width` | `minWidth` | `12,0` | minimum `2,0` | m | Largeur courte de la barre principale ; largeur de bande du Courtyard. |
| `tower_size_mode` | `towerSizeMode` | `1` | `0` à `3` | code | `0` Compact, `1` Medium, `2` Maximized, `3` Random. Change le tirage de surface ; dans Courtyard, change le nombre et la longueur des segments. |
| `min_footprint_per_tower` | `minFootprintPerTower` | `80,0` | minimum original `20,0` | m² | Surface minimale d’une empreinte candidate. |
| `max_footprint_per_tower` | `maxFootprintPerTower` | `350,0` | au moins `min + 30` | m² | Surface maximale d’une empreinte, bras compris. |
| `max_length_width_ratio` | `maxLengthWidthRatio` | `4,0` | minimum `2,0` | — | Limite d’allongement de la barre principale. |
| `min_tower_distance` | `minTowerDistance` | `12,0` | ≥ 0 | m | Distance libre minimale entre empreintes, vérifiée au placement et pendant la croissance. |
| `tower_bcr_priority` | `towerBCRPriority` | `0,65` | `[0,1]` | ratio | Part de la cible BCR recherchée par les tours avant expansion du podium. |
| `tower_grow_step` | `towerGrowStep` | `1,0` | minimum `0,1` | m | Incrément de longueur de barre pendant la croissance. |
| `tower_grow_iterations` | `towerGrowIterations` | `80` | ≥ 0 | itérations | Nombre maximal de tentatives de croissance. |
| `seed` | `seed` | `0` | réduit modulo `10000` | entier | Contrôle l’ordre des points, les tirages de taille, les angles, la croissance, le podium et les hauteurs. |

Les modes de taille ne modifient pas directement `Count`. Ils changent la surface de chaque candidat. À BCR fixe, des empreintes plus grandes peuvent atteindre la couverture avec moins de bâtiments ; des empreintes compactes peuvent nécessiter davantage de points valides. C’est l’une des raisons pour lesquelles deux réalisations avec le même site et le même BCR peuvent avoir des cardinalités différentes.

### 4.3 Typologies

| Code | Nom | Construction | Interaction principale |
|---:|---|---|---|
| `0` | I | Rectangle simple, longueur × largeur | Forme la plus facile à placer et à densifier. |
| `1` | L | Barre principale + un module perpendiculaire | Surface et encombrement augmentent avec `arm_length_ratio`. |
| `2` | T | Barre principale + module transversal | Peut échouer dans une parcelle étroite malgré une barre admissible. |
| `3` | H | Deux extrémités reliées par une barre ou modules latéraux | Demande davantage de largeur libre. |
| `4` | C/U | Forme en U autour d’un vide | Nécessite une emprise suffisante pour conserver une cour intérieure. |
| `5` | Plus | Barre centrale + bras croisés | Sensible aux distances minimales et à la rotation. |
| `6` | Random | Tirage déterministe entre `0` et `5` pour chaque bâtiment | Le seed et l’index du bâtiment déterminent le résultat. |
| `7` | Courtyard | Anneau de blocs sur le périmètre de la zone | Ignore les centroïdes et dérive la population de la frontière. |

| Python | GHA | Défaut | Bornes | Unité | Effet |
|---|---|---:|---|---|---|
| `tower_typology_mode` | `towerTypologyMode` | `0` | `0` à `7` | code | Sélection de la famille de forme. |
| `arm_length_ratio` | `armLengthRatio` | `1,3` | minimum `0,3` | ratio | Longueur des bras des typologies composées en multiple de `min_width`. |

### 4.4 Podium

| Python | GHA | Défaut | Unité | Effet |
|---|---|---:|---|---|
| `podium_floors` | `podiumFloors` | `2` | niveaux | Nombre de niveaux du podium. Zéro désactive le podium. |
| `podium_min_offset` | `podiumMinOffset` | `2,0` | m | Borne basse de l’offset autour des tours. |
| `podium_max_offset` | `podiumMaxOffset` | `15,0` | m | Borne haute de la recherche ; bornée aussi par une fraction de `√(site.area)` dans le contrat original. |
| `move_tower_to_podium_edge` | `moveTowerToPodiumEdge` | `False` | booléen | Déplace une tour vers le bord de son propre podium sans changer la forme du podium. |

Le podium est calculé après la phase de tours. Le port recherche un offset dont l’aire ajoutée permet de rapprocher la couverture totale de la cible. Une conséquence importante est que `tower_bcr_priority` répartit la BCR entre deux mécanismes : une valeur élevée favorise des tours plus couvrantes et un podium plus mince ; une valeur faible favorise un podium plus large.

### 4.5 Rotation et orientation

| Python | GHA | Défaut | Valeurs | Effet |
|---|---|---:|---|---|
| `global_rotation_mode` | `globalRotationMode` | `0` | `0` à `3` | Mode `0` : angles candidats par bâtiment ; `1` : angle fixe ; `2`/`3` : angle commun choisi dans un ensemble discret. |
| `uniform_rotation_deg` | `uniformRotationDeg` | `0,0` | `0` à `180` degrés | Angle fixe lorsque `global_rotation_mode = 1`. |
| `align_towers_to_edge` | `alignTowersToEdge` | `False` | booléen | Aligne la longueur de la tour avec l’arête de site la plus proche. |
| `edge_align_both_orientations` | `edgeAlignBothOrientations` | `True` | booléen | Autorise l’essai de l’orientation perpendiculaire si l’orientation parallèle échoue. |

Dans le mode `0`, le port essaie des angles discrets et des angles pseudo-aléatoires déterministes. L’orientation n’est pas un simple attribut décoratif : elle change le bounding box, les collisions, la couverture et donc la cardinalité finale.

### 4.6 Courtyard

| Python | GHA | Défaut | Bornes / normalisation | Unité | Effet |
|---|---|---:|---|---|---|
| `courtyard_count` | `courtyardCount` | `1` | minimum `1` | zones | Nombre de zones séparées, chacune avec son propre anneau. |
| `courtyard_break_count` | `courtyardBreakCount` | `4` | minimum `1` | ruptures | Nombre de ruptures de l’anneau avant ajustement par `tower_size_mode`. |
| `courtyard_break_width` | `courtyardBreakWidth` | `18,0` | minimum pratique `1,0` | m | Largeur des ouvertures entre blocs de l’anneau. |
| `courtyard_zone_gap` | `courtyardZoneGap` | `12,0` | ≥ 0 | m | Distance entre zones Courtyard adjacentes. |
| `courtyard_split_angle` | `courtyardSplitAngle` | `0,0` | `−45` à `+45` degrés | degrés | Rotation de l’axe de séparation des zones. |
| `courtyard_break_shift` | `courtyardBreakShift` | `0,0` | libre, modulo périmètre | m | Décalage des ouvertures le long de l’anneau. |
| `courtyard_layout_mode` | `courtyardLayoutMode` | `0` | `0` Corner, `1` Cluster | code | Corner : blocs ancrés aux coins ; Cluster : blocs groupés en séquence autour de l’anneau. |

Le port utilise `RingCache` pour convertir une position curviligne en point sur les anneaux extérieur et intérieur. La surface d’un segment est construite entre deux positions d’arc, puis rejetée si elle est invalide ou trop petite. La surface minimale Courtyard est volontairement distincte de `min_footprint_per_tower` : le code utilise une fraction (`0,15 × min_footprint_per_tower`) afin de ne pas supprimer les segments de lamelle admissibles uniquement parce qu’ils sont plus petits qu’une tour classique.

`tower_size_mode` a un sens particulier en Courtyard : Compact ajoute des ruptures, donc davantage de blocs plus courts ; Maximized retire une rupture lorsque cela est possible, donc produit moins de blocs plus longs ; Random perturbe déterministement le nombre de ruptures par zone.

### 4.7 Hauteur et réglementation

| Python | GHA | Défaut | Unité | Effet |
|---|---|---:|---|---|
| `height_variation` | `heightVariation` | `0,0` | ratio | Amplitude de variation autour du nombre de niveaux de base. Zéro donne une hauteur uniforme avant réglementation. |
| `enforce_height_regulation` | `enforceHeightRegulation` | `False` | booléen | Active l’application des bornes et du mode de réglementation. |
| `height_regulation_mode` | `heightRegulationMode` | `0` | code `0` à `2` | Stratégie de traitement des hauteurs hors limites. La sémantique exacte des trois modes doit être comparée à la sortie GHA pour une parité numérique. |
| `max_building_height` | `maxBuildingHeight` | `100,0` | m | Hauteur maximale réglementaire. |
| `min_building_height` | `minBuildingHeight` | `3,0` | m | Hauteur minimale réglementaire. |

Le calcul de base affecte un nombre de niveaux de manière à approcher `site.area × FAR`, après prise en compte de la GFA du podium. Les hauteurs sont ensuite converties par `floor_height`, éventuellement régulées, puis stockées dans les métadonnées des bâtiments.

### 4.8 Positionnement post-placement

| Python | GHA | Défaut | Effet |
|---|---|---:|---|
| `move_to_boundary` | `moveToBoundary` | `False` | Déplace radialement chaque tour vers la frontière de setback ; ouvre le centre et est ignoré en Courtyard. |
| `move_all_to_setback` | `moveAllToSetback` | `False` | Déplace chaque bâtiment suivant la plus courte direction vers la frontière de setback ; crée une couronne bâtie. |
| `align_towers_to_edge` | `alignTowersToEdge` | `False` | Oriente chaque bâtiment selon l’arête la plus proche, avec revalidation des collisions. |
| `edge_align_both_orientations` | `edgeAlignBothOrientations` | `True` | Autorise la seconde orientation si la première est impossible. |
| `move_tower_to_podium_edge` | `moveTowerToPodiumEdge` | `False` | Place la tour contre le bord de son podium le plus proche de la rue. |

Ces opérations sont contraintes : le déplacement est accepté seulement si l’empreinte reste dans le site constructible et conserve `min_tower_distance`. Leur activation peut donc réduire la cardinalité effective si aucune position admissible n’existe, même si les points de départ étaient valides.

## 5. Valeurs par défaut complètes de `UrbGENConfig`

| Champ | Valeur |
|---|---:|
| `bcr` | `0,50` |
| `upper_bcr` | automatique, `max(1,05×bcr, 1,10×bcr)` |
| `far` | `3,0` |
| `setback` | `5,0` |
| `min_width` | `12,0` |
| `tower_size_mode` | `1` |
| `min_footprint_per_tower` | `80,0` |
| `max_footprint_per_tower` | `350,0` |
| `max_length_width_ratio` | `4,0` |
| `min_tower_distance` | `12,0` |
| `tower_bcr_priority` | `0,65` |
| `tower_grow_step` | `1,0` |
| `tower_grow_iterations` | `80` |
| `seed` | `0` |
| `tower_typology_mode` | `0` |
| `arm_length_ratio` | `1,3` |
| `podium_floors` | `2` |
| `podium_min_offset` | `2,0` |
| `podium_max_offset` | `15,0` |
| `floor_height` | `3,5` |
| `global_rotation_mode` | `0` |
| `uniform_rotation_deg` | `0,0` |
| `courtyard_count` | `1` |
| `courtyard_break_count` | `4` |
| `courtyard_break_width` | `18,0` |
| `courtyard_zone_gap` | `12,0` |
| `courtyard_split_angle` | `0,0` |
| `courtyard_break_shift` | `0,0` |
| `courtyard_layout_mode` | `0` |
| `height_variation` | `0,0` |
| `enforce_height_regulation` | `False` |
| `height_regulation_mode` | `0` |
| `max_building_height` | `100,0` |
| `min_building_height` | `3,0` |
| `move_to_boundary` | `False` |
| `move_all_to_setback` | `False` |
| `align_towers_to_edge` | `False` |
| `edge_align_both_orientations` | `True` |
| `move_tower_to_podium_edge` | `False` |
| `floor_height_override` | `None` |

## 6. Pourquoi le nombre de bâtiments diffère des images originales

Le nombre final n’est pas égal automatiquement à `Count`. Il dépend du produit de cinq filtres : la population de points, le setback, la taille et la typologie, l’espacement entre empreintes et la capacité de convergence BCR. Une grande valeur de `min_tower_distance`, une typologie en U ou en Plus, une parcelle concave, un setback important ou un `min_width` élevé peuvent supprimer des candidats.

Les références UrbGEN montrent en plus un **district multi-îlots**. Si chaque îlot est fourni séparément au générateur, chaque parcelle possède son propre BCR, son propre FAR et ses propres points candidats. Un test sur un seul rectangle ne peut donc pas reproduire la répartition visuelle des exemples. Pour un cas réaliste, il faut : conserver la géométrie des îlots, appeler `PopulateRegion` par îlot, conserver les trous et les espaces publics, puis générer les bâtiments avec une graine contrôlée par îlot.

## 7. Correspondance avec l’API foampilot

| Élément original | API foampilot | Statut |
|---|---|---|
| Site fermé planaire | `generate_urbgen(site, config)` | Intégré |
| Points imposés | `centroids=Iterable[Point]` | Intégré |
| Population automatique | `populate_region(...)` puis `centroids=...` | Intégré ; le lattice interne reste un repli du générateur |
| Modes Random/Grid/Jittered/Staggered | `PopulateRegionConfig.mode` `0` à `3` | Intégrés et testés ; la calibration exacte contre Rhino reste à comparer |
| Typologies I/L/T/H/C/Plus | `tower_typology_mode` | Intégré et testé |
| Courtyard par anneaux | `CourtyardContext`, `RingCache` | Intégré et testé |
| BCR/FAR/podium | solveurs internes | Intégré et testé |
| Sorties 3D | `UrbanModel` et `Building` | Intégré |
| Gmsh/OpenFOAM | exporteurs foampilot | Intégré et validé sur cas généré |
| Sorties détaillées originales | `diagnostics` et métadonnées bâtiment | Partiel ; les principaux indicateurs sont disponibles |

## 8. Recommandation d’utilisation pour un quartier réaliste

Pour approcher les exemples originaux, il est recommandé de ne pas utiliser une seule emprise rectangulaire. Il faut découper le district en îlots réels ou reconstruits, placer les rues et les cours comme trous, générer les points avec `Mode=2` ou `Mode=3`, puis transmettre les points par parcelle :

```python
from shapely.geometry import Polygon, Point
from foampilot.urban.generation import UrbGENConfig, generate_urbgen

parcel = Polygon([(0, 0), (80, 0), (80, 55), (0, 55)])
centroids = [
    Point(12, 12), Point(35, 12), Point(58, 12),
    Point(12, 38), Point(35, 38), Point(58, 38),
]

config = UrbGENConfig(
    bcr=0.35,
    upper_bcr=0.385,
    far=2.8,
    setback=4.0,
    min_width=8.0,
    min_footprint_per_tower=45.0,
    max_footprint_per_tower=180.0,
    min_tower_distance=5.0,
    tower_typology_mode=6,
    tower_size_mode=1,
    global_rotation_mode=2,
    height_variation=0.25,
    podium_floors=1,
    seed=42,
)
result = generate_urbgen(parcel, config, centroids=centroids)
```

Pour le mode Courtyard, il ne faut pas fournir les centroïdes en espérant contrôler l’anneau : le contrat original indique que les centroïdes sont ignorés lorsque `towerTypologyMode=7`. Il faut contrôler `courtyard_count`, `courtyard_break_count`, `courtyard_break_width`, `courtyard_layout_mode` et la géométrie du site.

## 9. Validation recommandée

Une validation complète doit comparer, pour un même site et les mêmes paramètres : la liste des points `PopulateRegion`, le nombre de points, l’ordre des points, les empreintes, les angles, les typologies, les longueurs, la GFA, la BCR, la FAR, le podium, les hauteurs et les métadonnées. Une simple comparaison d’image ne suffit pas, car la projection, le rendu et le contexte peuvent masquer des différences numériques.

Les tests foampilot ciblés sont exécutables avec :

```bash
PYTHONPATH=foampilot/src pytest -q \
  foampilot/src/foampilot/urban/tests/test_urbgen.py \
  foampilot/src/foampilot/urban/tests/test_urbgen_features.py
```

La documentation de référence utilisée pour les contrats d’entrée et de sortie est disponible dans les composants originaux : [PopulateRegion](https://github.com/trongtintr/UrbGEN/blob/main/docs/components/UrbGENPopulateRegion.md) et [generator](https://github.com/trongtintr/UrbGEN/blob/main/docs/components/UrbGENgenerator.md). Le dépôt original et ses images de démonstration sont disponibles sur [github.com/trongtintr/UrbGEN](https://github.com/trongtintr/UrbGEN).

## Références

[1]: https://github.com/trongtintr/UrbGEN/blob/main/docs/components/UrbGENPopulateRegion.md "Documentation UrbGEN PopulateRegion"

[2]: https://github.com/trongtintr/UrbGEN/blob/main/docs/components/UrbGENgenerator.md "Documentation UrbGEN generator"

[3]: https://github.com/trongtintr/UrbGEN "Dépôt officiel UrbGEN"
