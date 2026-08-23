# Audit détaillé du maillage MakeHuman pour FoamPilot

## Conclusion principale

Le problème initial ne venait pas d’un maillage `body` MakeHuman intrinsèquement composé de 154 morceaux. Il venait du fait que `base.npz` contient plusieurs familles de géométrie et que l’exporteur envoyait **tous les groupes** vers OpenFOAM. Le groupe physique `body` est le groupe `0` et contient 13 378 quadrilatères, soit 26 756 triangles après triangulation. Les 5 108 quadrilatères restants appartiennent à des groupes `joint-*` et `helper-*` : joints d’animation, yeux, dents, cheveux, vêtements, génital et autres helpers.

L’ancien export global produisait 36 972 triangles, 154 composantes, 784 arêtes ouvertes et une surface géométrique brute de 4,563 m². Après filtrage `group=0`, le maillage physique contient 26 756 triangles et une seule composante fermée. Le convertisseur meshio a été corrigé pour sélectionner `group=0` par défaut ; l’option `--all-groups` reste disponible pour les diagnostics.

## 1. Structure réelle de `base.npz`

| Élément | Valeur |
|---|---:|
| Sommets | 19 158 |
| Faces quadrangulaires totales | 18 486 |
| Faces du groupe `body` | 13 378 |
| Groupes internes | 139 |
| Triangles après triangulation globale | 36 972 |
| Triangles `body` après triangulation | 26 756 |
| Indices de faces invalides | 0 |
| Quadrilatères avec sommets répétés | 0 |
| Triangles dégénérés dans l’export | 0 |
| Échelle appliquée | 0,1 |

Les métadonnées décodées montrent explicitement `body` comme groupe 0, suivi de `helper-tights`, de nombreux `joint-*`, puis des `helper-*` tels que cheveux, yeux, dents, langue et génital. Ces groupes ne doivent pas être introduits dans la surface de la peau thermorégulante.

## 2. Topologie avant conversion

L’audit des 36 972 triangles issus de tous les groupes donne 55 850 arêtes uniques, dont 784 arêtes incidentes à une seule face et aucune arête incidente à plus de deux faces. Il y a 154 composantes de faces. La composante principale contient 26 756 triangles, exactement la taille du groupe `body`. Les autres composantes sont donc principalement les géométries auxiliaires exportées par erreur.

En filtrant le groupe `body`, `surfaceCheck` OpenFOAM indique :

> Surface is closed. All edges connected to two faces.

La surface body-only a une seule composante, aucun triangle illégal et une aire de 1,61378754 m² avant le passage dans `snappyHexMesh`. La variante meshio et la variante trimesh donnent la même aire à moins de 1,3 × 10⁻⁹ m² près.

## 3. Comparaison des exports

| Propriété | Tous les groupes | Body-only |
|---|---:|---:|
| Quadrilatères source | 18 486 | 13 378 |
| Triangles exportés | 36 972 | 26 756 |
| Composantes | 154 | 1 |
| Arêtes ouvertes source | 784 | 0 |
| Arêtes non-manifold source | 0 | 0 |
| Surface brute | 4,56317 m² | 1,61379 m² |
| Volume signé | 0,19839 m³ | 0,05490 m³ |
| Surface après snappy | 3,20838 m² | 1,55497 m² |
| Faces du patch OpenFOAM | 20 223 | 9 418 |

La surface de 4,56 m² n’était donc pas une surface corporelle utilisable : elle incluait les éléments auxiliaires et plusieurs pièces déconnectées. La surface body-only de 1,61 m² est cohérente avec une surface corporelle de MakeHuman à l’échelle appliquée, sous réserve de la morphologie et de la définition exacte du modèle.

## 4. Qualité OpenFOAM avec le niveau de raffinement courant

Le cas body-only au niveau de surface `(2 2)` produit un maillage calculable mais encore imparfait :

| Indicateur | Tous groupes | Body-only |
|---|---:|---:|
| Cellules | 89 604 | 66 379 |
| Points | 114 861 | 84 589 |
| Faces | 291 913 | 215 756 |
| Aspect ratio maximal | 10,49 | 5,17 |
| Non-orthogonalité maximale | 64,26° | 43,01° |
| Non-orthogonalité moyenne | 11,05° | 10,74° |
| Skewness maximale | 3,62 | 4,85 |
| Faces concaves | 251 | 111 |
| Cellules concaves | 4 330 | 3 660 |
| Points non-manifold OpenFOAM | 12 | 4 |
| État `checkMesh` | Mesh OK | 2 contrôles échoués |

Le filtrage body-only améliore fortement l’aspect ratio, la non-orthogonalité, le nombre de faces concaves et les points non-manifold. Il laisse cependant deux faces fortement skew et quatre points non-manifold produits par le snapping OpenFOAM près des bras/mains. Les coordonnées des deux faces skew sont approximativement `(-0,394 ; 0,208 ; 0,283)` et `(0,394 ; 0,208 ; 0,283)`.

## 5. Variante body-only raffinée

Une variante avec surface `(3 3)`, `nCellsBetweenLevels 3`, `nSmoothPatch 5`, `tolerance 1,5`, `nSolveIter 80` et `nRelaxIter 10` produit :

| Indicateur | Variante raffinée |
|---|---:|
| Cellules | 259 196 |
| Points | 321 497 |
| Faces | 833 474 |
| Faces du patch `human` | 36 728 |
| Topologie du patch | closed singly connected |
| Aspect ratio maximal | 6,66 |
| Non-orthogonalité maximale | 56,36° |
| Non-orthogonalité moyenne | 10,79° |
| Skewness maximale | 3,64 |
| État `checkMesh` | Mesh OK |

Cette variante supprime le message `multiply connected (shared edge)` sur le patch `human` et passe `checkMesh`. Elle est néanmoins beaucoup plus coûteuse, avec environ 3,9 fois plus de cellules que body-only niveau 2.

## 6. Impact sur le calcul et le couplage

Le cas body-only niveau 2 échange correctement 9 418 faces avec JOS-3. Le pilote termine correctement et le modèle physiologique produit des températures cohérentes, mais la continuité CFD dérive fortement à 0,20 s. Avec une température de peau fixe, la même dérive apparaît ; elle ne peut donc pas être attribuée au calcul JOS-3 seul.

La variante raffinée passe `checkMesh`, mais le premier couplage montre une divergence de la pression vers 0,15 s. Son HTC maximal est plus élevé, jusqu’à environ 146 W m⁻² K⁻¹, contre environ 24,5 W m⁻² K⁻¹ pour la variante body-only niveau 2. Le raffinement améliore la topologie et la qualité géométrique mais modifie fortement les gradients thermiques et rend la configuration actuelle plus raide numériquement.

La conclusion est donc double :

1. **La sélection de la géométrie était bien la cause racine de la topologie initiale incohérente.** Le pipeline doit impérativement exporter `body` uniquement.
2. **La divergence résiduelle n’est plus principalement un problème de trous dans le STL.** Elle provient désormais de la résolution CFD, des gradients locaux autour des bras/mains, de la formulation thermophysique et des conditions ouvertes ou de pression.

## Corrections appliquées

Les deux exporteurs ont été corrigés : `convert_makehuman_meshio.py` et `export_makehuman_npz_clean.py` sélectionnent maintenant `group=0` par défaut et proposent `--all-groups` pour reproduire l’ancien export global. Les rapports d’audit et les cas comparatifs sont conservés sous `openfoam_runs/human_mesh_variants/`.

## Prochaines corrections recommandées

La prochaine étape ne doit pas être de remplir artificiellement les trous de l’ancien STL. Il faut repartir du groupe `body` propre, puis stabiliser progressivement le cas OpenFOAM. L’ordre recommandé est de conserver la variante body-only niveau 2 comme test de non-régression, de réduire les gradients de départ par une initialisation thermique plus progressive, d’améliorer les contrôles `p_rgh` et `U` aux frontières ouvertes, puis de réintroduire le raffinement local autour des mains et des bras plutôt qu’un niveau `(3 3)` global.

Pour la visualisation et le mapping, les faces du patch OpenFOAM doivent rester la référence d’échange. L’aire de la surface STL brute ne doit pas être utilisée directement comme facteur de correction global sans comparer les aires agrégées par zone JOS-3.

## Fichiers d’audit

- [`makehuman_source_audit.json`](makehuman_source_audit.json)
- [`makehuman_topology_audit.json`](makehuman_topology_audit.json)
- [`makehuman_groups_decoded.txt`](makehuman_groups_decoded.txt)
- [`compare_openfoam_human_surfaces.txt`](compare_openfoam_human_surfaces.txt)
- [`locate_openfoam_quality_sets.txt`](locate_openfoam_quality_sets.txt)
- [`meshio_body_only_surfaceCheck.log`](../../openfoam_runs/human_mesh_variants/meshio_body_only_surfaceCheck.log)
- [`refined_Allrun.log`](../../openfoam_runs/meshio_body_only_refined_case/refined_Allrun.log)
